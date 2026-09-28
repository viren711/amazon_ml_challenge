"""Business entity resolution, v2.

Stages (all processed one country at a time to bound peak RAM):
  1. load + normalize + hash-tokenize every record (multiprocess)
  2. blocking: IDF-weighted sparse retrieval -- each S1 record scores every
     pool record by the summed IDF of shared name/address tokens (sparse
     matmul, multiprocess over S1 row blocks), top-K kept
  3. features: blocking scores, rank / reverse-rank group features, IDF
     coverage, rapidfuzz string similarities (multithreaded cpdist)
  4. LightGBM matcher + threshold tuned on validation macro F0.5

Usage:
  python3 er.py diag    --data-dir <train_dir>
  python3 er.py train   --data-dir <train_dir> --out-dir <artifacts_dir>
  python3 er.py predict --data-dir <test_dir> --artifacts-dir <artifacts_dir> --out-dir <output_dir>
"""
import argparse
import gc
import json
import multiprocessing as mp
import os
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import scipy.sparse as sp
import numba
from numba import njit, prange
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import HashingVectorizer

NJOBS = os.cpu_count() or 4
# memory-latency-bound kernel / GBDT: SMT siblings only add cache pressure
PHYS = max(1, NJOBS // 2)
NFEAT = 1 << 24
READ_CHUNK = 100_000
FEAT_BATCH = 2_000_000
DEFAULT_K = 30
DEFAULT_KN = 10
DEFAULT_CAP_N = 2000
DEFAULT_CAP_A = 20000
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


# ----------------------------------------------------------------------------
# normalization / tokenization
# ----------------------------------------------------------------------------
_NONWORD = re.compile(r"(?:[^\w]|_)+")
_IDTAG = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)")
_WS_COMMA = re.compile(r"[\s,;]+")
_DIGITS = re.compile(r"\d+")

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "arp", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "goa", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "telangana": "ts", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "jammu and kashmir": "jk",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
}
ADDR_WORDS = {
    "street": "st", "road": "rd", "avenue": "ave", "av": "ave", "drive": "dr", "lane": "ln",
    "boulevard": "blvd", "court": "ct", "place": "pl", "terrace": "ter", "circle": "cir",
    "parkway": "pkwy", "highway": "hwy", "suite": "ste", "apartment": "apt", "floor": "fl",
    "north": "n", "south": "s", "east": "e", "west": "w", "number": "no", "nagar": "ngr",
}
NAME_WORDS = {
    "private": "pvt", "limited": "ltd", "corporation": "corp", "company": "co",
    "incorporated": "inc", "services": "service", "enterprises": "enterprise",
}
NAME_STOP = frozenset(
    "ltd pvt llc llp inc corp co the smt shri sri ms m s and of com www in net org".split())

_PHRASES = {}
_SINGLE = dict(ADDR_WORDS)
for _d in (US_STATES, IN_STATES):
    for _k, _v in _d.items():
        _w = tuple(_k.split())
        if len(_w) == 1:
            _SINGLE[_w[0]] = _v
        else:
            _PHRASES[_w] = _v
_PHRASE_FIRST = {k[0] for k in _PHRASES}


def _build_indic_tables():
    """Indic (U+0900-U+0DFF) char -> rough Latin consonant fragment, derived
    from Unicode character names ("DEVANAGARI LETTER TTHA" -> "tth"); vowels
    dropped, anusvara -> n. Also maps every Indic digit to ASCII."""
    tr, dig = {}, {}
    for cp in range(0x0900, 0x0E00):
        ch = chr(cp)
        nm = unicodedata.name(ch, "")
        if not nm:
            continue
        rest = nm.split(" ", 1)[1] if " " in nm else ""
        d = unicodedata.digit(ch, None)
        if d is not None:
            tr[cp] = dig[cp] = str(d)
        elif rest.startswith("LETTER "):
            v = rest[7:]
            if v.startswith("VOCALIC "):
                tr[cp] = v.split()[1][0].lower()
            elif set(v.replace(" ", "")) <= set("AEIOU") or v.split()[0] in ("SHORT", "CANDRA"):
                tr[cp] = ""
            else:
                frag = v.lower().replace(" ", "")
                tr[cp] = frag[:-1] if frag.endswith("a") else frag
        elif "VOWEL SIGN VOCALIC" in rest:
            tr[cp] = rest.split()[-1][0].lower()
        elif "ANUSVARA" in rest or "CANDRABINDU" in rest:
            tr[cp] = "n"
        elif "SIGN" in rest or "VIRAMA" in rest or "NUKTA" in rest or unicodedata.category(ch)[0] == "M":
            tr[cp] = ""
        else:
            tr[cp] = " "
    return tr, dig


_INDIC_TR, _DIGIT_TR = _build_indic_tables()
_SKEL_TR = str.maketrans("cqwzfdgb", "kkvjptkp", "aeiouyh")
_SOFT_C = re.compile(r"c(?=[eiy])")
_REPEAT = re.compile(r"(.)\1+")
_DIGRAPHS = (("tion", "shan"), ("x", "ks"), ("ck", "k"), ("ch", "c"), ("sh", "s"), ("ph", "p"),
             ("th", "t"), ("dh", "d"), ("kh", "k"), ("gh", "g"), ("bh", "b"), ("jh", "j"), ("zh", "l"))


def _strip_marks(s):
    return "".join(c for c in s if not unicodedata.category(c).startswith("M"))


def fold(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s.lower())
    if not s.isascii():
        s = _strip_marks(s.translate(_DIGIT_TR))
    return s


def skel(t):
    """Script-independent consonant skeleton: 'eastern' and 'ईस्टर्न' -> 'strn'."""
    t = t.replace("tion", "shan").replace("x", "ks")
    t = _SOFT_C.sub("s", t)
    for a, b in _DIGRAPHS[2:]:
        if a in t:
            t = t.replace(a, b)
    return _REPEAT.sub(r"\1", t.translate(_SKEL_TR))


def name_skeletons(raw):
    s = unicodedata.normalize("NFKD", raw.lower())
    if not s.isascii():
        s = _strip_marks(s.translate(_INDIC_TR))
    s = _IDTAG.sub(" ", s)
    out = []
    for t in _NONWORD.sub(" ", s).split():
        if t.isascii() and not t.isdigit():
            k = skel(t)
            if len(k) >= 2:
                out.append(k)
    return out


SKEL_STOP = frozenset(skel(w) for w in (
    "private pvt limited ltd llp llc inc incorporated corporation corp company co "
    "services service enterprises enterprise the and shri sri smt").split())


def name_tokens(raw):
    s = _IDTAG.sub(" ", fold(raw)).replace("&", " and ")
    return [NAME_WORDS.get(t, t) for t in _NONWORD.sub(" ", s).split()]


def _map_addr(toks):
    out, i, n = [], 0, len(toks)
    while i < n:
        t = toks[i]
        if t in _PHRASE_FIRST:
            hit = None
            for L in (3, 2):
                if i + L <= n:
                    hit = _PHRASES.get(tuple(toks[i:i + L]))
                    if hit:
                        out.append(hit)
                        i += L
                        break
            if hit:
                continue
        out.append(_SINGLE.get(t, t))
        i += 1
    return out


ADDR_BIGRAMS = os.environ.get("ER_ADDR_BIGRAMS", "0") == "1"


def addr_tokens(raw):
    s = fold(raw).replace("&", " and ")
    toks = _map_addr(_NONWORD.sub(" ", s).split())
    compact = []
    if ADDR_BIGRAMS:
        compact.extend(f"~{a}_{b}" for a, b in zip(toks, toks[1:]))
    for piece in _WS_COMMA.split(s):
        if any(c.isdigit() for c in piece):
            subs = _NONWORD.sub(" ", piece).split()
            if len(subs) > 1:
                compact.append("#" + "".join(subs))
    return toks, compact


def _identity(x):
    return x


_HV = HashingVectorizer(n_features=NFEAT, analyzer=_identity, alternate_sign=False,
                        norm=None, binary=True, dtype=np.float32)


def _prep_chunk(df):
    """Worker: normalize + hash-tokenize one chunk of records."""
    n_list, a_list, n_tok, a_tok, nasc, s_list = [], [], [], [], [], []
    for raw in df["business_name"].tolist():
        t = name_tokens(raw)
        n_list.append(" ".join(t))
        core = [x for x in t if x not in NAME_STOP]
        if len(core) >= 2:
            t = t + ["#" + "".join(core)]
        sk = [k for k in name_skeletons(raw) if k not in SKEL_STOP]
        s_list.append(" ".join(sk))
        n_tok.append(t + ["%" + k for k in sk])
        nasc.append(0.0 if raw.isascii() else 1.0)
    for raw in df["business_address"].tolist():
        toks, comp = addr_tokens(raw)
        a_list.append(" ".join(toks))
        a_tok.append(toks + comp)
    Xn = _HV.transform(n_tok).tocsr()
    Xa = _HV.transform(a_tok).tocsr()
    return {
        "id": df["entity_id"].str.slice(3).astype(np.int64).to_numpy(),
        "nn": np.array(n_list, dtype=object),
        "na": np.array(a_list, dtype=object),
        "ns": np.array(s_list, dtype=object),
        "nasc": np.array(nasc, dtype=np.float32),
        "nlen": np.fromiter(map(len, n_list), dtype=np.float32, count=len(n_list)),
        "alen": np.fromiter(map(len, a_list), dtype=np.float32, count=len(a_list)),
        # binary matrices stored as (indptr, indices) only; data is implicitly 1
        "Xn": (Xn.indptr.astype(np.int64), Xn.indices.astype(np.int32)),
        "Xa": (Xa.indptr.astype(np.int64), Xa.indices.astype(np.int32)),
    }


def _cat_csr(parts):
    ptrs, inds, off = [np.zeros(1, np.int64)], [], 0
    for indptr, indices in parts:
        ptrs.append(indptr[1:] + off)
        inds.append(indices)
        off += len(indices)
    return np.concatenate(ptrs), (np.concatenate(inds) if inds else np.zeros(0, np.int32))


def _cat_recs(parts, src):
    parts = [p for p in parts if len(p["id"])]
    if not parts:
        return None
    rec = {k: np.concatenate([p[k] for p in parts]) for k in ("id", "nn", "na", "ns", "nasc", "nlen", "alen")}
    rec["Xn"] = _cat_csr([p["Xn"] for p in parts])
    rec["Xa"] = _cat_csr([p["Xa"] for p in parts])
    rec["src"] = np.full(len(rec["id"]), src, np.int8)
    return rec


def concat_recs(a, b):
    if a is None:
        return b
    if b is None:
        return a
    rec = {k: np.concatenate([a[k], b[k]]) for k in ("id", "nn", "na", "ns", "nasc", "nlen", "alen", "src")}
    rec["Xn"] = _cat_csr([a["Xn"], b["Xn"]])
    rec["Xa"] = _cat_csr([a["Xa"], b["Xa"]])
    return rec


def list_countries(paths):
    cs = set()
    for p in paths:
        for ch in pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False,
                              usecols=["country"], chunksize=2_000_000):
            cs.update(ch["country"].unique().tolist())
    return sorted(cs)


_TPOOL = None


def _tpool():
    # forkserver: workers are forked from a small server process, never from
    # this (large, numba-threaded) process -- fork-safe and no COW blowup.
    global _TPOOL
    if _TPOOL is None:
        _TPOOL = mp.get_context("forkserver").Pool(NJOBS, maxtasksperchild=25)
    return _TPOOL


def load_country(path, country, src):
    """Read one source TSV, keep rows of `country`, tokenize in parallel."""
    parts = []
    pool = _tpool()
    reader = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=READ_CHUNK)
    done = False
    while not done:
        wave = []
        while len(wave) < NJOBS * 2:
            ch = next(reader, None)
            if ch is None:
                done = True
                break
            ch = ch[ch["country"] == country]
            if len(ch):
                wave.append(ch)
        if wave:
            parts.extend(pool.map(_prep_chunk, wave, chunksize=1))
        del wave
    return _cat_recs(parts, src)


def take(rec, sel):
    """Row subset of a record dict, including the (indptr, indices) matrices."""
    out = {k: v[sel] for k, v in rec.items() if not isinstance(v, tuple)}
    for k in ("Xn", "Xa"):
        indptr, indices = rec[k]
        cnt = np.diff(indptr)[sel]
        ptr = np.zeros(len(sel) + 1, np.int64)
        np.cumsum(cnt, out=ptr[1:])
        src_pos = np.repeat(indptr[:-1][sel] - ptr[:-1], cnt) + np.arange(ptr[-1])
        out[k] = (ptr, indices[src_pos])
    return out


# ----------------------------------------------------------------------------
# blocking
# ----------------------------------------------------------------------------
def _weighted(X, idf):
    """New CSR with data=idf[token], zero-idf tokens dropped. Never mutates X:
    scipy's eliminate_zeros() works in place on arrays shared with X."""
    indptr, indices = X
    w = idf[indices]
    keep = w > 0
    ck = np.zeros(len(indices) + 1, np.int64)
    np.cumsum(keep, out=ck[1:])
    return sp.csr_matrix((w[keep], indices[keep], ck[indptr]), shape=(len(indptr) - 1, NFEAT))


@njit(cache=True)
def _select_top(v, ix, n, k):
    """In-place quickselect on v[:n] (ix permuted alongside) so that the
    last k slots hold the k largest values."""
    target = n - k
    lo, hi = 0, n - 1
    while lo < hi:
        pivot = v[(lo + hi) >> 1]
        i, j = lo, hi
        while i <= j:
            while v[i] < pivot:
                i += 1
            while v[j] > pivot:
                j -= 1
            if i <= j:
                tv = v[i]
                v[i] = v[j]
                v[j] = tv
                ti = ix[i]
                ix[i] = ix[j]
                ix[j] = ti
                i += 1
                j -= 1
        if target <= j:
            hi = j
        elif target >= i:
            lo = i
        else:
            break


@njit(parallel=True)
def _topk_kernel(qn_ptr, qn_idx, qn_w, qa_ptr, qa_idx, qa_w,
                 pn_ptr, pn_doc, pa_ptr, pa_doc, p_totn, N, K, Kn, n_chunks):
    """Per query: accumulate summed IDF of shared tokens over the inverted
    index, keep the top K by total score plus the top Kn of the rest by
    name-only score (catches pool records with empty/truncated addresses).
    Scratch buffers are allocated once per thread: per-query allocation
    inside the parallel loop serializes all threads on mmap."""
    nq = len(qn_ptr) - 1
    KT = K + Kn
    out_c = np.full(nq * KT, -1, np.int32)
    out_s = np.zeros(nq * KT, np.float32)
    out_n = np.zeros(nq * KT, np.float32)
    out_t = np.zeros(nq, np.int32)
    T = numba.get_num_threads()
    acc_all = np.zeros((T, N), np.float32)
    accn_all = np.zeros((T, N), np.float32)
    touched_all = np.empty((T, N), np.int32)
    vals_all = np.empty((T, N), np.float32)
    cand_all = np.empty((T, N), np.int32)
    step = (nq + n_chunks - 1) // n_chunks
    for c in prange(n_chunks):
        tid = numba.get_thread_id()
        acc = acc_all[tid]
        accn = accn_all[tid]
        touched = touched_all[tid]
        vals = vals_all[tid]
        cand = cand_all[tid]
        lo = c * step
        hi = min(nq, lo + step)
        for i in range(lo, hi):
            cnt = 0
            for t in range(qn_ptr[i], qn_ptr[i + 1]):
                tok = qn_idx[t]
                w = qn_w[t]
                for e in range(pn_ptr[tok], pn_ptr[tok + 1]):
                    d = pn_doc[e]
                    if acc[d] == 0.0:
                        touched[cnt] = d
                        cnt += 1
                    acc[d] += w
                    accn[d] += w
            for t in range(qa_ptr[i], qa_ptr[i + 1]):
                tok = qa_idx[t]
                w = qa_w[t]
                for e in range(pa_ptr[tok], pa_ptr[tok + 1]):
                    d = pa_doc[e]
                    if acc[d] == 0.0:
                        touched[cnt] = d
                        cnt += 1
                    acc[d] += w
            out_t[i] = cnt
            base = i * KT
            first = 0
            if cnt > K:
                for m in range(cnt):
                    vals[m] = acc[touched[m]]
                _select_top(vals, touched, cnt, K)
                first = cnt - K
            k = 0
            for m in range(first, cnt):
                d = touched[m]
                out_c[base + k] = d
                out_s[base + k] = acc[d]
                out_n[base + k] = accn[d]
                k += 1
            for m in range(first, cnt):
                acc[touched[m]] = -acc[touched[m]]
            if Kn > 0:
                ne = 0
                for m in range(cnt):
                    d = touched[m]
                    if acc[d] > 0.0 and accn[d] > 0.0:
                        # tie-break equal name scores by how much of the pool
                        # record's own name the shared tokens explain
                        cov = min(accn[d] / max(p_totn[d], 1e-6), 1.0)
                        vals[ne] = accn[d] * (0.5 + 0.5 * cov)
                        cand[ne] = d
                        ne += 1
                f2 = 0
                if ne > Kn:
                    _select_top(vals, cand, ne, Kn)
                    f2 = ne - Kn
                b2 = base + K
                k2 = 0
                for m in range(f2, ne):
                    d = cand[m]
                    out_c[b2 + k2] = d
                    out_s[b2 + k2] = acc[d]
                    out_n[b2 + k2] = accn[d]
                    k2 += 1
            for m in range(cnt):
                d = touched[m]
                acc[d] = 0.0
                accn[d] = 0.0
    return out_c, out_s, out_n, out_t


def _idf(X, n_docs, cap):
    df = np.bincount(X[1], minlength=NFEAT)
    idf = np.zeros(NFEAT, np.float32)
    ok = (df > 0) & (df <= cap)
    idf[ok] = np.log((n_docs + 1.0) / (df[ok] + 1.0)).astype(np.float32)
    return idf


def _postings(X, idf, order):
    """Inverted index token -> pool docs, docs renumbered into locality
    `order` (each posting list comes out ascending: forward memory walks)."""
    M = _weighted(X, idf)
    tot = np.asarray(M.sum(axis=1)).ravel().astype(np.float32)
    M = M[order]
    T = M.T.tocsr()
    del M
    return T.indptr.astype(np.int64), T.indices.astype(np.int32), tot


def _locality_order(rec):
    """Order that puts records of the same locality next to each other
    (sorted by reversed address: state, city, street, ...). Applied to both
    pool doc ids and query processing order, so each thread's accumulator
    working set stays small: the kernel is memory-latency bound otherwise."""
    keys = np.array([" ".join(a.split()[::-1]) for a in rec["na"]], dtype=object)
    return np.argsort(keys, kind="stable")


def block(q, p, K, cap_n, cap_a, Kn=0):
    """Returns pairs dict (qi, pj, s, sn, sa, qtouch) and per-record IDF totals."""
    N = len(p["id"])
    nq = len(q["id"])
    t0 = time.time()
    order = _locality_order(p)
    rank = np.empty(N, np.int32)
    rank[order] = np.arange(N, dtype=np.int32)
    idf_n = _idf(p["Xn"], N, cap_n)
    idf_a = _idf(p["Xa"], N, cap_a)
    tots = {}
    pn_ptr, pn_doc, tots["pn"] = _postings(p["Xn"], idf_n, order)
    pa_ptr, pa_doc, tots["pa"] = _postings(p["Xa"], idf_a, order)
    p_totn = np.empty(N, np.float32)
    p_totn[rank] = tots["pn"]
    del rank
    Qn = _weighted(q["Xn"], idf_n)
    Qa = _weighted(q["Xa"], idf_a)
    tots["qn"] = np.asarray(Qn.sum(axis=1)).ravel().astype(np.float32)
    tots["qa"] = np.asarray(Qa.sum(axis=1)).ravel().astype(np.float32)
    qperm = _locality_order(q)
    Qn = Qn[qperm]
    Qa = Qa[qperm]
    del idf_n, idf_a
    gc.collect()
    t1 = time.time()
    numba.set_num_threads(PHYS)
    out_c, out_s, out_n, out_t = _topk_kernel(
        Qn.indptr.astype(np.int64), Qn.indices.astype(np.int32), Qn.data.astype(np.float32),
        Qa.indptr.astype(np.int64), Qa.indices.astype(np.int32), Qa.data.astype(np.float32),
        pn_ptr, pn_doc, pa_ptr, pa_doc, p_totn, N, K, Kn, PHYS * 32)
    del Qn, Qa, pn_ptr, pn_doc, pa_ptr, pa_doc
    log(f"  [block] nq={nq} N={N} index={t1 - t0:.1f}s kernel={time.time() - t1:.1f}s")
    m = out_c >= 0
    qpos = np.repeat(np.arange(nq, dtype=np.int32), K + Kn)[m]
    srt = np.argsort(qperm[qpos], kind="stable")
    qpos = qpos[srt]
    pr = {"qi": qperm[qpos].astype(np.int32), "pj": order[out_c[m][srt]].astype(np.int32),
          "s": out_s[m][srt], "sn": out_n[m][srt], "qtouch": out_t[qpos].astype(np.float32)}
    pr["sa"] = pr["s"] - pr["sn"]
    del out_c, out_s, out_n, m, order, qperm, qpos, srt
    gc.collect()
    return pr, tots


# ----------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------
FEATURES = [
    "s", "sn", "sa", "s_rel", "s_gap", "q_margin", "qrank", "qcnt", "qnear", "prank", "pcnt", "s_prel",
    "qtouch",
    "covq_n", "covp_n", "covq_a", "covp_a", "tq_n", "tp_n", "tq_a", "tp_a",
    "src", "nasc_q", "nasc_p", "nlen_q", "nlen_p", "alen_q", "alen_p",
    "n_ratio", "n_tset", "n_tsort", "n_partial", "n_jw", "nc_ratio", "nc_partial", "sk_ratio", "sk_tset",
    "a_ratio", "a_tset", "a_tsort", "a_partial", "d_tset", "d_eq", "d_empty",
]


def group_feats(pr):
    df = pd.DataFrame({"qi": pr["qi"], "pj": pr["pj"], "s": pr["s"]})
    gq = df.groupby("qi", sort=False)["s"]
    qmax = gq.transform("max").to_numpy(np.float32)
    qrank = gq.rank(ascending=False, method="first").to_numpy(np.float32)
    qcnt = gq.transform("size").to_numpy(np.float32)
    sec = pd.Series(np.where(qrank == 2, pr["s"], 0.0)).groupby(df["qi"].to_numpy(), sort=False)
    qsec = sec.transform("max").to_numpy(np.float32)
    near = pd.Series((pr["s"] >= 0.8 * qmax).astype(np.float32))
    qnear = near.groupby(df["qi"].to_numpy(), sort=False).transform("sum").to_numpy(np.float32)
    gp = df.groupby("pj", sort=False)["s"]
    pmax = gp.transform("max").to_numpy(np.float32)
    prank = gp.rank(ascending=False, method="first").to_numpy(np.float32)
    pcnt = gp.transform("size").to_numpy(np.float32)
    return {"qmax": qmax, "qrank": qrank, "qcnt": qcnt, "qsec": qsec, "qnear": qnear,
            "pmax": pmax, "prank": prank, "pcnt": pcnt}


def _digits(arr):
    return np.array([" ".join(sorted(set(_DIGITS.findall(x)))) for x in arr], dtype=object)


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def assemble(q, p, pr, gf, tots, idx):
    qi = pr["qi"][idx]
    pj = pr["pj"][idx]
    s, sn, sa = pr["s"][idx], pr["sn"][idx], pr["sa"][idx]
    eps = np.float32(1e-6)
    qmax = gf["qmax"][idx]
    c = {
        "s": s, "sn": sn, "sa": sa,
        "s_rel": s / np.maximum(qmax, eps), "s_gap": qmax - s,
        "q_margin": (qmax - gf["qsec"][idx]) / np.maximum(qmax, eps),
        "qrank": gf["qrank"][idx], "qcnt": gf["qcnt"][idx], "qnear": gf["qnear"][idx],
        "prank": gf["prank"][idx], "pcnt": gf["pcnt"][idx],
        "s_prel": s / np.maximum(gf["pmax"][idx], eps),
        "qtouch": pr["qtouch"][idx],
        "tq_n": tots["qn"][qi], "tp_n": tots["pn"][pj], "tq_a": tots["qa"][qi], "tp_a": tots["pa"][pj],
        "src": p["src"][pj].astype(np.float32),
        "nasc_q": q["nasc"][qi], "nasc_p": p["nasc"][pj],
        "nlen_q": q["nlen"][qi], "nlen_p": p["nlen"][pj],
        "alen_q": q["alen"][qi], "alen_p": p["alen"][pj],
    }
    c["covq_n"] = sn / np.maximum(c["tq_n"], eps)
    c["covp_n"] = sn / np.maximum(c["tp_n"], eps)
    c["covq_a"] = sa / np.maximum(c["tq_a"], eps)
    c["covp_a"] = sa / np.maximum(c["tp_a"], eps)

    qn, pn = q["nn"][qi], p["nn"][pj]
    c["n_ratio"] = _cp(qn, pn, fuzz.ratio)
    c["n_tset"] = _cp(qn, pn, fuzz.token_set_ratio)
    c["n_tsort"] = _cp(qn, pn, fuzz.token_sort_ratio)
    c["n_partial"] = _cp(qn, pn, fuzz.partial_ratio)
    c["n_jw"] = _cp(qn, pn, JaroWinkler.normalized_similarity)
    qnc = np.array([x.replace(" ", "") for x in qn], dtype=object)
    pnc = np.array([x.replace(" ", "") for x in pn], dtype=object)
    c["nc_ratio"] = _cp(qnc, pnc, fuzz.ratio)
    c["nc_partial"] = _cp(qnc, pnc, fuzz.partial_ratio)
    del qn, pn, qnc, pnc
    qs_, ps_ = q["ns"][qi], p["ns"][pj]
    c["sk_ratio"] = _cp(qs_, ps_, fuzz.ratio)
    c["sk_tset"] = _cp(qs_, ps_, fuzz.token_set_ratio)
    del qs_, ps_

    qa, pa = q["na"][qi], p["na"][pj]
    c["a_ratio"] = _cp(qa, pa, fuzz.ratio)
    c["a_tset"] = _cp(qa, pa, fuzz.token_set_ratio)
    c["a_tsort"] = _cp(qa, pa, fuzz.token_sort_ratio)
    c["a_partial"] = _cp(qa, pa, fuzz.partial_ratio)
    qd, pd_ = _digits(qa), _digits(pa)
    del qa, pa
    c["d_tset"] = _cp(qd, pd_, fuzz.token_set_ratio)
    q_empty = qd == ""
    p_empty = pd_ == ""
    c["d_eq"] = ((qd == pd_) & ~q_empty).astype(np.float32)
    c["d_empty"] = (q_empty.astype(np.float32) + 2 * p_empty.astype(np.float32))
    return np.column_stack([np.asarray(c[f], dtype=np.float32) for f in FEATURES])


def assemble_batched(q, p, pr, gf, tots, idx):
    out = np.empty((len(idx), len(FEATURES)), np.float32)
    for st in range(0, len(idx), FEAT_BATCH):
        out[st:st + FEAT_BATCH] = assemble(q, p, pr, gf, tots, idx[st:st + FEAT_BATCH])
    return out


# ----------------------------------------------------------------------------
# ground truth / metric
# ----------------------------------------------------------------------------
KEY_Q = np.int64(4_000_000_000)
KEY_S = np.int64(2_000_000_000)


def pair_keys(qid, src, pid):
    return qid.astype(np.int64) * KEY_Q + (src.astype(np.int64) - 2) * KEY_S + pid.astype(np.int64)


def load_gt(path):
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    s1 = gt["source1_entity_id"].str.slice(3).astype(np.int64).to_numpy()
    lists = gt["matched_entity_ids"].str.split(",")
    cnt = lists.map(lambda l: sum(1 for x in l if x)).to_numpy(np.int32)
    ex = pd.DataFrame({"q": s1, "m": lists}).explode("m")
    ex = ex[ex["m"].notna() & (ex["m"] != "")]
    src = ex["m"].str.slice(1, 2).astype(np.int64).to_numpy()
    pid = ex["m"].str.slice(3).astype(np.int64).to_numpy()
    keys = np.sort(pair_keys(ex["q"].to_numpy(np.int64), src, pid))
    order = np.argsort(s1)
    return keys, s1[order], cnt[order]


def is_match(keys, gt_keys):
    pos = np.searchsorted(gt_keys, keys)
    pos[pos >= len(gt_keys)] = 0
    return gt_keys[pos] == keys


def true_counts(qids, gt_q, gt_cnt):
    pos = np.searchsorted(gt_q, qids)
    pos[pos >= len(gt_q)] = 0
    return np.where(gt_q[pos] == qids, gt_cnt[pos], 0)


def macro_f05(n_q, true_cnt, qloc, y, pred):
    """qloc: per-pair index into 0..n_q-1; true_cnt: per query true match count."""
    P = np.bincount(qloc[pred], minlength=n_q)
    TP = np.bincount(qloc[pred & y], minlength=n_q)
    T = true_cnt
    f = np.zeros(n_q)
    both_empty = (T == 0) & (P == 0)
    f[both_empty] = 1.0
    ok = (T > 0) & (P > 0) & (TP > 0)
    prec = TP[ok] / P[ok]
    rec = TP[ok] / T[ok]
    f[ok] = 1.25 * prec * rec / (0.25 * prec + rec)
    return float(f.mean())


# ----------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------
def _paths(d, prefix):
    return [f"{d}/{prefix}_source{i}.tsv" for i in (1, 2, 3)]


def _load_all(paths, country):
    q = load_country(paths[0], country, 1)
    p = concat_recs(load_country(paths[1], country, 2), load_country(paths[2], country, 3))
    return q, p


def _drop_mats(*recs):
    for r in recs:
        if r is not None:
            r.pop("Xn", None)
            r.pop("Xa", None)
    gc.collect()


def _show_misses(qs, p, pr, y, gt_keys, T, n_show):
    """Print true matches that blocking failed to retrieve."""
    found = np.bincount(pr["qi"][y], minlength=len(qs["id"]))
    pkey = (p["src"].astype(np.int64) - 2) * KEY_S + p["id"]
    order = np.argsort(pkey)
    shown = 0
    for i in np.flatnonzero(found < T)[:n_show]:
        lo = np.searchsorted(gt_keys, qs["id"][i] * KEY_Q)
        hi = np.searchsorted(gt_keys, (qs["id"][i] + 1) * KEY_Q)
        got = set(pr["pj"][(pr["qi"] == i) & y].tolist())
        print(flush=True)
        print(f"  MISS S1 {qs['nn'][i]} | {qs['na'][i]}")
        for k in gt_keys[lo:hi] - qs["id"][i] * KEY_Q:
            j = order[np.searchsorted(pkey[order], k)]
            print(f"     {'ok ' if j in got else 'XX '} {p['nn'][j]} | {p['na'][j]}", flush=True)
        shown += 1


def cmd_diag(args):
    paths = _paths(args.data_dir, "train")
    gt_keys, gt_q, gt_cnt = load_gt(f"{args.data_dir}/train_ground_truth.tsv")
    configs = [tuple(int(v) for v in c.split(":")) for c in args.configs.split(",")]
    countries = args.countries.split(",") if args.countries else list_countries(paths[:1])
    for country in countries:
        q, p = _load_all(paths, country)
        log(f"[{country}] loaded S1={len(q['id'])} pool={0 if p is None else len(p['id'])}")
        rng = np.random.RandomState(0)
        sel = np.flatnonzero(rng.rand(len(q["id"])) < args.frac)
        qs = take(q, sel)
        T = true_counts(qs["id"], gt_q, gt_cnt)
        for ci, cfg_ in enumerate(configs):
            cap_n, cap_a, K = cfg_[:3]
            Kn = cfg_[3] if len(cfg_) > 3 else 0
            t = time.time()
            pr, _ = block(qs, p, K, cap_n, cap_a, Kn)
            y = is_match(pair_keys(qs["id"][pr["qi"]], p["src"][pr["pj"]], p["id"][pr["pj"]]), gt_keys)
            gf = group_feats(pr)
            msg = []
            for k in sorted({5, 10, 20, K, K + Kn}):
                m = gf["qrank"] <= k
                found = np.bincount(pr["qi"][m & y], minlength=len(sel))
                rec = found.sum() / max(1, T.sum())
                f = np.where(T == 0, 1.0, 1.25 * (found / np.maximum(T, 1)) / (0.25 + found / np.maximum(T, 1)))
                msg.append(f"R@{k}={rec:.4f} oracleF@{k}={f.mean():.4f}")
            log(f"[{country}] cap_n={cap_n} cap_a={cap_a} K={K} Kn={Kn} n_q={len(sel)} pairs={len(pr['qi'])} "
                f"t={time.time() - t:.1f}s " + " ".join(msg))
            if args.show_misses and ci == len(configs) - 1:
                _show_misses(qs, p, pr, y, gt_keys, T, args.show_misses)
        del q, p, qs
        gc.collect()


def cmd_train(args):
    import lightgbm as lgb
    paths = _paths(args.data_dir, "train")
    gt_keys, gt_q, gt_cnt = load_gt(f"{args.data_dir}/train_ground_truth.tsv")
    log(f"[gt] pairs={len(gt_keys)} s1={len(gt_q)}")
    Xtr, ytr, Xva, yva, qva, Tva = [], [], [], [], [], []
    n_va_total = 0
    stats = {}
    for country in list_countries(paths[:1]):
        q, p = _load_all(paths, country)
        nq = len(q["id"])
        log(f"[{country}] loaded S1={nq} pool={0 if p is None else len(p['id'])}")
        if p is None or nq == 0:
            continue
        pr, tots = block(q, p, args.K, args.cap_n, args.cap_a, args.Kn)
        _drop_mats(q, p)
        log(f"[{country}] blocked pairs={len(pr['qi'])} avg={len(pr['qi']) / nq:.1f}")
        gf = group_feats(pr)

        rng = np.random.RandomState(42)
        u = rng.rand(nq)
        is_val = u < args.val_frac
        va_sel = u < args.val_frac * args.val_sample
        tr_sel = (~is_val) & (rng.rand(nq) < args.train_sample)
        qloc_va = np.full(nq, -1, np.int64)
        va_idx_q = np.flatnonzero(va_sel)
        qloc_va[va_idx_q] = np.arange(len(va_idx_q)) + n_va_total
        n_va_total += len(va_idx_q)
        Tva.append(true_counts(q["id"][va_idx_q], gt_q, gt_cnt))

        y_all = is_match(pair_keys(q["id"][pr["qi"]], p["src"][pr["pj"]], p["id"][pr["pj"]]), gt_keys)
        T_all = true_counts(q["id"], gt_q, gt_cnt)
        stats[country] = {"n_s1": int(nq), "pairs": int(len(pr["qi"])),
                          "pair_recall": float(y_all.sum() / max(1, T_all.sum()))}
        log(f"[{country}] pair recall={stats[country]['pair_recall']:.4f}")

        tr_idx = np.flatnonzero(tr_sel[pr["qi"]])
        va_idx = np.flatnonzero(va_sel[pr["qi"]])
        Xtr.append(assemble_batched(q, p, pr, gf, tots, tr_idx))
        ytr.append(y_all[tr_idx])
        log(f"[{country}] train feats {Xtr[-1].shape}")
        Xva.append(assemble_batched(q, p, pr, gf, tots, va_idx))
        yva.append(y_all[va_idx])
        qva.append(qloc_va[pr["qi"][va_idx]])
        log(f"[{country}] val feats {Xva[-1].shape}")
        del q, p, pr, gf, tots, y_all
        gc.collect()

    Xtr = np.concatenate(Xtr)
    ytr = np.concatenate(ytr)
    Xva = np.concatenate(Xva)
    yva = np.concatenate(yva)
    qva = np.concatenate(qva)
    Tva = np.concatenate(Tva)
    log(f"[data] train {Xtr.shape} pos={ytr.sum()} | val {Xva.shape} pos={yva.sum()} val_s1={n_va_total}")
    n_train_pairs, n_val_pairs = int(Xtr.shape[0]), int(Xva.shape[0])

    params = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.5, bagging_freq=1, lambda_l2=1.0,
                  num_threads=PHYS, verbose=-1, max_bin=255)
    dtr = lgb.Dataset(Xtr, ytr.astype(np.float32), feature_name=FEATURES, free_raw_data=True)
    es_n = min(len(yva), 3_000_000)
    dva = lgb.Dataset(Xva[:es_n], yva[:es_n].astype(np.float32), reference=dtr)
    t = time.time()
    booster = lgb.train(params, dtr, num_boost_round=args.rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)])
    log(f"[lgbm] trained {booster.best_iteration} rounds in {time.time() - t:.1f}s")
    del dtr, dva, Xtr, ytr
    gc.collect()

    prob = booster.predict(Xva, num_iteration=booster.best_iteration, num_threads=PHYS)
    best = (0.0, 0.5)
    for thr in np.arange(0.05, 0.96, 0.01):
        f = macro_f05(n_va_total, Tva, qva, yva, prob >= thr)
        if f > best[0]:
            best = (f, float(thr))
    oracle = macro_f05(n_va_total, Tva, qva, yva, yva)
    log(f"[val] macro F0.5={best[0]:.4f} at thr={best[1]:.2f} (blocking oracle={oracle:.4f})")

    imp = dict(zip(FEATURES, booster.feature_importance("gain").round(1).tolist()))
    os.makedirs(args.out_dir, exist_ok=True)
    booster.save_model(f"{args.out_dir}/lgbm.txt", num_iteration=booster.best_iteration)
    cfg = {"threshold": best[1], "val_macro_f05": best[0], "val_oracle_f05": oracle,
           "K": args.K, "Kn": args.Kn, "cap_n": args.cap_n, "cap_a": args.cap_a, "features": FEATURES,
           "blocking": stats, "n_train_pairs": n_train_pairs, "n_val_pairs": n_val_pairs,
           "n_val_s1": int(n_va_total),
           "best_iteration": booster.best_iteration, "feature_gain": imp}
    with open(f"{args.out_dir}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    log("[done]", json.dumps({k: cfg[k] for k in ("threshold", "val_macro_f05", "val_oracle_f05")}))


def _write_groups(fh, qids, qi, pid, psrc, n_q, step=100_000):
    """Write one line per query 0..n_q-1 (qi sorted ascending), in bounded chunks."""
    bounds = np.searchsorted(qi, np.arange(n_q + 1))
    for s0 in range(0, n_q, step):
        s1 = min(s0 + step, n_q)
        a, b = bounds[s0], bounds[s1]
        strs = [f"S{sr}-{x}" for sr, x in zip(psrc[a:b].tolist(), pid[a:b].tolist())]
        lines = []
        for i in range(s0, s1):
            lines.append(f"S1-{qids[i]}\t{','.join(strs[bounds[i] - a:bounds[i + 1] - a])}\n")
        fh.write("".join(lines))


def cmd_predict(args):
    import lightgbm as lgb
    with open(f"{args.artifacts_dir}/config.json") as f:
        cfg = json.load(f)
    booster = lgb.Booster(model_file=f"{args.artifacts_dir}/lgbm.txt")
    thr = cfg["threshold"]
    paths = _paths(args.data_dir, "test")
    os.makedirs(args.out_dir, exist_ok=True)
    cand_fh = open(f"{args.out_dir}/candidate_pairs.tsv", "w", encoding="utf-8")
    match_fh = open(f"{args.out_dir}/matching_results.tsv", "w", encoding="utf-8")
    cand_fh.write("source1_entity_id\tcandidate_entity_ids\n")
    match_fh.write("source1_entity_id\tmatched_entity_ids\n")
    tot_q = tot_pairs = tot_matched = 0
    for country in list_countries(paths[:1]):
        q, p = _load_all(paths, country)
        nq = len(q["id"])
        log(f"[{country}] loaded S1={nq} pool={0 if p is None else len(p['id'])}")
        tot_q += nq
        if p is None:
            e = np.zeros(0, np.int64)
            _write_groups(cand_fh, q["id"], e, e, e, nq)
            _write_groups(match_fh, q["id"], e, e, e, nq)
            continue
        pr, tots = block(q, p, cfg["K"], cfg["cap_n"], cfg["cap_a"], cfg.get("Kn", 0))
        _drop_mats(q, p)
        tot_pairs += len(pr["qi"])
        log(f"[{country}] blocked pairs={len(pr['qi'])} avg={len(pr['qi']) / nq:.1f}")
        gf = group_feats(pr)
        prob = np.empty(len(pr["qi"]), np.float32)
        for st in range(0, len(prob), FEAT_BATCH):
            idx = np.arange(st, min(st + FEAT_BATCH, len(prob)))
            X = assemble(q, p, pr, gf, tots, idx)
            prob[idx] = booster.predict(X, num_threads=PHYS)
            log(f"[{country}] scored {idx[-1] + 1}/{len(prob)}")
        del gf, tots
        order = np.lexsort((-prob, pr["qi"]))
        qi = pr["qi"][order]
        pj = pr["pj"][order]
        prob = prob[order]
        pid = p["id"][pj]
        psrc = p["src"][pj]
        _write_groups(cand_fh, q["id"], qi, pid, psrc, nq)
        m = prob >= thr
        if not args.no_exclusive:
            # every pool record belongs to at most one S1 entity (holds for
            # 100% of train ground truth): keep only its most probable claim
            idx = np.flatnonzero(m)
            o = idx[np.lexsort((-prob[idx], pj[idx]))]
            first = np.ones(len(o), bool)
            first[1:] = pj[o][1:] != pj[o][:-1]
            log(f"[{country}] exclusivity: {int((~first).sum())} of {len(o)} predicted links dropped")
            m[:] = False
            m[o[first]] = True
        keep = prob >= 0.05
        np.savez(f"{args.out_dir}/probs_{country}.npz", q=q["id"][qi[keep]], src=psrc[keep],
                 p=pid[keep], prob=prob[keep])
        _write_groups(match_fh, q["id"], qi[m], pid[m], psrc[m], nq)
        tot_matched += int(np.unique(qi[m]).size)
        log(f"[{country}] written; s1 with match={np.unique(qi[m]).size}/{nq}")
        del q, p, pr, prob, qi, pj, pid, psrc, order, m
        gc.collect()
    cand_fh.close()
    match_fh.close()
    log(f"[done] s1={tot_q} pairs={tot_pairs} avg_cand={tot_pairs / max(1, tot_q):.1f} s1_with_match={tot_matched}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diag")
    d.add_argument("--data-dir", required=True)
    d.add_argument("--frac", type=float, default=0.03)
    d.add_argument("--configs", default="2000:20000:50")
    d.add_argument("--show-misses", type=int, default=0)
    d.add_argument("--countries", default="")
    t = sub.add_parser("train")
    t.add_argument("--data-dir", required=True)
    t.add_argument("--out-dir", required=True)
    t.add_argument("--K", type=int, default=DEFAULT_K)
    t.add_argument("--Kn", type=int, default=DEFAULT_KN)
    t.add_argument("--cap-n", type=int, default=DEFAULT_CAP_N)
    t.add_argument("--cap-a", type=int, default=DEFAULT_CAP_A)
    t.add_argument("--val-frac", type=float, default=0.15)
    t.add_argument("--val-sample", type=float, default=0.3)
    t.add_argument("--train-sample", type=float, default=0.12)
    t.add_argument("--rounds", type=int, default=800)
    pz = sub.add_parser("predict")
    pz.add_argument("--data-dir", required=True)
    pz.add_argument("--artifacts-dir", required=True)
    pz.add_argument("--out-dir", required=True)
    pz.add_argument("--no-exclusive", action="store_true")
    args = ap.parse_args()
    try:
        {"diag": cmd_diag, "train": cmd_train, "predict": cmd_predict}[args.cmd](args)
    finally:
        if _TPOOL is not None:
            _TPOOL.terminate()
            _TPOOL.join()


if __name__ == "__main__":
    main()
