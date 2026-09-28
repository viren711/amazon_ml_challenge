"""Pairwise similarity feature engineering for candidate S1-vs-(S2/S3) pairs."""
import numpy as np
import pandas as pd

# See pipeline_common.py for why this is disabled: pandas' inferred "str"
# extension dtype (default on in pandas 3.x) was observed to silently corrupt
# large-scale entity_id merge joins in _lookup_join below. Repeated here
# (idempotent) so this module is safe even if imported before
# pipeline_common.
pd.set_option("future.infer_string", False)

from rapidfuzz import fuzz

from normalize import name_tokens, addr_tokens, normalized_name, normalized_addr

FEATURE_COLS = [
    "name_ratio", "name_tsort", "name_tset",
    "addr_ratio", "addr_tsort", "addr_tset",
    "name_jaccard", "addr_jaccard",
    "n_shared_name_tok", "n_shared_addr_tok",
    "country_match",
    "name_len_diff", "addr_len_diff",
    "name_len_ratio", "addr_len_ratio",
]


def build_lookup(df):
    """Precompute per-record normalized strings/token sets. Returns a DataFrame
    indexed by entity_id.

    ROOT CAUSE (finally isolated) of the "lookup drop" issue this module used
    to work around defensively: `out = pd.DataFrame({"entity_id": ...})`
    creates a fresh 0-based RangeIndex, but `df` (a `.iloc[]` slice of a
    country-filtered subset) keeps its *original*, non-contiguous row index.
    Assigning `out["norm_name"] = df["business_name"].map(...)` is an
    INDEX-ALIGNED assignment, not positional -- pandas silently NaN-fills
    every row of `out` whose index doesn't happen to also appear in `df`'s
    original index. Depending on which slice of the data a chunk is, that
    overlap ranges from partial to (mostly) none, which is exactly the
    "some pairs mysteriously unresolvable" pattern observed. Fix: every
    per-row value pulled from `df` is materialized via `.to_numpy()` (a
    plain positional array, no index) before being placed into `out`, so
    there is no index to misalign.
    """
    out = pd.DataFrame({"entity_id": df["entity_id"].to_numpy(dtype=object)})
    out["country"] = df["country"].fillna("").to_numpy(dtype=object)
    out["norm_name"] = df["business_name"].map(normalized_name).to_numpy()
    out["norm_addr"] = df["business_address"].map(normalized_addr).to_numpy()
    # Tuples, not frozensets: a persistent pool_lookup can hold token sets for
    # millions of records at once (any S1 chunk can touch a large fraction of
    # a country's pool), and a frozenset's hash-table overhead per record
    # adds up fast. Tuples are converted to sets only transiently, per
    # feature-computation batch, in _compute_features_batch.
    out["name_tok"] = df["business_name"].map(lambda s: tuple(name_tokens(s))).to_numpy()
    out["addr_tok"] = df["business_address"].map(lambda s: tuple(addr_tokens(s))).to_numpy()
    out["name_len"] = out["norm_name"].str.len()
    out["addr_len"] = out["norm_addr"].str.len()
    return out


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    u = len(a | b)
    if u == 0:
        return 1.0
    return len(a & b) / u


FEATURE_BATCH = 500_000


def _build_idx_map(lookup):
    """Build entity_id -> row-position dict once per lookup table (reused
    across every batch of a compute_features call instead of being rebuilt
    per-batch, which was measured to be a significant, avoidable cost when
    pool_lookup has many rows)."""
    idx_map = {}
    for i, eid in enumerate(lookup["entity_id"].to_numpy()):
        seid = str(eid)
        if seid not in idx_map:
            idx_map[seid] = i
    return idx_map


def _lookup_join(ids, lookup, idx_map):
    """Look up `ids` (in order) against `lookup["entity_id"]` via a
    precomputed entity_id -> row-position dict (`idx_map`, from
    `_build_idx_map`).

    NOTE: this deliberately does NOT use pd.merge. A pd.merge-based version
    was measured, on this pipeline's real data, to silently and
    deterministically drop a large (~20-45%), reproducible fraction of
    genuinely-present keys in some large batches (confirmed via a targeted
    repro: identical inputs, identical missing IDs, across repeated runs,
    with and without pandas' string-dtype inference disabled) -- root cause
    not fully isolated within the time available, but bypassing pd.merge
    entirely eliminates it. Instead: resolve every id via a plain positional
    `.iloc` take, which cannot mis-join.
    """
    row_idx = np.fromiter((idx_map.get(str(i), -1) for i in ids), dtype=np.int64, count=len(ids))
    valid_mask = row_idx >= 0
    safe_idx = np.where(valid_mask, row_idx, 0)
    out = lookup.iloc[safe_idx].reset_index(drop=True)
    if not valid_mask.all():
        out = out.copy()
        out.loc[~valid_mask, ["norm_name", "norm_addr", "country"]] = np.nan
    return out


def _compute_features_batch(s1_ids, cand_ids, s1_lookup, pool_lookup, s1_idx_map, pool_idx_map):
    n = len(s1_ids)
    a = _lookup_join(s1_ids, s1_lookup, s1_idx_map)
    b = _lookup_join(cand_ids, pool_lookup, pool_idx_map)
    if len(a) != n or len(b) != n:
        raise ValueError(
            f"lookup join changed row count: {n} pairs but got "
            f"{len(a)} s1 rows / {len(b)} pool rows back -- a duplicate "
            f"entity_id in the lookup table is the likely cause."
        )
    missing_a = a["norm_name"].isna().to_numpy()
    missing_b = b["norm_name"].isna().to_numpy()
    valid = ~(missing_a | missing_b)
    if not valid.all():
        n_missing = int((~valid).sum())
        bad_s1 = sorted(set(np.asarray(s1_ids)[missing_a][:5]))
        bad_cand = sorted(set(np.asarray(cand_ids)[missing_b][:5]))
        print(f"[compute_features WARNING] dropping {n_missing}/{n} pairs with an "
              f"id missing from its lookup table (examples: s1={bad_s1} cand={bad_cand})",
              flush=True)
        # Fill missing rows with a neutral placeholder so every array below
        # stays a fixed, aligned length; these rows are zeroed out and
        # reported via `valid` for the caller to exclude from training/output.
        a = a.copy()
        b = b.copy()
        a.loc[missing_a, ["norm_name", "norm_addr", "country"]] = ""
        b.loc[missing_b, ["norm_name", "norm_addr", "country"]] = ""
        a.loc[missing_a, ["name_len", "addr_len"]] = 0
        b.loc[missing_b, ["name_len", "addr_len"]] = 0
        empty_tuple = pd.Series([()] * n)
        a.loc[missing_a, "name_tok"] = empty_tuple[missing_a].values
        a.loc[missing_a, "addr_tok"] = empty_tuple[missing_a].values
        b.loc[missing_b, "name_tok"] = empty_tuple[missing_b].values
        b.loc[missing_b, "addr_tok"] = empty_tuple[missing_b].values

    a_name = a["norm_name"].values
    b_name = b["norm_name"].values
    a_addr = a["norm_addr"].values
    b_addr = b["norm_addr"].values

    name_ratio = np.fromiter((fuzz.ratio(x, y) for x, y in zip(a_name, b_name)), dtype=np.float32, count=n)
    name_tsort = np.fromiter((fuzz.token_sort_ratio(x, y) for x, y in zip(a_name, b_name)), dtype=np.float32, count=n)
    name_tset = np.fromiter((fuzz.token_set_ratio(x, y) for x, y in zip(a_name, b_name)), dtype=np.float32, count=n)
    addr_ratio = np.fromiter((fuzz.ratio(x, y) for x, y in zip(a_addr, b_addr)), dtype=np.float32, count=n)
    addr_tsort = np.fromiter((fuzz.token_sort_ratio(x, y) for x, y in zip(a_addr, b_addr)), dtype=np.float32, count=n)
    addr_tset = np.fromiter((fuzz.token_set_ratio(x, y) for x, y in zip(a_addr, b_addr)), dtype=np.float32, count=n)

    # Convert this batch's tuples to sets just-in-time (bounded by batch_size,
    # unlike the persistent pool_lookup which stores the cheaper tuples).
    a_ntok = [frozenset(t) for t in a["name_tok"].values]
    b_ntok = [frozenset(t) for t in b["name_tok"].values]
    a_atok = [frozenset(t) for t in a["addr_tok"].values]
    b_atok = [frozenset(t) for t in b["addr_tok"].values]

    name_jac = np.fromiter((_jaccard(x, y) for x, y in zip(a_ntok, b_ntok)), dtype=np.float32, count=n)
    addr_jac = np.fromiter((_jaccard(x, y) for x, y in zip(a_atok, b_atok)), dtype=np.float32, count=n)
    n_shared_name = np.fromiter((len(x & y) for x, y in zip(a_ntok, b_ntok)), dtype=np.float32, count=n)
    n_shared_addr = np.fromiter((len(x & y) for x, y in zip(a_atok, b_atok)), dtype=np.float32, count=n)

    country_match = (a["country"].values == b["country"].values).astype(np.float32)

    a_nlen = a["name_len"].values.astype(np.float32)
    b_nlen = b["name_len"].values.astype(np.float32)
    a_alen = a["addr_len"].values.astype(np.float32)
    b_alen = b["addr_len"].values.astype(np.float32)

    name_len_diff = np.abs(a_nlen - b_nlen)
    addr_len_diff = np.abs(a_alen - b_alen)
    name_len_ratio = np.minimum(a_nlen, b_nlen) / np.maximum(a_nlen, b_nlen, out=np.ones_like(a_nlen), where=(np.maximum(a_nlen, b_nlen) > 0))
    addr_len_ratio = np.minimum(a_alen, b_alen) / np.maximum(a_alen, b_alen, out=np.ones_like(a_alen), where=(np.maximum(a_alen, b_alen) > 0))

    feats = np.column_stack([
        name_ratio, name_tsort, name_tset,
        addr_ratio, addr_tsort, addr_tset,
        name_jac, addr_jac,
        n_shared_name, n_shared_addr,
        country_match,
        name_len_diff, addr_len_diff,
        name_len_ratio, addr_len_ratio,
    ])
    return feats.astype(np.float32), valid


def compute_features(pairs, s1_lookup, pool_lookup, batch_size=FEATURE_BATCH):
    """pairs: DataFrame[s1_id, cand_id]. Returns (feats, valid_mask):
    feats is a numpy array (n, len(FEATURE_COLS)) (zeroed for invalid rows),
    valid_mask is a boolean array of length n -- False for any pair whose
    s1_id/cand_id could not be found in its lookup table (should be rare;
    callers should drop these rows rather than train/predict on them).

    Processed in fixed-size batches (default 500k pairs) regardless of the
    total candidate-pair count for this call: the per-batch working set
    (``.loc`` lookups + the various np.fromiter temporaries) is what actually
    drives peak memory, not the persistent pool_index/pool_lookup structures,
    so bounding it here keeps memory flat even when a single blocking chunk
    produces tens of millions of candidate pairs.
    """
    n = len(pairs)
    if n == 0:
        return np.empty((0, len(FEATURE_COLS)), dtype=np.float32), np.empty((0,), dtype=bool)
    s1_ids_all = np.asarray(pairs["s1_id"].values, dtype=object)
    cand_ids_all = np.asarray(pairs["cand_id"].values, dtype=object)
    # Built once per call and reused across every batch below -- these tables
    # don't change between batches, so rebuilding them per-batch was pure
    # wasted work (significant when pool_lookup has many rows).
    s1_idx_map = _build_idx_map(s1_lookup)
    pool_idx_map = _build_idx_map(pool_lookup)
    if n <= batch_size:
        return _compute_features_batch(s1_ids_all, cand_ids_all, s1_lookup, pool_lookup,
                                        s1_idx_map, pool_idx_map)

    out = np.empty((n, len(FEATURE_COLS)), dtype=np.float32)
    valid_out = np.empty((n,), dtype=bool)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        feats_batch, valid_batch = _compute_features_batch(
            s1_ids_all[start:end], cand_ids_all[start:end], s1_lookup, pool_lookup,
            s1_idx_map, pool_idx_map)
        out[start:end] = feats_batch
        valid_out[start:end] = valid_batch
    return out, valid_out
