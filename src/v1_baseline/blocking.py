"""Scalable inverted-index blocking.

Strategy
--------
1. Tokenize business_name and business_address of every record (stop-word /
   legal-suffix stripped) -- see ``normalize.py``.
2. Compute per-(country, token) document frequency over the "pool" side
   (Source 2 + Source 3). Tokens that are too common (df > MAX_DF) are
   dropped from the index entirely -- they carry no discriminative power and
   would blow up the candidate set.
3. For every Source-1 record, keep only its K rarest surviving tokens
   (ranked by df in the pool) as *query keys* -- this bounds fan-out from the
   S1 side regardless of how many tokens a name/address has.
4. Candidate pairs = S1 query keys JOINed against the pool index on
   (country, token), for name-tokens and address-tokens separately, then
   unioned. This turns blocking into a handful of vectorized pandas merges
   instead of any O(n*m) comparison.
"""
import gc

import pandas as pd

# See pipeline_common.py: disabled globally to avoid pandas' inferred "str"
# extension dtype, whose merge behavior was observed to be unreliable at this
# pipeline's scale. Repeated here (idempotent) in case this module loads
# first.
pd.set_option("future.infer_string", False)

from normalize import name_tokens, addr_tokens

NAME_MAX_DF = 350
ADDR_MAX_DF = 450
NAME_TOPK = 5
ADDR_TOPK = 5

# Row-chunk size used while tokenizing/exploding a source DataFrame. Keeping
# this bounded (rather than tokenizing an entire multi-million-row country
# partition in one go) is what actually controls peak transient memory: the
# dominant cost is the *transient* python list of freshly-created token
# strings produced before they are folded into a compact Categorical, and
# that list's size is proportional to however many rows we tokenize at once.
EXPLODE_ROW_CHUNK = 300_000


def _explode_tokens_block(df, tokenizer, col):
    toks = df[col].map(tokenizer)
    lengths = toks.map(len).to_numpy()
    flat_tokens = [t for sub in toks for t in sub]
    return pd.DataFrame({
        "entity_id": df["entity_id"].to_numpy().repeat(lengths),
        "country": pd.Categorical(df["country"].to_numpy().repeat(lengths)),
        "token": pd.Categorical(flat_tokens),
    })


def explode_tokens(df, tokenizer, kind, row_chunk=EXPLODE_ROW_CHUNK):
    """df: entity_id, business_name/address, country -> long (entity_id, country, token).

    Token/country columns are cast to `category` immediately: the token
    vocabulary is orders of magnitude smaller than the exploded row count, so
    this keeps memory bounded on multi-million-row pools. Tokenization itself
    is done in row-chunks (see EXPLODE_ROW_CHUNK) to bound the transient
    per-token python-string-list memory spike on large sources.
    """
    col = "business_name" if kind == "name" else "business_address"
    if len(df) <= row_chunk:
        return _explode_tokens_block(df, tokenizer, col)
    blocks = []
    for start in range(0, len(df), row_chunk):
        blocks.append(_explode_tokens_block(df.iloc[start:start + row_chunk], tokenizer, col))
    out = pd.concat(blocks, ignore_index=True)
    del blocks
    return out


def _doc_freq(long_df):
    return (long_df.groupby(["country", "token"], observed=True)
            .size().rename("df").reset_index())


def _select_topk(long_df, freq_df, k, max_df):
    m = long_df.merge(freq_df, on=["country", "token"], how="left")
    m = m[m["df"].notna() & (m["df"] <= max_df)]
    m = m.sort_values(["entity_id", "df"])
    m = m.groupby("entity_id", sort=False).head(k)
    return m[["entity_id", "country", "token"]]


def build_pool_index(pool_sources):
    """pool_sources: list of source DataFrames (e.g. [s2_df, s3_df]).

    Processes each source's tokens separately and concatenates only the
    compact (categorical) long-format token frames -- this avoids ever
    materializing a full concatenated copy of the raw S2+S3 text columns,
    which is the dominant memory cost at multi-million-row scale.
    """
    name_longs, addr_longs = [], []
    for src in pool_sources:
        name_longs.append(explode_tokens(src, name_tokens, "name"))
        addr_longs.append(explode_tokens(src, addr_tokens, "addr"))
    name_long = pd.concat(name_longs, ignore_index=True)
    addr_long = pd.concat(addr_longs, ignore_index=True)
    del name_longs, addr_longs
    gc.collect()

    name_freq = _doc_freq(name_long)
    addr_freq = _doc_freq(addr_long)
    name_idx = name_long.merge(name_freq, on=["country", "token"], how="left")
    name_idx = name_idx[name_idx["df"] <= NAME_MAX_DF][["country", "token", "entity_id"]]
    addr_idx = addr_long.merge(addr_freq, on=["country", "token"], how="left")
    addr_idx = addr_idx[addr_idx["df"] <= ADDR_MAX_DF][["country", "token", "entity_id"]]
    del name_long, addr_long
    gc.collect()
    return {
        "name_idx": name_idx.rename(columns={"entity_id": "cand_id"}),
        "addr_idx": addr_idx.rename(columns={"entity_id": "cand_id"}),
        "name_freq": name_freq,
        "addr_freq": addr_freq,
    }


def generate_candidates(s1_df, pool_index):
    """Returns DataFrame[s1_entity_id, cand_id] (deduped), using the pool index
    built by build_pool_index over S2+S3."""
    s1_name_long = explode_tokens(s1_df, name_tokens, "name")
    s1_addr_long = explode_tokens(s1_df, addr_tokens, "addr")

    s1_name_keys = _select_topk(s1_name_long, pool_index["name_freq"], NAME_TOPK, NAME_MAX_DF)
    s1_addr_keys = _select_topk(s1_addr_long, pool_index["addr_freq"], ADDR_TOPK, ADDR_MAX_DF)

    name_pairs = s1_name_keys.merge(pool_index["name_idx"], on=["country", "token"], how="inner")
    addr_pairs = s1_addr_keys.merge(pool_index["addr_idx"], on=["country", "token"], how="inner")

    pairs = pd.concat([
        name_pairs[["entity_id", "cand_id"]],
        addr_pairs[["entity_id", "cand_id"]],
    ], ignore_index=True)
    pairs = pairs.rename(columns={"entity_id": "s1_id"}).drop_duplicates()
    return pairs
