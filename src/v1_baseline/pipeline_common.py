"""Shared helpers for train.py / predict.py.

Memory strategy (14GB box, pool up to ~10.3M rows):
1. **Country partitioning.** Country is a given field, not inferred, and
   US/India/France candidates can never match across countries (blocking
   already conditions on country). So we build one inverted-index pool
   structure *per country* instead of one over the whole pool -- this bounds
   the single largest in-memory structure to the biggest country partition
   (US) rather than the full pool.
2. **S1 row-chunking within a country.** Even a single country's S1 slice
   (e.g. ~1.3M US rows) is chunked (default 150k rows) before querying the
   index, so the resulting candidate-pair frame per step stays small.
3. **Per-chunk feature lookup.** The expensive per-record lookup structures
   (normalized strings + token sets) are only ever built for entity_ids that
   actually appear as a candidate in the *current chunk*, not the whole pool
   or even the whole country partition.
"""
import gc

import pandas as pd

# pandas >=2.2 ships a "future.infer_string" option (default True as of
# pandas 3.x) that silently backs every string column with a new StringDtype
# extension array instead of classic numpy object dtype -- including columns
# freshly built from a plain numpy object array via pd.DataFrame(...). Under
# this pipeline's multi-million-row merges (the entity_id lookup joins in
# features.py), that extension array's hash-join path was observed to
# silently drop a large, non-deterministic fraction of otherwise-valid
# matches (verified via a targeted repro: identical inputs merge correctly
# under plain object dtype, incorrectly under the inferred "str" dtype).
# Disabling it forces classic, reliable object-dtype string handling
# throughout. Set once, here, since this module is imported by every entry
# point (train.py, predict.py) before any pandas DataFrame is built.
pd.set_option("future.infer_string", False)

from blocking import build_pool_index, generate_candidates
from features import build_lookup

DEFAULT_CHUNK_SIZE = 150_000


def read_source(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def filter_sources_by_ids(sources, ids):
    """sources: list of DataFrames with an entity_id column. Returns a single
    concatenated DataFrame containing only rows whose entity_id is in `ids`."""
    ids = set(ids)
    parts = [src[src["entity_id"].isin(ids)] for src in sources]
    return pd.concat(parts, ignore_index=True)


def build_pool_lookup_for_pairs(pool_sources, pairs):
    """Build the feature lookup only for pool entity_ids that appear as a
    candidate in `pairs` -- this is typically a tiny fraction of the pool."""
    needed_ids = pairs["cand_id"].unique()
    subset = filter_sources_by_ids(pool_sources, needed_ids)
    lookup = build_lookup(subset)
    del subset
    gc.collect()
    return lookup


def iter_candidate_chunks(s1_df, pool_sources, chunk_size=DEFAULT_CHUNK_SIZE):
    """Yields (country, s1_chunk_df, pairs_df) for every S1 chunk, country by
    country. Builds (and frees) one pool_index per country -- the only
    "large" structure held in memory at a time, sized to that country's pool
    partition rather than the whole S2+S3 pool.
    """
    countries = s1_df["country"].unique()
    for country in countries:
        s1_c = s1_df[s1_df["country"] == country]
        pool_c = [src[src["country"] == country].reset_index(drop=True) for src in pool_sources]
        n_pool_c = sum(len(p) for p in pool_c)
        if len(s1_c) == 0:
            continue
        if n_pool_c == 0:
            # No pool records at all for this country (e.g. a brand-new
            # country with zero S2/S3 rows) -- every S1 entity here is
            # necessarily a singleton; yield empty pair chunks so callers
            # still see every entity_id.
            for start in range(0, len(s1_c), chunk_size):
                chunk = s1_c.iloc[start:start + chunk_size]
                empty_pairs = pd.DataFrame({"s1_id": pd.Series(dtype=str), "cand_id": pd.Series(dtype=str)})
                yield country, chunk, empty_pairs, pool_c
            continue

        pool_index = build_pool_index(pool_c)
        try:
            for start in range(0, len(s1_c), chunk_size):
                chunk = s1_c.iloc[start:start + chunk_size]
                pairs = generate_candidates(chunk, pool_index)
                yield country, chunk, pairs, pool_c
        finally:
            del pool_index
            gc.collect()
