"""Run the full pipeline (blocking -> features -> trained model -> threshold)
over the actual test data and write output/matching_results.tsv and
output/candidate_pairs.tsv.

Uses the same bounded-memory, country-partitioned + row-chunked candidate
generation as train.py (see pipeline_common.iter_candidate_chunks): peak
memory is governed by the largest single country's S2/S3 partition and the
per-chunk feature lookup, never by the full ~10M-row test pool at once.
Both output files are written incrementally per S1 entity as chunks are
processed.

Usage:
    python3 predict.py --data-dir <test_dir> --artifacts-dir <artifacts_dir> \
        --out-dir <output_dir> [--chunk-size N]
"""
import argparse
import gc
import json
import os
import time

import joblib
import pandas as pd

from features import build_lookup, compute_features
from pipeline_common import iter_candidate_chunks, build_pool_lookup_for_pairs, DEFAULT_CHUNK_SIZE


def write_rows(fh, id_dict, s1_ids):
    for s1id in s1_ids:
        ids = id_dict.get(s1id, ())
        fh.write(f"{s1id}\t{','.join(ids)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = joblib.load(f"{args.artifacts_dir}/model.joblib")
    with open(f"{args.artifacts_dir}/config.json") as f:
        cfg = json.load(f)
    threshold = cfg["threshold"]
    print(f"[loaded model] threshold={threshold}", flush=True)

    t0 = time.time()
    s1 = pd.read_csv(f"{args.data_dir}/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(f"{args.data_dir}/test_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(f"{args.data_dir}/test_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    print(f"[load] {time.time()-t0:.1f}s S1={len(s1)} S2={len(s2)} S3={len(s3)}", flush=True)
    all_s1_ids = s1["entity_id"].tolist()

    cand_fh = open(f"{args.out_dir}/candidate_pairs.tsv", "w", encoding="utf-8")
    match_fh = open(f"{args.out_dir}/matching_results.tsv", "w", encoding="utf-8")
    cand_fh.write("source1_entity_id\tcandidate_entity_ids\n")
    match_fh.write("source1_entity_id\tmatched_entity_ids\n")

    n_pairs_total = 0
    n_matched_s1 = 0
    n_s1_seen = 0
    t0 = time.time()
    for country, chunk, pairs, pool_c in iter_candidate_chunks(s1, [s2, s3], args.chunk_size):
        chunk_ids = chunk["entity_id"].tolist()
        n_s1_seen += len(chunk_ids)
        n_pairs_total += len(pairs)

        if len(pairs) == 0:
            write_rows(cand_fh, {}, chunk_ids)
            write_rows(match_fh, {}, chunk_ids)
            print(f"  [chunk] country={country} rows={len(chunk_ids)} pairs=0 "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
            continue

        pool_lookup = build_pool_lookup_for_pairs(pool_c, pairs)
        s1_lookup = build_lookup(chunk)
        feats, valid = compute_features(pairs, s1_lookup, pool_lookup)
        if not valid.all():
            # A candidate pair whose id couldn't be resolved is dropped
            # entirely (never scored, never a candidate/match) rather than
            # risking a spurious match on placeholder features.
            pairs = pairs[valid]
            feats = feats[valid]
        probs = model.predict_proba(feats)[:, 1]
        del pool_lookup, s1_lookup, feats
        gc.collect()

        pairs = pairs.reset_index(drop=True)
        pairs["prob"] = probs

        cand_dict = pairs.groupby("s1_id")["cand_id"].apply(list).to_dict()
        write_rows(cand_fh, cand_dict, chunk_ids)

        matched = pairs[pairs["prob"] >= threshold]
        match_dict = matched.groupby("s1_id")["cand_id"].apply(list).to_dict()
        write_rows(match_fh, match_dict, chunk_ids)
        n_matched_s1 += sum(1 for v in match_dict.values() if v)

        print(f"  [chunk] country={country} rows={len(chunk_ids)} pairs={len(pairs)} "
              f"elapsed={time.time()-t0:.1f}s", flush=True)
        del pairs, probs, cand_dict, match_dict
        gc.collect()

    cand_fh.close()
    match_fh.close()
    print(f"[done] s1_total={len(all_s1_ids)} s1_seen={n_s1_seen} s1_with_match={n_matched_s1} "
          f"total_pairs={n_pairs_total} avg_candidates={n_pairs_total/max(1,len(all_s1_ids)):.2f}", flush=True)


if __name__ == "__main__":
    main()
