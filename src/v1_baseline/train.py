"""Train the matching classifier on the training split, tune the decision
threshold on the held-out validation split to maximize macro F_0.5, and save
the fitted model + threshold + blocking stats artifacts.

Runs blocking + feature extraction in bounded memory: candidates are
generated country-partition by country-partition, and within a country in
row-chunks of S1 (see pipeline_common.iter_candidate_chunks), so peak memory
never has to hold an index/lookup over the full ~10.3M-row S2+S3 pool at
once. Only the resulting (compact, float32) feature matrix is accumulated in
memory across chunks -- this is the part that must exist all at once for
sklearn/LightGBM training and threshold search.

Usage:
    python3 train.py --data-dir <train_dir> --out-dir <artifacts_dir> [--sample N] [--chunk-size N]
"""
import argparse
import gc
import json
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.linear_model import LogisticRegression

from features import build_lookup, compute_features, FEATURE_COLS
from pipeline_common import iter_candidate_chunks, build_pool_lookup_for_pairs, DEFAULT_CHUNK_SIZE
from score import parse_id_list, macro_f05, blocking_recall_and_reduction


def split_s1(s1_df, val_frac=0.15, seed=42):
    rng = np.random.RandomState(seed)
    # Force a plain numpy object array before shuffling: pandas may back a
    # dtype=str column with a StringArray/ArrowStringArray, which numpy's
    # RandomState.shuffle does not permute safely in place (can duplicate
    # entries), silently corrupting the train/val split.
    ids = np.array(s1_df["entity_id"].tolist(), dtype=object)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_frac)
    val_ids = set(ids[:n_val])
    is_val = s1_df["entity_id"].isin(val_ids).values
    return np.where(is_val, "val", "train")


def gt_dict_to_pairs_df(gt_dict):
    s1_ids, cand_ids = [], []
    for s1id, cset in gt_dict.items():
        for c in cset:
            s1_ids.append(s1id)
            cand_ids.append(c)
    return pd.DataFrame({"s1_id": s1_ids, "cand_id": cand_ids, "label": np.int8(1)})


def label_pairs(pairs, gt_pairs_df):
    """Vectorized labeling via merge (positive if (s1_id, cand_id) is a true match)."""
    merged = pairs[["s1_id", "cand_id"]].merge(gt_pairs_df, on=["s1_id", "cand_id"], how="left")
    return merged["label"].fillna(0).astype(np.int8).values


def pick_threshold(pair_s1_ids, probs, labels, all_s1_ids, gt_dict):
    """Grid-search a probability threshold that maximizes macro F0.5,
    using vectorized per-s1 TP/pred-count aggregation."""
    gt_len = pd.Series({s1: len(gt_dict.get(s1, ())) for s1 in all_s1_ids}, name="gt_len")
    all_s1_df = pd.DataFrame({"s1_id": all_s1_ids}).merge(
        gt_len.rename_axis("s1_id").reset_index(), on="s1_id", how="left")

    best_thr, best_f05 = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.02):
        pred = probs >= thr
        tp = labels & pred
        g = pd.DataFrame({"s1_id": pair_s1_ids, "tp": tp.astype(np.int64), "pred": pred.astype(np.int64)})
        agg = g.groupby("s1_id", sort=False).sum()
        merged = all_s1_df.merge(agg, on="s1_id", how="left").fillna(0)
        tp_v = merged["tp"].values
        pred_v = merged["pred"].values
        gt_v = merged["gt_len"].values
        f05 = np.zeros(len(merged))
        both_empty = (gt_v == 0) & (pred_v == 0)
        f05[both_empty] = 1.0
        has_tp = tp_v > 0
        precision = np.divide(tp_v, pred_v, out=np.zeros_like(tp_v, dtype=float), where=pred_v > 0)
        recall = np.divide(tp_v, gt_v, out=np.zeros_like(tp_v, dtype=float), where=gt_v > 0)
        denom = 0.25 * precision + recall
        f05_valid = np.divide(1.25 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
        f05[has_tp] = f05_valid[has_tp]
        mean_f05 = f05.mean()
        if mean_f05 > best_f05:
            best_f05, best_thr = mean_f05, thr
    return best_thr, best_f05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sample", type=int, default=0, help="if >0, subsample S1 rows for quick iteration")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    s1 = pd.read_csv(f"{args.data_dir}/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(f"{args.data_dir}/train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(f"{args.data_dir}/train_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt_df = pd.read_csv(f"{args.data_dir}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt_dict = {r.source1_entity_id: parse_id_list(r.matched_entity_ids) for r in gt_df.itertuples(index=False)}
    gt_pairs_df = gt_dict_to_pairs_df(gt_dict)
    print(f"[load] {time.time()-t0:.1f}s  S1={len(s1)} S2={len(s2)} S3={len(s3)}", flush=True)

    if args.sample:
        s1 = s1.sample(args.sample, random_state=0).reset_index(drop=True)

    s1["split"] = split_s1(s1, args.val_frac)
    print(f"[split] train={sum(s1['split']=='train')} val={sum(s1['split']=='val')}", flush=True)

    pool_sources = [s2, s3]
    n_pool_by_country = pd.concat([s2["country"], s3["country"]]).value_counts().to_dict()

    feats_list, labels_list, s1ids_list, candids_list, split_list = [], [], [], [], []
    n_pairs_total = 0
    t0 = time.time()
    for country, chunk, pairs, pool_c in iter_candidate_chunks(s1, pool_sources, args.chunk_size):
        n_pairs_total += len(pairs)
        if len(pairs) == 0:
            continue
        pool_lookup = build_pool_lookup_for_pairs(pool_c, pairs)
        s1_lookup = build_lookup(chunk)
        feats, valid = compute_features(pairs, s1_lookup, pool_lookup)
        labels = label_pairs(pairs, gt_pairs_df)
        split_map = chunk.set_index("entity_id")["split"]
        pair_split = pairs["s1_id"].map(split_map).values

        if not valid.all():
            feats, labels = feats[valid], labels[valid]
            pair_s1ids = pairs["s1_id"].values[valid]
            pair_candids = pairs["cand_id"].values[valid]
            pair_split = pair_split[valid]
        else:
            pair_s1ids = pairs["s1_id"].values
            pair_candids = pairs["cand_id"].values

        feats_list.append(feats)
        labels_list.append(labels)
        s1ids_list.append(pair_s1ids)
        candids_list.append(pair_candids)
        split_list.append(pair_split)
        del pool_lookup, s1_lookup, feats, labels
        gc.collect()
        print(f"  [chunk] country={country} chunk_rows={len(chunk)} pairs={len(pairs)} "
              f"elapsed={time.time()-t0:.1f}s", flush=True)

    print(f"[candidates] total_pairs={n_pairs_total} time={time.time()-t0:.1f}s", flush=True)

    all_feats = np.concatenate(feats_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    all_s1ids = np.concatenate(s1ids_list, axis=0)
    all_candids = np.concatenate(candids_list, axis=0)
    all_split = np.concatenate(split_list, axis=0)
    del feats_list, labels_list, s1ids_list, candids_list, split_list
    gc.collect()

    train_mask = all_split == "train"
    val_mask = all_split == "val"
    train_feats, val_feats = all_feats[train_mask], all_feats[val_mask]
    train_labels, val_labels = all_labels[train_mask], all_labels[val_mask]
    val_s1ids = all_s1ids[val_mask]
    print(f"[labels] train pos={train_labels.sum()}/{len(train_labels)} "
          f"val pos={val_labels.sum()}/{len(val_labels)}", flush=True)

    # blocking quality (recall ceiling / reduction ratio) on the validation split
    val_s1_entities = s1[s1["split"] == "val"]
    val_gt = {k: gt_dict[k] for k in val_s1_entities["entity_id"]}
    val_cand_dict = pd.DataFrame({"s1_id": val_s1ids, "cand_id": all_candids[val_mask]}) \
        .groupby("s1_id")["cand_id"].apply(frozenset).to_dict()
    n_pool_total = sum(n_pool_by_country.values())
    block_stats = blocking_recall_and_reduction(val_cand_dict, val_gt, n_pool_total)
    print(f"[blocking-quality] {block_stats}", flush=True)
    del val_cand_dict
    gc.collect()

    name_idx = FEATURE_COLS.index("name_tset")
    addr_idx = FEATURE_COLS.index("addr_tset")
    comparison = {}
    all_val_s1_ids = val_s1_entities["entity_id"].values

    # --- Approach A: rule-based weighted-similarity threshold (no ML) --------
    def rule_score(feats):
        return 0.5 * feats[:, name_idx] / 100.0 + 0.5 * feats[:, addr_idx] / 100.0

    val_rule_probs = rule_score(val_feats)
    thr_a, f05_a = pick_threshold(val_s1ids, val_rule_probs, val_labels, all_val_s1_ids, gt_dict)
    comparison["A_rule_based_weighted_similarity"] = {"threshold": float(thr_a), "val_macro_f05": float(f05_a)}
    print(f"[approach A: rule-based] thr={thr_a:.2f} f05={f05_a:.4f}", flush=True)

    # --- Approach B: logistic regression --------------------------------------
    t0 = time.time()
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(train_feats, train_labels)
    val_lr_probs = logreg.predict_proba(val_feats)[:, 1]
    thr_b, f05_b = pick_threshold(val_s1ids, val_lr_probs, val_labels, all_val_s1_ids, gt_dict)
    comparison["B_logistic_regression"] = {"threshold": float(thr_b), "val_macro_f05": float(f05_b),
                                            "train_time_s": time.time() - t0}
    print(f"[approach B: logreg] {time.time()-t0:.1f}s thr={thr_b:.2f} f05={f05_b:.4f}", flush=True)

    # --- Approach C: LightGBM gradient boosting -------------------------------
    t0 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=300, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        objective="binary", class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model.fit(train_feats, train_labels)
    print(f"[train lgbm] {time.time()-t0:.1f}s", flush=True)

    val_probs = model.predict_proba(val_feats)[:, 1]
    thr, f05 = pick_threshold(val_s1ids, val_probs, val_labels, all_val_s1_ids, gt_dict)
    comparison["C_lightgbm"] = {"threshold": float(thr), "val_macro_f05": float(f05),
                                 "train_time_s": time.time() - t0}
    print(f"[approach C: lightgbm] thr={thr:.2f} val_macro_f05={f05:.4f}", flush=True)

    print("[comparison table]", json.dumps(comparison, indent=2), flush=True)

    best_name = max(comparison, key=lambda k: comparison[k]["val_macro_f05"])
    print(f"[selected approach] {best_name}", flush=True)

    joblib.dump(model, f"{args.out_dir}/model.joblib")
    with open(f"{args.out_dir}/config.json", "w") as f:
        json.dump({
            "threshold": float(thr),
            "val_macro_f05": float(f05),
            "feature_cols": FEATURE_COLS,
            "blocking_quality": block_stats,
            "n_pairs_total": int(n_pairs_total),
            "n_train_pairs": int(train_mask.sum()),
            "n_val_pairs": int(val_mask.sum()),
            "train_pos": int(train_labels.sum()),
            "val_pos": int(val_labels.sum()),
            "comparison": comparison,
            "selected_approach": best_name,
            "chunk_size": args.chunk_size,
        }, f, indent=2)
    print("[done] artifacts saved to", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
