"""Macro-averaged F_0.5 scorer, reusable for any candidate/prediction dict."""
import pandas as pd


def parse_id_list(s):
    if s is None or (isinstance(s, float)):
        return frozenset()
    s = str(s).strip()
    if not s:
        return frozenset()
    return frozenset(x for x in s.split(",") if x)


def load_gt_dict(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {row.source1_entity_id: parse_id_list(row.matched_entity_ids)
            for row in df.itertuples(index=False)}


def f05_per_entity(pred_set, gt_set):
    if not gt_set and not pred_set:
        return 1.0
    if not gt_set and pred_set:
        return 0.0
    if gt_set and not pred_set:
        return 0.0
    tp = len(pred_set & gt_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(gt_set)
    beta2 = 0.25
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def macro_f05(pred_dict, gt_dict):
    """pred_dict/gt_dict: s1_entity_id -> frozenset(matched_ids). Every key in
    gt_dict must have an entry (possibly empty set) in pred_dict."""
    scores = []
    for s1_id, gt_set in gt_dict.items():
        pred_set = pred_dict.get(s1_id, frozenset())
        scores.append(f05_per_entity(pred_set, gt_set))
    return sum(scores) / len(scores) if scores else 0.0


def blocking_recall_and_reduction(cand_dict, gt_dict, n_pool):
    """cand_dict: s1_id -> frozenset(candidate ids). n_pool: size of S2+S3 pool
    the blocker drew from (for reduction-ratio denominator)."""
    total_gt_pairs = 0
    recovered = 0
    total_candidates = 0
    for s1_id, gt_set in gt_dict.items():
        cand_set = cand_dict.get(s1_id, frozenset())
        total_candidates += len(cand_set)
        if gt_set:
            total_gt_pairs += len(gt_set)
            recovered += len(gt_set & cand_set)
    recall_ceiling = recovered / total_gt_pairs if total_gt_pairs else 1.0
    n_s1 = len(gt_dict)
    avg_candidates = total_candidates / n_s1 if n_s1 else 0.0
    reduction_ratio = 1 - (total_candidates / (n_s1 * n_pool)) if n_s1 and n_pool else 0.0
    return {
        "recall_ceiling": recall_ceiling,
        "avg_candidates_per_s1": avg_candidates,
        "reduction_ratio": reduction_ratio,
        "total_candidates": total_candidates,
    }
