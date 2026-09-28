"""Quick, lightweight EDA over the training data. Prints a compact summary."""
import sys
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from score import parse_id_list


def main(data_dir):
    s1 = pd.read_csv(f"{data_dir}/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(f"{data_dir}/train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(f"{data_dir}/train_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt = pd.read_csv(f"{data_dir}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)

    print("=== Row counts ===")
    print(f"S1={len(s1)}  S2={len(s2)}  S3={len(s3)}  GT rows={len(gt)}")

    print("\n=== Country distribution (S1) ===")
    print(s1["country"].value_counts())
    print("\n=== Country distribution (S2) ===")
    print(s2["country"].value_counts())
    print("\n=== Country distribution (S3) ===")
    print(s3["country"].value_counts())

    match_lists = gt["matched_entity_ids"].map(parse_id_list)
    n_sizes = match_lists.map(len)
    n_singleton = (n_sizes == 0).sum()
    print("\n=== Match cardinality ===")
    print(f"singleton (0 matches): {n_singleton} ({n_singleton/len(gt):.4%})")
    print(f"non-singleton: {len(gt)-n_singleton}")
    print(f"avg matches per non-singleton: {n_sizes[n_sizes>0].mean():.3f}")
    print(f"max matches: {n_sizes.max()}")
    print(n_sizes.value_counts().sort_index().head(15))

    # source split of matched ids
    all_ids = [i for lst in match_lists for i in lst]
    n_s2 = sum(1 for i in all_ids if i.startswith("S2-"))
    n_s3 = sum(1 for i in all_ids if i.startswith("S3-"))
    print(f"\nmatched ids from S2: {n_s2}, from S3: {n_s3}")

    print("\n=== Sample noisy name pairs (S1 vs matched) ===")
    gt_pos = gt[n_sizes > 0].head(5)
    s1_idx = s1.set_index("entity_id")
    s2_idx = s2.set_index("entity_id")
    s3_idx = s3.set_index("entity_id")
    for row in gt_pos.itertuples(index=False):
        s1r = s1_idx.loc[row.source1_entity_id]
        print(f"S1: {s1r.business_name!r} | {s1r.business_address!r} | {s1r.country}")
        for mid in parse_id_list(row.matched_entity_ids):
            src = s2_idx if mid.startswith("S2-") else s3_idx
            r = src.loc[mid]
            print(f"   -> {mid}: {r.business_name!r} | {r.business_address!r} | {r.country}")
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dataset/train")
