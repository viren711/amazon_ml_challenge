# Business Entity Resolution — Pipeline

Reproducible pipeline for the Amazon ML Challenge "Business Entity Resolution" task:
given Source-1 (deduplicated reference) business records and noisy Source-2 /
Source-3 records, find every Source-2/3 record that refers to the same
real-world business as each Source-1 entity.

## Approach summary

1. **Blocking (`blocking.py`)** — inverted-index token blocking. Names and
   addresses are normalized (`normalize.py`: Unicode NFKC, lowercasing,
   legal-suffix / street-abbreviation expansion, stop-word stripping) and
   tokenized. Tokens with document frequency above a cap are dropped from the
   index (too common to be discriminative); each Source-1 record then queries
   the index using only its `K` rarest surviving name tokens and address
   tokens. This turns candidate generation into a few vectorized pandas
   merges instead of any all-pairs comparison, and naturally bounds fan-out
   regardless of pool size.
2. **Feature engineering (`features.py`)** — per candidate pair: rapidfuzz
   ratio / token-sort ratio / token-set ratio on name and address, token
   Jaccard overlap, shared-token counts, country match, and length
   diff/ratio features (15 features total).
3. **Matching model (`train.py`)** — trains and compares three approaches on
   a held-out validation split of `train_source1` (with ground truth):
   (A) a rule-based weighted name/address similarity threshold, (B) logistic
   regression, (C) LightGBM gradient boosting — all licensed MIT/BSD/Apache-2.0
   and far under the 8B-parameter cap. The decision threshold for each is
   grid-searched directly against macro F_0.5 on the validation split (not a
   generic 0.5 cutoff), since the metric is precision-heavy. The best
   approach (by validation F_0.5) is reported in `artifacts/config.json`
   under `comparison`; LightGBM is shipped as the production model.
4. **Inference (`predict.py`)** — re-runs the identical blocking + feature
   pipeline over the real test data, scores every candidate with the trained
   model, and writes `candidate_pairs.tsv` (the exact, uncapped candidate set
   fed to the model) and `matching_results.tsv` (candidates above the tuned
   threshold). Both files are written incrementally, country-partition by
   country-partition, so memory never has to hold the full ~1.7M-entity test
   set's candidates at once.

**Bug found and fixed:** an earlier version of `build_lookup` in
`features.py` assigned a `.map()` result (which retains its input's original
index) into a freshly 0-indexed output DataFrame — an index-aligned, not
positional, assignment — causing pandas to silently NaN-fill rows wherever
the two indices didn't coincide. Fixed by materializing values via
`.to_numpy()` before assignment. See the top-level `README.md` and
`Documentation_template.md` (section 5) for the full writeup.

## Environment setup

```bash
python3 -m pip install --user -r requirements.txt
# (or: python3 -m pip install --user --break-system-packages -r requirements.txt
#  on a PEP 668 "externally managed" system Python with no venv available)
```

Python 3.10+ recommended. No internet access, external APIs, or external data
are used anywhere in this pipeline — everything is derived only from the
provided train/test TSVs.

## Reproducing end-to-end

Run everything from this directory (`code/business_entity_resolution/`), with
`DATASET_ROOT` pointing at the challenge's `dataset/` folder (the one
containing `train/` and `test/`):

```bash
export DATASET_ROOT=/path/to/student_resource/dataset
export PYTHONPATH=src

# 1) EDA (optional, prints a summary to stdout)
python3 src/eda.py "$DATASET_ROOT/train"

# 2) Train: builds candidates + features on a held-out val split of train,
#    trains + compares 3 approaches, tunes the decision threshold for macro
#    F_0.5, and saves model + config to ./artifacts
python3 src/train.py --data-dir "$DATASET_ROOT/train" --out-dir ./artifacts --chunk-size 50000

# 3) Predict on the real test set -> output/matching_results.tsv,
#    output/candidate_pairs.tsv
python3 src/predict.py --data-dir "$DATASET_ROOT/test" \
    --artifacts-dir ./artifacts --out-dir ./output --chunk-size 50000

# 4) Validate (from the student_resource/ directory of the challenge repo)
cd /path/to/student_resource
python3 utils/validate_submission.py \
    --matching /path/to/code/business_entity_resolution/output/matching_results.tsv \
    --candidate /path/to/code/business_entity_resolution/output/candidate_pairs.tsv \
    --test-dir dataset/test
```

`train.py` runs on the full ~2.2M-row train Source-1 file against the full
~5M+5.3M Source-2/3 pool (a full-scale run, not a subsample) so the reported
validation F_0.5 and blocking-quality numbers are representative of the real
test-time behavior. `--sample N` is available for fast local iteration on a
subsample while developing, but is not how the shipped model/threshold were
produced. `--chunk-size` controls how many Source-1 rows are processed per
blocking/feature step within a country; 50,000 was tuned for a 14GB-RAM
machine (see `pipeline_common.py`'s module docstring for the memory
strategy) — increase it on a box with more RAM to reduce wall-clock time.

## Source layout

- `normalize.py` — text normalization / tokenization rules
- `blocking.py` — inverted-index candidate generation
- `features.py` — pairwise similarity feature engineering
- `pipeline_common.py` — shared blocking+feature glue
- `score.py` — reusable macro F_0.5 scorer + blocking recall/reduction metrics
- `train.py` — trains/compares approaches, tunes threshold, saves artifacts
- `predict.py` — full inference over test data, writes both output TSVs
- `eda.py` — quick data-characterization script
