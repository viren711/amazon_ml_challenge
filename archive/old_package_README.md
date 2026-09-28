# Business Entity Resolution: reproduction guide

The final pipeline is `src/er.py`, a single file. `src/v1_baseline/` holds the first
pipeline (rare-token union blocking with an unranked candidate list), kept so the
approach comparison in `Documentation_template.md` can be reproduced.

## Environment

Python 3.12, CPU only. Tested on a 6-core / 12-thread Ryzen 5 5600G with 14 GB RAM.

```bash
pip install -r requirements.txt
```

Every dependency has a permissive license (BSD or MIT). The matcher is a LightGBM
gradient-boosted tree model of a few MB, far below the 8B-parameter cap. It uses no
external data, APIs or lookups: the Indic transliteration table is derived at runtime
from Python's built-in `unicodedata` character names.

## Run end-to-end (data → blocking → matching → output)

Run from `src/`. `DATA` is the `dataset/` folder that contains `train/` and `test/`.

```bash
cd src
# 1. train: blocks all of train, builds features, trains LightGBM, tunes the threshold (~17 min)
python3 er.py train   --data-dir $DATA/train --out-dir ../artifacts

# 2. predict: blocks all of test, scores every candidate, writes both output files (~15 min)
python3 er.py predict --data-dir $DATA/test --artifacts-dir ../artifacts --out-dir ../output

# optional: blocking-only recall diagnostics on a sample of train S1
python3 er.py diag    --data-dir $DATA/train --frac 0.03 --configs 2000:20000:30:10
```

`predict` writes:
- `output/matching_results.tsv`: final matches (`source1_entity_id`, `matched_entity_ids`)
- `output/candidate_pairs.tsv`: every blocking candidate fed to the model (`source1_entity_id`, `candidate_entity_ids`)

Both files list every S1 entity exactly once. An empty list means no match or no candidate.

## Pipeline stages

1. **Load and tokenize** (12 worker processes, one country at a time). Names and
   addresses are normalized: NFKD, accent stripping, Indic digits to ASCII, and
   state and street-suffix abbreviations. Tokens are then hashed into sparse
   matrices. Names also get a script-independent consonant skeleton, so
   "Eastern Trading" and "ईस्टर्न ट्रेडिंग" both become `strn trdng`.
2. **Blocking**. Each S1 record scores every pool (S2+S3) record of its country by
   the summed IDF of shared name and address tokens. Tokens are capped at document
   frequency 2,000 for names and 20,000 for addresses. It keeps the top 30 by total
   score plus the top 10 of the rest by name-only score. A numba kernel does this in
   parallel on an inverted index, and pool and query records are sorted by locality
   so memory access stays cache-friendly.
3. **Features** (45). Blocking scores; rank and reverse-rank within each S1's
   candidates and within each pool record's claimants; IDF coverage;
   multithreaded rapidfuzz similarities on names, compact names, skeletons,
   addresses and house numbers.
4. **Matcher**. LightGBM, with the threshold tuned on held-out validation
   macro F0.5. At prediction time each pool record is assigned to at most one S1
   (its most probable claimant). That is a hard constraint in the training ground
   truth.

Peak RAM stays around 10 GB because only one country is in memory at a time.
