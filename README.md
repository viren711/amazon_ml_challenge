# Business Entity Resolution (Amazon ML Challenge 2026)

This pipeline links each reference business record to every noisy duplicate of it
across two other data sources. It reaches a macro F0.5 of **0.954** on held-out
validation data, trains in about 17 minutes and predicts in about 26 minutes on a
6-core, 14 GB desktop.

---

## 1. Problem

There are three sources of business records. Each record has `entity_id`,
`business_name`, `business_address` and `country`.

| Source | Role | Train rows | Test rows |
|---|---|---|---|
| Source 1 (S1) | clean, deduplicated reference | 2.21 M | 1.73 M |
| Source 2 (S2) | noisy | 5.03 M | 4.89 M |
| Source 3 (S3) | noisy | 5.29 M | 5.08 M |

**Task:** for every S1 entity, list all S2/S3 records that refer to the same
real-world business. The list can be empty (a singleton), have one entry, or have
many (up to 11).

**What makes it hard:**
- typos ("Bilodeau Idnurstgies")
- legal-suffix drift ("Pvt Ltd" / "Private Limited" / missing)
- names replaced by domains ("krishnainvestment.com") or brand names
- reordered, truncated or empty addresses
- state names versus codes ("Arkansas" / "AR")
- **Indian names transliterated into Devanagari, Kannada, Telugu, Gujarati, Bengali or Tamil** while S1 stays in Latin script
- countries: US and India in train; **France appears only in test**
- scale: comparing every pair would mean about 10^13 comparisons

**Metric:** F0.5 per S1 entity, averaged over all S1 entities (macro).
F0.5 = 1.25·P·R / (0.25·P + R), so precision counts twice as much as recall. A
singleton scores 1.0 only if its prediction is empty.

**Rules:**
- no external data, APIs or lookups
- the final model must be MIT or Apache-2.0 licensed and at most 8B parameters

The full statement is in [docs/problem_statement.txt](docs/problem_statement.txt).

**Outputs** (tab-separated, one row per S1 entity):
- `output/matching_results.tsv`: `source1_entity_id`, `matched_entity_ids`. This is the only file scored on the leaderboard.
- `output/candidate_pairs.tsv`: `source1_entity_id`, `candidate_entity_ids`. The blocking candidate set that the model scored.

---

## 2. Solution overview

```
 S1, S2, S3 (one country at a time)
   │
   ├─ 1. normalize + tokenize ──── names, addresses, consonant skeletons → hashed sparse vectors
   │
   ├─ 2. blocking (retrieval) ──── for each S1: score every S2/S3 record by Σ IDF of shared tokens
   │                                keep top 30 by total score + top 10 by name-only score
   │
   ├─ 3. features (45) ─────────── retrieval scores, ranks, reverse ranks, IDF coverage,
   │                                fuzzy string similarities
   │
   ├─ 4. LightGBM matcher ──────── P(match) per candidate pair; threshold tuned on macro F0.5
   │
   └─ 5. exclusivity ───────────── each S2/S3 record is assigned to at most one S1 entity
```

---

## 3. Approach

### 3.1 Key observations from the data

1. **Blocking is the bottleneck.** The first pipeline (v1) scored 0.705, and its
   blocking only ever reached about 72% of the true matches. The classifier was
   already close to that ceiling.
2. **One owner per record.** No S2/S3 record belongs to more than one S1 entity.
   This holds for all 7.6 M training links. If a candidate is a better match for
   another S1, it is almost certainly not ours.
3. **Script changes and truncated addresses often come together.** A Devanagari
   name with a partial address shares almost no tokens with its Latin S1 record,
   so plain token matching cannot find it.
4. **Generic names are shared by many different businesses** ("Green Exports Pvt
   Ltd", "Fiix LLC"). Name tokens that are too common promote the wrong entities.

### 3.2 Normalization and tokens

- **Text normalization:**
  - NFKD, accent stripping, Indic digits → ASCII, `&` → `and`
  - "(ID: nnn)" tags removed
  - legal words unified (private→pvt, limited→ltd, corporation→corp, …)
  - street words unified (street→st, road→rd, highway→hwy, floor→fl, …)
  - US and Indian state names mapped to their codes, including multi-word names such as "west bengal"
- **Name tokens:**
  - the words
  - the concatenated core words, without legal suffixes ("krishnainvestment" matches the domain "krishnainvestment.com")
  - **consonant skeletons.** Each Indic character maps to a Latin fragment derived from its Unicode name ("DEVANAGARI LETTER TTHA" → `tth`; no lookup tables or external data). Both scripts then drop vowels and merge similar sounds (ph→p, th→t, d→t, g→k, …). So "Eastern Trading" and "ईस्टर्न ट्रेडिंग" both become `strn trtnk`.
- **Address tokens:** the words, plus a compact form of every number-bearing piece ("13-B/1" → `13b1`).
- All tokens are hashed into 2^24 columns, which produces sparse binary matrices.

### 3.3 Blocking: IDF-weighted retrieval

- IDF is computed on each country's S2+S3 pool.
- Tokens whose document frequency exceeds the cap are ignored: **2,000 for names and 20,000 for addresses**.
- Candidate score = Σ IDF over shared tokens, reported separately for name and address.
- **Kept per S1:**
  - the **top 30** by total score
  - the **top 10** of the rest by name-only score. Ties are broken by how much of the candidate's own name the shared tokens explain. This recovers records with empty or truncated addresses, which otherwise lose to other businesses at the same address.
- **Implementation:**
  - a numba kernel over an inverted index, with one dense accumulator per thread and in-place quickselect for the top K
  - pool records and queries are both processed in *locality order* (sorted by reversed address: state → city → street), so each thread's working set stays in the CPU cache
  - the kernel is memory-latency bound: locality ordering made it 2.4x faster, and 6 threads beat 12

### 3.4 Features (45)

| Group | Features |
|---|---|
| Retrieval | total / name / address score; score ÷ best candidate's score; gap to the best; margin between the best and second-best; rank within this S1's list; list size; number of near-best candidates; number of records touched |
| Reverse rank | rank of this S1 among every S1 that retrieved this record; score ÷ best claimant's score; number of claimants |
| IDF coverage | shared ÷ total IDF, for both sides and both fields; total IDF of each side |
| Record | source (S2/S3), non-Latin name flag, name and address lengths |
| String similarity (rapidfuzz, multithreaded) | name: ratio, token_set, token_sort, partial, Jaro-Winkler · space-free name: ratio, partial · skeleton: ratio, token_set · address: ratio, token_set, token_sort, partial · house numbers: token_set, exact match, which side is empty |

The most important features by gain:
1. reverse rank
2. score ÷ best claimant's score
3. rank within the S1's list
4. house-number similarity
5. space-free name similarity
6. address token_set
7. skeleton token_set

### 3.5 Matcher and decision

- **LightGBM** binary classifier on the 45 features.
- The decision threshold is grid-searched from 0.05 to 0.95 to **maximize macro F0.5** on held-out validation S1 records, with singletons included.
- **Exclusivity:** if several S1 entities claim the same S2/S3 record above the threshold, only the most probable claimant keeps it.

---

## 4. Parameters

| Stage | Parameter | Value | Where |
|---|---|---|---|
| Tokens | hash space | 2^24 | `NFEAT` |
| Blocking | name document-frequency cap | 2,000 | `--cap-n` |
| | address document-frequency cap | 20,000 | `--cap-a` |
| | top-K by total score | 30 | `--K` |
| | extra top-K by name-only score | 10 | `--Kn` |
| | kernel threads | physical cores (6) | `PHYS` |
| Training data | validation fraction of S1 | 15% | `--val-frac` |
| | share of validation S1 scored | 30% (98,577 S1 / 3.76 M pairs) | `--val-sample` |
| | share of training S1 used | 12% (8.60 M pairs) | `--train-sample` |
| | split seed | 42 | `cmd_train` |
| LightGBM | learning rate / leaves / min child samples | 0.1 / 127 / 100 | `cmd_train` |
| | feature / bagging fraction | 0.8 / 0.5 (every iteration) | |
| | L2 regularization / max bins | 1.0 / 255 | |
| | rounds | up to 800, early stopping 30 → 750 used | `--rounds` |
| Decision | threshold | **0.74** (tuned) | `artifacts/config.json` |
| | exclusivity | on | `--no-exclusive` turns it off |

All learned values (threshold, blocking stats, feature gains) are saved in
[artifacts/config.json](artifacts/config.json). The model is [artifacts/lgbm.txt](artifacts/lgbm.txt).

---

## 5. Results

### 5.1 Approaches compared (validation macro F0.5, 15% of train S1 held out)

| Approach | Blocking | Val macro F0.5 |
|---|---|---|
| A. Rule-based weighted name/address similarity | v1: union of rarest-5 tokens (~200 unranked candidates) | 0.587 |
| B. Logistic regression, 15 features | v1 | 0.591 |
| C. LightGBM, 15 features | v1 | 0.705 |
| **D. LightGBM, 45 features + exclusivity (final)** | **v2: IDF retrieval, 30 + 10 candidates** | **0.954** |

A perfect classifier on D's candidates would score **0.973**, so the matcher
reaches 98% of what the blocking allows.

The leaderboard score of the v1 file was 0.69, consistent with its validation
score. The v2 file is `output/matching_results.tsv`.

### 5.2 Blocking quality (all train S1)

| Country | S1 | Candidates per S1 | Recall of true pairs |
|---|---|---|---|
| US | 1,323,633 | 37.6 | 0.944 |
| India | 883,188 | 39.0 | 0.910 |
| v1 blocker (both) | 2,206,821 | ~200 | ~0.72 |

### 5.3 Blocking experiments

Values are the best possible F0.5 given the candidates, on a 3% sample of train S1
with 50 candidates each.

| Variant | US | India |
|---|---|---|
| caps 1k / 1k | 0.878 | 0.925 |
| caps 2k / 2k | 0.919 | 0.940 |
| caps 5k / 5k | 0.959 | 0.955 |
| **caps 2k name / 20k address (chosen)** | **0.984** | **0.965** |
| caps 2k / 50k | 0.988 | 0.968 (2–4x slower) |
| caps 2k / 2k + address bigrams | 0.979 | 0.961 |
| name cap 20k | worse: generic words promote other businesses | worse |

### 5.4 Test set

| Metric | Value |
|---|---|
| S1 entities | 1,732,544, all present in both files |
| Candidates per S1 | 38.4 on average (66.5 M pairs in total) |
| S1 with at least one match | 1,616,082 |
| Links dropped by exclusivity | 0.23% (France 0.9%, India 0.15%, US 0.06%) |
| Official validator | **PASS**, including `--check-ids` |

### 5.5 Known limitations

- **Records with a generic name and an empty address** are indistinguishable from other businesses with the same name. This is the main remaining source of missed matches.
- **Skeletons diverge for some words:** Tamil voicing, and loanword endings such as "services" → "सर्विसेज".
- **France has no training data.** The model relies on country-independent features there.

---

## 6. Reproducibility

### 6.1 Environment

- Python 3.12, CPU only.
- Reference machine: AMD Ryzen 5 5600G (6 cores / 12 threads, 16 MB L3), 14 GB RAM, Ubuntu.
- Peak RAM is about 10 GB, because only one country is held in memory at a time.

```bash
pip install -r requirements.txt   # pandas, numpy, scipy, scikit-learn, lightgbm, rapidfuzz, numba, llvmlite
```

All dependencies are BSD or MIT licensed. The model is a LightGBM tree
ensemble of about 10 MB, far below the 8B-parameter cap. No pretrained models,
external data or network access are used.

### 6.2 Data

Place the competition data at `student_resource/dataset/{train,test}/`, or point
`DATA` at another location:

```
student_resource/dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
student_resource/dataset/test/test_source{1,2,3}.tsv
```

### 6.3 Run

```bash
scripts/run_pipeline.sh                # train (~17 min) → predict (~26 min) → validate
scripts/run_pipeline.sh predict        # predict only, using the committed artifacts/ model
scripts/run_pipeline.sh validate
scripts/make_submission.sh             # builds submission/team_solution_submission.zip and the leaderboard zip
DATA=/path/to/dataset scripts/run_pipeline.sh
```

Or call the pipeline directly:

```bash
python3 src/er.py train   --data-dir $DATA/train --out-dir artifacts
python3 src/er.py predict --data-dir $DATA/test  --artifacts-dir artifacts --out-dir output
python3 src/er.py diag    --data-dir $DATA/train --frac 0.03 --configs 2000:20000:30:10   # blocking recall study
```

**Determinism:**
- The train/validation split and training sample use a fixed seed (42), and blocking is fully deterministic.
- LightGBM with multiple threads can differ in the last decimals between machines.
- Use the committed `artifacts/lgbm.txt` to reproduce the exact test predictions.

### 6.4 Repository layout

```
.
├── README.md                  this file
├── requirements.txt           pinned dependencies
├── src/
│   ├── er.py                  final pipeline: diag / train / predict
│   └── v1_baseline/           first pipeline (approaches A–C), kept for comparison
├── scripts/
│   ├── run_pipeline.sh        train → predict → validate
│   └── make_submission.sh     packages the required submission zip + leaderboard zip
├── artifacts/
│   ├── lgbm.txt               trained LightGBM model
│   └── config.json            threshold, parameters, blocking stats, feature gains
├── docs/
│   ├── methodology.md         filled-in competition documentation template
│   ├── problem_statement.txt  task description
│   └── eda/                   exploratory analysis notes and output
├── student_resource/          organizer files: README, validator, (dataset/ – not tracked)
├── output/                    generated TSVs              (not tracked)
├── submission/                generated zips              (not tracked)
└── archive/                   v1 model/outputs, old READMEs (v1/ not tracked)
```

The submission zip is assembled by `scripts/make_submission.sh` in the layout the
organizers require: `output/`, `code/business_entity_resolution/{src,README.md,requirements.txt}`
and `Documentation_template.md`.
