# Amazon ML Challenge: Business Entity Resolution

For each Source-1 business, find every Source-2/3 record that refers to the same
real-world business. The records are noisy: typos, abbreviations, reordered or
missing addresses, and names transliterated into Indic scripts. Scoring is macro
F0.5 per S1 entity (precision weighted 2x). Singletons score 1.0 only when
predicted empty.

## Result

| Approach | Val macro F0.5 |
|---|---|
| A. Rule-based name/address similarity (v1 blocking) | 0.587 |
| B. Logistic regression, 15 features (v1 blocking) | 0.591 |
| C. LightGBM, 15 features (v1 blocking) | 0.705 |
| **D. IDF-retrieval blocking + LightGBM, 45 features + exclusivity (final)** | **0.954** |

The score is measured on 15% of train S1 records held out from training, with
threshold 0.74. A perfect classifier on the final candidate set would score 0.973;
v1's blocking capped every approach at about 0.72.

## Final approach

1. **Blocking as ranked retrieval.** Each S1 record scores every S2+S3 record of
   its country by the summed IDF of the name and address tokens they share.
   Tokens are ignored above document frequency 2,000 (names) or 20,000 (addresses).
   It keeps the top 30, plus the top 10 by name-only score. The name-only list
   recovers records with empty or truncated addresses, which otherwise lose to
   neighbouring businesses at the same address.
2. **Script-independent name skeletons.** Indian S2/S3 names often appear in
   Devanagari, Kannada, Telugu, Gujarati, Bengali or Tamil script. Each Indic
   character maps to a Latin consonant fragment derived from its Unicode name,
   and both scripts reduce to a consonant skeleton. "Eastern Trading" and
   "ईस्टर्न ट्रेडिंग" both become `strn trtnk`.
3. **The one-owner structure.** Every S2/S3 record belongs to at most one S1
   entity (true for 100% of the 7.6M training links). Reverse-rank features
   ("is this S1 the best claimant of this record?") are the model's most
   important features. At prediction time each record is kept only for its most
   probable claimant.
4. **Matcher.** LightGBM on 45 features:
   - retrieval scores, ranks and reverse ranks
   - IDF coverage
   - rapidfuzz similarities on names, space-free names, skeletons, addresses and house numbers

   The threshold is tuned directly on validation macro F0.5.

The pipeline is fast and memory-safe on a 6-core, 14 GB machine:
- Data is processed one country at a time.
- Tokenization uses 12 worker processes.
- Retrieval is a numba kernel over an inverted index. Records are ordered by
  locality so each thread's memory stays in cache.
- String similarity uses rapidfuzz's multithreaded pairwise scorer.

Train (all 2.2M S1) takes about 17 minutes; test prediction about 26 minutes.
No external data, APIs or pretrained models are used.

## Test run

- 1,732,544 S1 records; 38.4 candidates each on average; 1,616,082 with at least one match.
- Validator: **PASS**, including `--check-ids`.

## Layout

- `code/business_entity_resolution/`: runnable pipeline. `src/er.py` is the final version and `src/v1_baseline/` holds approaches A–C. Also contains `README.md` (reproduction steps) and `requirements.txt`.
- `output/`: final `matching_results.tsv` and `candidate_pairs.tsv`.
- `artifacts/`: final LightGBM model, `config.json` (threshold, blocking stats, feature gains) and test-set pair probabilities.
- `artifacts_v1/`, `output_v1/`: the earlier v1 model and its outputs.
- `eda/`: exploratory analysis.
- `Documentation_template.md`: full methodology write-up.
- `team_solution_submission.zip`: the submission package.
- `matching_results_leaderboard_upload.zip`: matching_results.tsv only, for the portal (under 51 MB).
