# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** solution
**Team Members:** N/A
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We resolve each Source-1 business against noisy Source-2/3 records in two stages.

1. **IDF-weighted sparse retrieval blocking.** Every S1 record scores every pool
   record of its country by the summed IDF of the name and address tokens they
   share. It keeps the top 30 candidates plus the 10 best name-only candidates.
2. **LightGBM matcher.** It scores 45 features per candidate pair. The decision
   threshold is tuned directly on validation macro F0.5, and at prediction time
   each pool record is assigned to at most one S1 entity.

On a held-out validation split of train this reaches **macro F0.5 = 0.954**. The
first pipeline scored 0.705. That pipeline was capped by its blocker, whose recall
ceiling was about 0.72; the new blocker's ceiling is 0.973. A full train or test
run takes about 15 minutes on a 6-core desktop with 14 GB RAM.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on train (2.2M S1, 5.0M S2, 5.3M S3, 7.6M true links) showed:

- 5.6% of S1 entities are singletons. The rest have 3.7 matches on average (max 11), split almost evenly between S2 and S3.
- **Each S2/S3 record belongs to at most one S1 entity.** This holds for 100% of the 7.6M ground-truth links. It is the single most useful structural fact: a candidate that is a better match for a different S1 is almost certainly not ours.
- Noise patterns seen in the data:
  - legal-suffix drift ("Pvt Ltd" / "Private Limited" / missing)
  - heavy character typos ("Bilodeau Idnurstgies")
  - name replaced by a domain ("krishnainvestment.com")
  - name replaced by an unrelated brand ("calonexflux", "X f/k/a Y")
  - addresses reordered, truncated or empty
  - state names versus codes ("Arkansas" / "AR", "Maharashtra" / "MH")
  - **Indian S2/S3 names transliterated into Devanagari, Kannada, Telugu, Gujarati, Bengali or Tamil** while S1 stays in Latin script
- Country is given and never crosses between matches, so everything is partitioned by country. France appears only in test.
- Many *different* synthetic businesses share generic names ("Green Exports Pvt Ltd", "Fiix LLC"). When such a record also has an empty address, it cannot be told apart from the others. This sets the practical recall ceiling.

### 2.2 Solution Strategy

**Approach type:** blocking (candidate retrieval) followed by a classifier, plus exclusivity post-processing.

**Core innovations:**
1. **Ranked retrieval instead of unranked token-key blocking.** Each candidate is ranked by summed IDF over *all* reasonably discriminative shared tokens, so the ranking itself carries the match signal.
2. **Script-independent consonant skeletons.** These let Indic-script names match their Latin S1 names.
3. **Reverse-rank features plus hard exclusivity.** These exploit the one-owner-per-record structure of the data.

---

## 3. Candidate Generation (Blocking)

- **Normalization:**
  - NFKD with combining marks stripped (accents), Indic digits mapped to ASCII, `&` → `and`
  - "(ID: nnn)" tags removed
  - legal words unified (private→pvt, limited→ltd, corporation→corp, …)
  - address abbreviations unified (street→st, road→rd, highway→hwy, floor→fl, …)
  - all US and Indian state names mapped to their codes, including multi-word names such as "west bengal" and "tamil nadu"
- **Tokens (hashed into 2^24 buckets):**
  - Name tokens: the words themselves, a concatenation of the non-legal core words ("krishnainvestment", which catches domain-style names), and consonant skeletons.
  - Skeletons are built by mapping each Indic character to a Latin fragment, derived from its Unicode character name ("DEVANAGARI LETTER TTHA" → "tth"; vowels dropped, anusvara → n). Both scripts then pass through the same rules: vowels removed; digraphs and voicing merged (ph→p, th→t, d→t, g→k, …); repeats collapsed. So "Eastern Trading" and "ईस्टर्न ट्रेडिंग" both give `strn trtnk`.
  - Address tokens: the words themselves plus a compacted form of each number-bearing piece ("13-B/1" → `13b1`).
- **Scoring:** IDF is computed on the country's pool (S2+S3). Tokens with document frequency above 2,000 (names) or 20,000 (addresses) are ignored. Score = Σ IDF over shared tokens.
- **Kept per S1:**
  - the top 30 by total score
  - the top 10 of the remaining candidates by *name-only* score, with ties broken by how much of the candidate's own name the shared tokens explain

  The name-only list recovers pool records with empty or truncated addresses, which otherwise lose the total-score race to businesses at the same address.
- **Implementation:**
  - a numba kernel runs over an inverted index, with one dense score accumulator per thread and in-place quickselect for the top K
  - pool records and queries are renumbered and processed in "locality order" (sorted by reversed address: state, city, street). Each thread's working set then stays in cache, which made the kernel 2.4x faster; the kernel is memory-latency-bound on this machine
  - 6 threads (physical cores) turned out faster than 12
  - the whole train set (2.2M queries against 10.3M pool records) is blocked in about 6 minutes
- **Measured blocking quality (all train S1):**

  | Country | S1 | Candidates / S1 | Pair recall |
  |---|---|---|---|
  | US | 1,323,633 | 37.6 | 0.944 |
  | India | 883,188 | 39.0 | 0.910 |

  - Oracle macro F0.5 (a perfect classifier on these candidates) on validation: **0.973**
  - Reduction ratio vs. all pairs within a country: > 99.999%
  - Test: 1.73M S1, about 38 candidates each

- **Blocking design experiments.** Oracle F0.5 is measured on a 3% sample of train S1 at 50 candidates.

  | Variant | US | India |
  |---|---|---|
  | v1: union of rarest-5 tokens (unranked, ~200 cand.) | 0.72 pair recall (both countries) | — |
  | IDF retrieval, caps 2k/2k | 0.919 | 0.940 |
  | caps 1k/1k | 0.878 | 0.925 |
  | caps 5k/5k | 0.959 | 0.955 |
  | **caps 2k name / 20k address** | **0.984** | **0.965** |
  | caps 2k / 50k (2–4x slower) | 0.988 | 0.968 |
  | + address bigrams (2k/2k) | 0.979 | 0.961 |
  | name cap 20k | worse: generic name words promote other entities | — |

---

## 4. Matching Model

**Features (45):**

- **Retrieval features:**
  - total, name and address scores
  - score relative to the S1's best candidate, gap to the best, margin between the best and second-best
  - rank within the S1's list, list size, number of near-best candidates
  - **reverse rank:** the candidate's rank among all S1s that retrieved it, plus its score relative to that best claimant
  - number of pool records touched
- **IDF coverage:** shared / total IDF, for each side and each field.
- **Record features:** source (S2/S3), non-Latin-script name flag, lengths.
- **rapidfuzz similarities (multithreaded `cpdist`):**
  - names: ratio, token_set, token_sort, partial, Jaro-Winkler
  - space-free names: ratio, partial (for domains and concatenations)
  - consonant skeletons: ratio, token_set (for Indic-script names)
  - addresses: ratio, token_set, token_sort, partial
  - house-number sets: token_set, exact equality, which side is empty

**Model type:** LightGBM binary classifier (MIT license, 127 leaves, learning rate 0.1, 750 rounds, early stopping). It is a model of a few MB, far below the 8B-parameter limit. Training used every candidate of a random 12% of train S1 (8.6M pairs).

**Threshold selection:** grid search over 0.05–0.95, maximizing macro F0.5 on the held-out validation S1s (15% of train, never trained on). Singletons count as 1.0 when predicted empty. Selected threshold: 0.74.

**Post-processing:** a pool record that clears the threshold for several S1s is kept only for the most probable one. This dropped 0.23% of predicted links on test (0.9% in France, 0.15% in India, 0.06% in the US).

**Comparison table (validation split of train, macro F0.5):**

| Approach | Blocking | Val macro F0.5 |
|---|---|---|
| A. Rule-based weighted name/address similarity | v1 rare-token union | 0.587 |
| B. Logistic regression, 15 features | v1 | 0.591 |
| C. LightGBM, 15 features | v1 | 0.705 |
| **D. LightGBM, 45 features (final)** | **v2 IDF retrieval** | **0.954** |

Approaches A–C were limited by the v1 blocker's ~0.72 recall ceiling. D's blocker raised the oracle ceiling to 0.973, and the classifier reaches 98% of that ceiling.

---

## 5. Results & Error Analysis

- **F0.5 score (macro, validation):** **0.954**. The oracle ceiling at the chosen blocking settings is 0.973.
- **Most important features (gain):**
  1. reverse rank of the candidate among its claimants
  2. score relative to the best claimant
  3. rank within the S1's list
  4. house-number similarity
  5. compact-name similarity
  6. address token-set similarity
  7. skeleton similarity
- **Common false negatives:**
  - pool records whose name is generic and shared by many entities *and* whose address is empty. Indistinguishable in principle.
  - Indic-script names with truncated addresses, where the skeletons diverge (Tamil voicing, loanword endings such as "-ises" → "-ij")
  - heavily typo'd names whose address is also missing
- **Common false positives:**
  - different businesses at the same address, in the same building or on the same street
  - businesses with the same generic name in the same city
  - the exclusivity constraint and reverse-rank features remove many of these
- **Bugs found and fixed during development:**
  - (v1) a pandas index-aligned assignment silently NaN-filled features for part of every chunk
  - (v2) scipy's `eliminate_zeros()` compacts arrays *in place*, and those arrays were shared with the stored token matrices. The first blocking call was correct, but every later call silently ran on corrupted data. This first showed up as a misleading "higher IDF cap hurts recall" result, which was caught by checking the matmul against brute-force dot products.

---

## 6. Conclusion

The dominant factor was candidate generation. Replacing unranked rare-token keys with ranked IDF retrieval raised the recall ceiling from 0.72 to 0.97. After that, the biggest gains came from structure in the data rather than string similarity: one owner per pool record (reverse-rank features and exclusivity) and script-independent name skeletons for Indian records. Engineering for the hardware mattered too. The retrieval kernel is memory-latency-bound, so locality ordering and thread count gave larger speedups than adding cores.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- `src/er.py`: the complete final pipeline (`train`, `predict`, `diag` commands)
- `src/v1_baseline/`: the first pipeline, used for approaches A–C
- `README.md`: exact reproduction commands
- `requirements.txt`: pinned versions

### B. Additional Results

- Validation threshold sweep and feature gains: `artifacts/config.json` (written by `er.py train`)
- Test run: 1,732,544 S1 records, all present in both output files; 38.4 candidates each on average; 1,616,082 with at least one match; validator PASS (including `--check-ids`)
