# EDA notes (train data)

Generated from `code/business_entity_resolution/src/eda.py` — see
`eda_output.txt` in this folder for the full run.

- Row counts: S1 = 2,206,821 (reference, one row per real entity),
  S2 = 5,034,616, S3 = 5,285,603, ground truth rows = 2,206,821 (one per S1 id).
- Country distribution is US-majority, India second, in all three sources
  (roughly 60/40 US/India split); test additionally introduces France, unseen
  in train — the pipeline must not hardcode to {US, India}.
- Singleton rate: 123,247 / 2,206,821 = **5.58%** of S1 entities have zero
  matches — matches the ~5.6% figure in the problem statement.
- Match cardinality among non-singletons: mean **3.666** matches, mode 3,
  max 11. Matches split almost evenly between S2 (3,693,619) and S3
  (3,944,746).
- Noise patterns observed directly in matched pairs:
  - Legal-suffix / abbreviation drift ("Inc" appended/dropped, "LLC" vs none).
  - Heavy character-level typos in names ("Payne Enterpires", "PAYNE-ENRTPRMISES",
    "Dahlia Ponr Reliable Scientific LLC") while the address stays exact or
    near-exact — address blocking is essential to recover these.
  - Completely different-looking names sharing an address with typos
    (e.g. S1 "Maure Williams Colombier Inc" <-> S3 "Dréxkor" at
    "85 Wanye Avenue, Ticonderoga Townshiip, New York") — a pure name-similarity
    approach would miss this; only shared address tokens ("wayne"/"wanye" via
    fuzzy match is even hard, but "ticonderoga", "avenue", street-number if
    present) recover it.
  - Domain-name-as-business-name records ("maurewilliamscolombier.com").
  - India records: S2/S3 frequently transliterate the business name into
    Devanagari (Hindi) or other Indic scripts (Tamil observed too) while the
    address stays in Latin script and matches almost verbatim — again, address
    tokens (including PIN codes, which survive verbatim as numeric tokens) are
    the reliable link when the name script differs.
  - Address component reordering ("KANSAS CITY, MO, 630 45ND TERRACE" vs
    "630 45th Terrace, Kansas City, MO") and missing components (empty address
    string in some S2 rows) are common — token-set based matching (not
    substring/prefix matching) is required to be robust to reordering.

These observations directly motivated the blocking design (union of name-token
and address-token inverted indices, since either side can be the "noisy" one)
and the feature set (token-set overlap / Jaccard on both name and address,
not just edit-distance ratios which are reorder-sensitive).
