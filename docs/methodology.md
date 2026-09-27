# Business Entity Resolution Methodology

## Amazon ML Challenge 2026

### 1. Executive Summary & Pipeline Overview
This document specifies the end-to-end production pipeline architecture for the Amazon ML Challenge 2026 Business Entity Resolution task. The objective is to resolve canonical reference business entities ($S_1$) against noisy records ($S_2, S_3$) across multiple countries (United States, India, France) while maximizing Macro $F_{0.5}$ (prioritizing precision with $\beta = 0.5$).

The final production pipeline is structured as follows:
```
Raw Records (S1, S2, S3)
   │
   ▼
[1] Text Normalization & Cleaning (Unicode NFKD, legal entity suffix stripping, digit isolation)
   │
   ▼
[2] Cross-Script Transliteration (AnyAscii phonetic mapping for Devanagari, Tamil, Bengali, etc.)
   │
   ▼
[3] Multi-Stage Variant C Adaptive Blocking (Dual-script inverted indices, address fallback, max 40 candidates)
   │
   ▼
[4] Dense 30-Feature Pair Representation (Fuzzy string ratios, token sets, Jaro-Winkler, PIN/locality Jaccard)
   │
   ▼
[5] LightGBM Classifier (220 trees, max_depth=6, colsample=0.85, subsample=0.85, binary logloss)
   │
   ▼
[6] High-Precision Decision Thresholding (τ = 0.80)
   │
   ▼
Final Output Artifacts (output/matching_results.tsv, output/candidate_pairs.tsv)
```

---

### 2. Pipeline Components

#### Component 1: Text Normalization
* **Unicode Normalization**: NFKD normalization, lowercasing, and removal of non-alphanumeric noise while preserving international alphabets.
* **Legal Suffix Harmonization**: Strips corporate designators (e.g., `pvt ltd`, `inc`, `corp`, `llc`, `sa`, `sarl`) to extract the distinctive `core_name`.
* **Address Normalization**: Isolates postal PIN codes (5-6 digits), street numbers, and locality tokens while removing common address stopwords (`road`, `street`, `floor`, `suite`, `rue`, `boulevard`).

#### Component 2: Cross-Script Transliteration
* Detects non-Latin scripts (Devanagari, Tamil, Telugu, Bengali, Kannada, Malayalam, Gujarati).
* Employs deterministic phonetic transliteration using `anyascii` to bridge cross-script matches (e.g., `নর্থ গ্লোবাল` $\to$ `North Global`).
* Generates dual representations (`clean_name`, `trans_clean`, `core_name`, `trans_core`) for candidate generation and feature extraction.

#### Component 3: Variant C Adaptive Blocking
* **Primary Inverted Indices**:
  * Exact alphanumeric core name index
  * 5-character core name prefix index
  * 2-token ordered name bigram index
  * Rare name token index (frequency $\le 35$)
  * Transliterated core and transliterated rare token indices
  * Address number + locality token index and PIN code index
* **Address & Locality Fallback**:
  * Activated when postal PIN codes are missing or candidate density is sparse ($< 10$).
  * Retrieves candidates via house number + locality tokens and name token + house number keys.
* **Adaptive Candidate Budget**:
  * Base budget: `top_k = 30`.
  * Adaptive ceiling: `maximum 40 candidates / S1 entity` for difficult records.
  * Candidate set remains frozen and guarantees $\text{matches} \subseteq \text{candidates}$.

#### Component 4: 30-Feature Pair Representation
The feature vector maps candidate pairs into a 30-dimensional metric space:
1. `name_sort_ratio`: Token sort ratio between clean names
2. `name_set_ratio`: Token set ratio between clean names
3. `name_ratio`: Levenshtein ratio between clean names
4. `name_partial_ratio`: Substring alignment score
5. `core_exact`: Exact match indicator of core legal-stripped names
6. `token_jaccard`: Jaccard similarity over business name tokens
7. `token_intersection`: Absolute count of overlapping name tokens
8. `name_len_diff`: Absolute difference in character lengths
9. `name_len_ratio`: Ratio of shorter to longer name length
10. `addr_sort_ratio`: Token sort ratio between addresses
11. `addr_set_ratio`: Token set ratio between addresses
12. `addr_ratio`: Levenshtein ratio between addresses
13. `addr_jaccard`: Jaccard similarity over address tokens
14. `num_overlap`: Count of shared numeric tokens (street/building numbers)
15. `has_common_num`: Boolean flag for presence of shared numeric token
16. `cand_addr_empty`: Indicator flag if candidate address is missing
17. `source_is_s2`: Source indicator for S2
18. `source_is_s3`: Source indicator for S3
19. `jaro_winkler_similarity`: Jaro-Winkler metric with prefix weighting
20. `token_containment`: Maximum token containment ratio
21. `first_token_match`: Indicator for matching initial name tokens
22. `acronym_match`: Acronym/initialism match indicator
23. `translit_name_ratio`: Name similarity evaluated on transliterated strings
24. `cross_script_flag`: Indicator whether either record was transliterated
25. `pin_match_status`: Postal PIN code agreement (+1 match, -1 mismatch, 0 unobserved)
26. `house_num_status`: Street number agreement (+1 match, -1 mismatch, 0 unobserved)
27. `locality_jaccard`: Jaccard similarity over non-numeric locality tokens
28. `addr_len_ratio`: Ratio of address character lengths
29. `cand_heuristic_score`: Maximum of name and address similarity scores
30. `cand_rank_normalized`: Normalized candidate rank from blocking stage

#### Component 5: Machine Learning Model (LightGBM)
* **Architecture**: Gradient Boosted Decision Trees (`LGBMClassifier`)
* **Hyperparameters**:
  * `n_estimators`: 220
  * `learning_rate`: 0.08
  * `num_leaves`: 31
  * `max_depth`: 6
  * `subsample`: 0.85
  * `colsample_bytree`: 0.85
  * `objective`: `binary_logloss`
  * `random_state`: 42

#### Component 6: High-Precision Decision Thresholding
* **Threshold**: $\tau = 0.80$
* Selected via threshold sweeps to optimize Macro $F_{0.5}$ (which heavily weights precision $\beta=0.5$).
* Eliminates singleton false positives and spurious cross-entity matches.

---

### 3. Validated Benchmark Performance (Validation Set)

> [!NOTE]
> All metrics below are strictly **VALIDATION results** measured on the held-out 2,500 $S_1$ validation entities (8,743 ground-truth links) with seed 42. They are NOT test results.

| Metric | Measured Validation Value |
| :--- | :--- |
| **Candidate Recall** | **96.88%** |
| **Macro $F_{0.5}$** | **0.9747** |
| **Pair Precision** | **99.30%** |
| **Pair Recall** | **94.76%** |
| **Pair F1-Score** | **96.98%** |
| **Exact Entity Accuracy** | **83.36%** |
| **Average Candidates / S1** | **24.48** |
| **Maximum Candidates / S1** | **40** |
| **Candidate Reduction Ratio** | **> 99.999%** |

---

### 4. Country Partitioning & Execution Strategy
Cross-country ground truth analysis proved $0.0000\%$ cross-country matches across 7.6M pairs. Test inference is executed partitioned by country (`France`, `US`, `India`):
1. Loads test $S_1$ and candidates per country.
2. Builds `AdaptiveBlocking` indices per country.
3. Computes 30 features in vectorized batches.
4. Emits candidate pairs and match predictions directly into output TSVs.
5. Frees memory after each partition to ensure stable memory utilization.
