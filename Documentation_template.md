# ML Challenge 2026: Business Entity Resolution Solution Documentation

**Team Name:** Bugs2  
**Team Members:** Jaishree Verma, Anshika Pandey  
**Submission Date:** September 2026  

---

## 1. Executive Summary

We developed a high-precision, scalable Business Entity Resolution (ER) system designed to match noisy, multi-source records ($S_2, S_3$) to canonical reference entities ($S_1$) while directly optimizing for the competition evaluation metric **Macro $F_{0.5}$** ($\beta = 0.5$). 

### End-to-End Pipeline Architecture

```
Raw S1/S2/S3
→ Unicode normalization
→ Name/address normalization
→ Cross-script transliteration
→ Variant C blocking
→ Candidate ranking
→ 30 pairwise features
→ LightGBM
→ threshold 0.80
→ final matching results
```

On our held-out validation benchmark (2,500 $S_1$ entities, 8,743 ground-truth links), the pipeline achieves **96.88% Candidate Recall**, **99.30% Pair Precision**, **94.76% Pair Recall**, **0.9747 Macro $F_{0.5}$**, and **83.36% Exact Entity Set Accuracy**.

---

## 2. Methodology & Architecture Details

### A. Data
- **Three Independent Sources**: The problem operates over three distinct entity datasets: Source-1 ($S_1$), Source-2 ($S_2$), and Source-3 ($S_3$).
- **$S_1$ as Canonical Reference**: Source-1 serves as the ground-truth reference entity catalog. Every test evaluation requires an entity-level decision for each record in $S_1$.
- **$S_2$ and $S_3$ as Potential Matches**: Records in $S_2$ and $S_3$ are noisy external observations that may link to one or more $S_1$ entities.
- **Strict Country Isolation**: Empirical analysis verified that true entity matches never cross country borders (US, India, France). All indexing, blocking, and inference steps are strictly partitioned by country with zero cross-country leakage.

### B. Normalization
- **Unicode-Aware Normalization**: Applies NFKD Unicode normalization, lowercase folding, accent removal, punctuation stripping, and whitespace collapse.
- **Legal Suffix Handling**: Systematically identifies and strips common corporate and legal business designators (`pvt ltd`, `private limited`, `inc`, `llc`, `corp`, `gmbh`, `sa`, `sarl`, etc.) to produce clean "core names" while retaining full normalized names for secondary verification.
- **Original + Transliterated Representations**: Dual phonetic representations are created using `anyascii` for non-Latin scripts (e.g., Devanagari, Tamil, Bengali).
- **Preservation of Non-Latin Information**: Native-script characters and language structures are preserved alongside transliterations, ensuring that native-to-native matches retain full signal while native-to-Latin cross-script variants are seamlessly bridged.

### C. Blocking (Candidate Generation)
Our blocking strategy implements **Variant C Adaptive Inverted Index Blocking** with address fallback and dynamic candidate ceilings:
- **Exact Normalized / Core-Name Keys**: Inverted index matching alphanumeric normalized core business names.
- **Transliterated Name Keys**: Phonetic core names and transliterated rare tokens bridging multi-script representations.
- **Prefix / Token Keys**: 5-character alphanumeric prefixes and ordered 2-token bigrams (`token1_token2`).
- **PIN / Locality Keys**: Postal PIN codes paired with 2-character name prefixes, and locality tokens paired with name prefixes.
- **Address-Number / Locality Fallback**: For challenging records where postal PIN is missing or corrupted, secondary index keys combine street numbers with locality tokens and core name tokens.
- **Adaptive Retrieval**: Expands candidate exploration dynamically up to a strict ceiling of **40 candidates per entity** only when candidate density is low or non-Latin script divergence is detected.

### D. Features (30 Dense Pairwise Features)
The feature engineering layer transforms each candidate pair $(S_1, \text{cand})$ into a deterministic 30-dimensional real-valued feature vector:

1. **Name Similarity (9 features)**:
   - `name_sort_ratio`: Token sort fuzzy ratio between normalized names.
   - `name_set_ratio`: Token set fuzzy ratio handling word reordering.
   - `name_ratio`: Full string Levenshtein similarity ratio.
   - `name_partial_ratio`: Substring containment alignment ratio.
   - `core_exact`: Binary indicator for exact alphanumeric core name equality.
   - `token_jaccard`: Jaccard intersection over union of name tokens.
   - `token_intersection`: Absolute count of overlapping name tokens.
   - `name_len_diff`: Absolute character length difference.
   - `name_len_ratio`: Ratio of shorter name length to longer name length.

2. **Transliteration & Cross-Script (6 features)**:
   - `jaro_winkler_similarity`: Jaro-Winkler distance sensitive to prefix alignments.
   - `token_containment`: Degree to which smaller token set is entirely contained in larger set.
   - `first_token_match`: Binary flag indicating agreement on the primary brand token.
   - `acronym_match`: Binary match between entity initialisms/acronyms.
   - `translit_name_ratio`: Cross-script phonetic similarity score via `anyascii`.
   - `cross_script_flag`: Binary indicator signaling non-Latin script pairing.

3. **Address Similarity (6 features)**:
   - `addr_sort_ratio`: Token sort ratio on normalized address strings.
   - `addr_set_ratio`: Token set ratio on address tokens.
   - `addr_ratio`: Direct string similarity ratio on addresses.
   - `addr_jaccard`: Jaccard similarity across whitespace-delimited address tokens.
   - `cand_addr_empty`: Binary indicator for missing candidate address.
   - `addr_len_ratio`: Ratio of address string lengths.

4. **Numeric / PIN Information (3 features)**:
   - `num_overlap`: Count of overlapping numeric sequences in addresses.
   - `has_common_num`: Binary indicator for shared street/building numbers.
   - `pin_match_status`: Tri-state feature (+1 match, -1 mismatch, 0 missing).

5. **Locality (2 features)**:
   - `house_num_status`: Tri-state street/house number agreement.
   - `locality_jaccard`: Jaccard similarity on extracted city/state locality tokens.

6. **Candidate & Blocking Signals (4 features)**:
   - `source_is_s2`: Binary indicator for Source-2 origin.
   - `source_is_s3`: Binary indicator for Source-3 origin.
   - `cand_heuristic_score`: Composite blocking heuristic score.
   - `cand_rank_normalized`: Relative rank of candidate within blocking candidate pool.

### E. Model Configuration
A gradient-boosted decision tree classifier is trained on pairwise feature vectors to output match probabilities:
- **Model Type**: LightGBM Binary Classifier (`LGBMClassifier`)
- **Objective**: `binary_logloss`
- **Number of Estimators**: `n_estimators = 220`
- **Learning Rate**: `learning_rate = 0.08`
- **Number of Leaves**: `num_leaves = 31`
- **Maximum Depth**: `max_depth = 6`
- **Subsample Ratio**: `subsample = 0.85`
- **Colsample by Tree**: `colsample_bytree = 0.85`
- **Decision Threshold**: $\tau = 0.80$ (selected to optimize Macro $F_{0.5}$ under strict precision weighting).

### F. Validation Results

> [!IMPORTANT]
> The metrics below are strictly **VALIDATION results** measured on our held-out 2,500 $S_1$ benchmark (8,743 ground-truth links) with seed 42. These are NOT test-set or leaderboard scores.

| Metric | Validation Benchmark Score |
| :--- | :--- |
| **Candidate Recall** | **96.88%** |
| **Pair Precision** | **99.30%** |
| **Pair Recall** | **94.76%** |
| **Pair F1-Score** | **96.98%** |
| **Macro $F_{0.5}$** | **0.9747** |
| **Exact Entity Accuracy** | **83.36%** |
| **Singleton Accuracy** | **97.14%** (Only 4 false merges out of 140 singletons) |

---

## 3. Submission Integrity & Code Artefacts

### Directory Structure

```
Bugs2_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── normalize.py
│       │   ├── blocking.py
│       │   ├── features.py
│       │   └── metrics.py
│       ├── execute_model.py
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

### Reproducibility & Compliance
- **Submission Output Files**: `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- **Validation Compliance**: Verified using the official validator (`utils/validate_submission.py`) across all 1,732,544 test $S_1$ entities.
- **Candidate Requirement Guarantee**: Every predicted match in `matching_results.tsv` is strictly guaranteed to exist within the corresponding candidate set in `candidate_pairs.tsv`.
