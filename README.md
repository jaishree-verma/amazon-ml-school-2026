# Amazon ML Challenge 2026 - Business Entity Resolution

High-quality, scalable Business Entity Resolution pipeline for matching noisy, multi-source business records against a reference dataset to maximize **Macro $F_{0.5}$** (precision-weighted, $\beta = 0.5$).

---

## Validated Benchmark Results (Validation Set)

> [!NOTE]
> All metrics below are strictly **VALIDATION results** measured on the held-out 2,500 $S_1$ validation entities (8,743 ground-truth links) with seed 42. They are NOT test results.

- **Candidate recall**: **96.88%**
- **Macro F0.5**: **0.9747**
- **Pair precision**: **99.30%**
- **Pair recall**: **94.76%**
- **Exact entity accuracy**: **83.36%**

---

## 🏗️ Production Pipeline Architecture

The final production pipeline follows a deterministic multi-stage design:

```
Normalization
  → Cross-script transliteration
  → Variant C blocking
  → Candidate ranking
  → 30-feature pair representation
  → LightGBM
  → threshold 0.80
  → final matching_results.tsv
```

### Pipeline Details:

1. **Normalization**:
   - Universal Unicode NFKD normalization and lowercasing.
   - Legal entity suffix removal (`pvt ltd`, `inc`, `llc`, `corp`, `sarl`, `sa`, etc.) to isolate the distinctive `core_name`.
   - Address token cleaning, postal code extraction (PIN), and street number isolation.

2. **Cross-Script Transliteration**:
   - Non-Latin script detection for Indic languages (Devanagari, Tamil, Bengali, Telugu, Kannada, Malayalam, Gujarati).
   - Deterministic phonetic transliteration using `anyascii` to bridge cross-script pairs.

3. **Variant C Adaptive Blocking**:
   - Multi-key inverted indexing (exact core name, 5-char prefix, ordered bigrams, rare brand tokens, transliterated core & tokens, address + PIN keys).
   - Conservative address number/locality fallback for records missing postal PIN codes.
   - Adaptive candidate retrieval with a base `top_k = 30` and a strict maximum ceiling of `40 candidates / S1`.
   - Candidate set remains completely frozen before inference.

4. **Candidate Ranking**:
   - Heuristic dual-similarity candidate scoring based on token sort ratios and address alignment.

5. **30-Feature Pair Representation**:
   - 30-feature dense vector capturing name fuzzy ratios, core exact match, token Jaccard & intersection, length ratios, address token & sort ratios, shared street numbers, missing candidate address indicator, Source S2/S3 indicators, Jaro-Winkler similarity, token containment, first token match, acronym match, cross-script transliteration ratio, tri-state PIN code status, street number match status, locality Jaccard, address length ratio, heuristic candidate score, and normalized candidate rank.

6. **LightGBM Model**:
   - `n_estimators = 220`
   - `learning_rate = 0.08`
   - `num_leaves = 31`
   - `max_depth = 6`
   - `subsample = 0.85`
   - `colsample_bytree = 0.85`
   - `objective = binary_logloss`

7. **Decision Policy**:
   - Calibrated decision threshold: $\tau = 0.80$.
   - Maximizes Macro $F_{0.5}$ by heavily suppressing false positive links while capturing multi-target branches.

8. **Output Generation**:
   - Final matching predictions: `output/matching_results.tsv`.
   - Candidate set immediately prior to ML inference: `output/candidate_pairs.tsv`.
   - Guarantees $\text{set}(final\_matches) \subseteq \text{set}(candidate\_pairs)$.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Pipeline & Inference
```bash
python execute_model.py
```

### 3. Validate Submission
```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```
