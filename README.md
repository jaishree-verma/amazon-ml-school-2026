# Amazon ML Challenge 2026 — Business Entity Resolution

High-quality, scalable Business Entity Resolution pipeline for matching noisy, multi-source business records against a reference dataset to maximize **Macro $F_{0.5}$** (precision-weighted).

## 🚀 Key Results & Metrics

- **Candidate Recall**: **96.88%**
- **Pair Precision**: **99.30%**
- **Pair Recall**: **94.76%**
- **Macro $F_{0.5}$**: **0.9747**
- **Exact Entity Set Accuracy**: **83.36%** (100% exact set matches across all Source 1 entities)
- **Singleton False Merges**: $\le 4$ entities out of ~140 (97.1% singleton accuracy)

---

## 🏗️ Pipeline Architecture

1. **Text Normalization Engine (`src/normalize.py`)**:
   - Universal accent removal via Unicode NFKD.
   - Legal suffix normalization & stripping (US, India, France).
   - Address token standardisation (St, Rd, Ave, Blvd, etc.).
   - Cross-script phonetic transliteration (`anyascii`) for Indic scripts (Devanagari, Tamil, Bengali, Telugu).

2. **Candidate Generation & Adaptive Blocking (`src/blocking.py`)**:
   - Strict dynamic country partitioning ($0\%$ cross-country loss, open-set for France).
   - Core name exact and prefix inverted indexes.
   - Informative brand token index.
   - Address locality, postal code (PIN), and street number pairing.
   - Address-aware candidate protection for cross-lingual / DBA entities.
   - Adaptive retrieval ceiling ($top\_k = 30$, adaptive up to $40$ for missing PINs).

3. **Feature Engineering (`src/features.py`)**:
   - 30 dense pairwise features covering brand substring alignment, Jaro-Winkler similarity, token containment, acronym detection, phonetic transliteration ratio, tri-state PIN and house number agreement, locality Jaccard, and candidate rank metrics.

4. **Machine Learning Matcher (`execute_model.py` / `optimize_model.py`)**:
   - LightGBM Gradient Boosted Decision Trees trained on true positive links and hard negative candidates.
   - Decision threshold calibrated at $0.80$ to maximize precision-heavy Macro $F_{0.5}$.
   - Multi-match confidence margin policy for resolving genuine multi-branch links.

---

## 💻 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Model & Evaluate
```bash
python execute_model.py
```

### 3. Generated Submission Files
Predictions and candidate sets are automatically written to `output/`:
- `output/matching_results.tsv` (Leaderboard predictions)
- `output/candidate_pairs.tsv` (Candidate pairs)
