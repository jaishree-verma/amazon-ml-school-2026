import os
import sys
import csv
import json
import time
import random
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from collections import defaultdict, Counter
import rapidfuzz

from src.normalize import normalize_business_name, normalize_address, get_transliterated_name, has_non_latin
from src.features import compute_pairwise_features as compute_base_features, FEATURE_NAMES as BASE_FEATURE_NAMES
from src.metrics import compute_macro_f05
from src.blocking import AdaptiveBlocking, extract_pin_codes, extract_locality_tokens

# Reproducibility
np.random.seed(42)
random.seed(42)

base_dir = r"c:\jaishree_projects\amazon_ml_challenge\student_resource\dataset\train"

print("=================================================================", flush=True)
print("ENTITY RESOLUTION: MODEL OPTIMIZATION & IN-DEPTH ERROR ANALYSIS", flush=True)
print("=================================================================", flush=True)

# 1. Load Ground Truth & Entity Sets (Fixed seed=42 split)
s1_gt = {}
with open(os.path.join(base_dir, "train_ground_truth.tsv"), 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        s1_id = row[0].strip()
        matches = set(row[1].strip().split(',')) if len(row) > 1 and row[1].strip() else set()
        s1_gt[s1_id] = matches

all_s1_keys = list(s1_gt.keys())
random.shuffle(all_s1_keys)

train_s1_ids = set(all_s1_keys[:5000])
val_s1_ids = set(all_s1_keys[5000:7500])

val_true_matches_total = sum(len(s1_gt[s]) for s in val_s1_ids)
print(f"Validation S1: {len(val_s1_ids):,} | True Links: {val_true_matches_total:,}", flush=True)

# Load S1 records
s1_records = {}
with open(os.path.join(base_dir, "train_source1.tsv"), 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        eid = row[0]
        if eid in train_s1_ids or eid in val_s1_ids:
            clean_name, core_name, tokens = normalize_business_name(row[1])
            clean_addr, addr_tokens, numbers = normalize_address(row[2])
            t_clean, t_core, t_tokens = get_transliterated_name(row[1])
            s1_records[eid] = {
                "id": eid,
                "name": row[1],
                "clean_name": clean_name,
                "core_name": core_name,
                "tokens": set(tokens),
                "trans_clean": t_clean,
                "trans_core": t_core,
                "trans_tokens": set(t_tokens),
                "clean_addr": clean_addr,
                "addr_tokens": addr_tokens,
                "numbers": numbers,
                "country": row[3]
            }

train_targets = set()
for sid in train_s1_ids:
    train_targets.update(s1_gt[sid])

val_targets = set()
for sid in val_s1_ids:
    val_targets.update(s1_gt[sid])

all_needed_targets = train_targets | val_targets

# Load candidates pool
candidates_pool = {}
def load_candidates(file_path, prefix, max_distractors=30000):
    distractors = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            eid = row[0]
            is_target = (eid in all_needed_targets)
            if is_target or (distractors < max_distractors):
                clean_name, core_name, tokens = normalize_business_name(row[1])
                clean_addr, addr_tokens, numbers = normalize_address(row[2])
                t_clean, t_core, t_tokens = get_transliterated_name(row[1])
                candidates_pool[eid] = {
                    "id": eid,
                    "name": row[1],
                    "clean_name": clean_name,
                    "core_name": core_name,
                    "tokens": set(tokens),
                    "trans_clean": t_clean,
                    "trans_core": t_core,
                    "trans_tokens": set(t_tokens),
                    "clean_addr": clean_addr,
                    "addr_tokens": addr_tokens,
                    "numbers": numbers,
                    "country": row[3],
                    "source": prefix
                }
                if not is_target:
                    distractors += 1

load_candidates(os.path.join(base_dir, "train_source2.tsv"), "S2", 30000)
load_candidates(os.path.join(base_dir, "train_source3.tsv"), "S3", 30000)
print(f"Candidate records pool loaded: {len(candidates_pool):,}", flush=True)

# 2. Frozen Candidate Generator (Variant C: AdaptiveBlocking)
print("\n[Step 1/8] Generating Frozen Candidate Sets using Variant C...", flush=True)
adaptive_blocker = AdaptiveBlocking()
adaptive_blocker.build_indexes(candidates_pool)

val_ground_truth = {sid: s1_gt[sid] for sid in val_s1_ids}
val_candidates = {}
for sid in val_s1_ids:
    s1_rec = s1_records[sid]
    val_candidates[sid] = adaptive_blocker.get_candidates(s1_rec, candidates_pool)

total_val_candidates = sum(len(c) for c in val_candidates.values())
print(f"Frozen Candidate Pairs on Validation: {total_val_candidates:,} (Avg: {total_val_candidates/len(val_s1_ids):.2f})", flush=True)

# Also generate training pairs
train_candidate_pairs = []
for sid in train_s1_ids:
    s1_rec = s1_records[sid]
    true_set = s1_gt[sid]
    cands = adaptive_blocker.get_candidates(s1_rec, candidates_pool)
    all_cands = cands | (true_set & set(candidates_pool.keys()))
    for cid in all_cands:
        label = 1 if cid in true_set else 0
        train_candidate_pairs.append((sid, cid, label))

print(f"Training Pairs: {len(train_candidate_pairs):,} (Positives: {sum(y for _, _, y in train_candidate_pairs):,})", flush=True)

# 3. Enhanced Feature Engineering
print("\n[Step 2/8] Defining Business Entity Features...", flush=True)

NEW_FEATURE_NAMES = BASE_FEATURE_NAMES + [
    "jaro_winkler_similarity",
    "token_containment",
    "first_token_match",
    "acronym_match",
    "translit_name_ratio",
    "cross_script_flag",
    "pin_match_status",
    "house_num_status",
    "locality_jaccard",
    "addr_len_ratio",
    "cand_heuristic_score",
    "cand_rank_normalized"
]

def compute_enhanced_features(s1_rec, cand_rec, rank=0, total_cands=1):
    base_feats = compute_base_features(s1_rec, cand_rec)
    
    s1_name_clean = s1_rec["clean_name"]
    cand_name_clean = cand_rec["clean_name"]
    s1_tokens = s1_rec["tokens"]
    cand_tokens = cand_rec["tokens"]
    
    # 1. Jaro-Winkler
    jw_sim = rapidfuzz.distance.JaroWinkler.similarity(s1_name_clean, cand_name_clean)
    
    # 2. Token Containment
    min_t = min(len(s1_tokens), len(cand_tokens))
    containment = (len(s1_tokens & cand_tokens) / min_t) if min_t > 0 else 0.0
    
    # 3. First Token Match
    t1_s1 = s1_name_clean.split()[0] if s1_name_clean else ""
    t1_cand = cand_name_clean.split()[0] if cand_name_clean else ""
    first_tok_match = 1.0 if (t1_s1 and t1_cand and t1_s1 == t1_cand) else 0.0
    
    # 4. Acronym Match
    s1_words = [w for w in s1_name_clean.split() if w]
    cand_words = [w for w in cand_name_clean.split() if w]
    s1_acro = "".join(w[0] for w in s1_words[:4])
    cand_acro = "".join(w[0] for w in cand_words[:4])
    acro_match = 1.0 if (s1_acro and cand_acro and (s1_acro == cand_name_clean or cand_acro == s1_name_clean or s1_acro == cand_acro)) else 0.0
    
    # 5. Transliteration Name Ratio & Cross-Script Flag
    translit_ratio = 0.0
    cross_script = 0.0
    if cand_rec.get("trans_clean"):
        cross_script = 1.0
        translit_ratio = max(translit_ratio, rapidfuzz.fuzz.token_sort_ratio(s1_name_clean, cand_rec["trans_clean"]) / 100.0)
    if s1_rec.get("trans_clean"):
        cross_script = 1.0
        translit_ratio = max(translit_ratio, rapidfuzz.fuzz.token_sort_ratio(s1_rec["trans_clean"], cand_name_clean) / 100.0)
        
    # 6. PIN Match Status: +1.0 if match, -1.0 if both have PINs but mismatch, 0.0 if missing
    p1 = extract_pin_codes(s1_rec["addr_tokens"], s1_rec["numbers"])
    p2 = extract_pin_codes(cand_rec["addr_tokens"], cand_rec["numbers"])
    if p1 and p2:
        pin_status = 1.0 if set(p1) & set(p2) else -1.0
    else:
        pin_status = 0.0
        
    # 7. House Number Status: +1.0 if match, -1.0 if both have number but mismatch, 0.0 if missing
    n1 = s1_rec["numbers"]
    n2 = cand_rec["numbers"]
    if n1 and n2:
        h1 = n1[0]
        h2 = n2[0]
        house_status = 1.0 if h1 == h2 else -1.0
    else:
        house_status = 0.0
        
    # 8. Locality Jaccard
    loc1 = set(extract_locality_tokens(s1_rec["addr_tokens"]))
    loc2 = set(extract_locality_tokens(cand_rec["addr_tokens"]))
    loc_inter = len(loc1 & loc2)
    loc_union = len(loc1 | loc2)
    loc_jaccard = (loc_inter / loc_union) if loc_union > 0 else 0.0
    
    # 9. Address Length Ratio
    l_a1 = len(s1_rec["clean_addr"])
    l_a2 = len(cand_rec["clean_addr"])
    addr_len_ratio = (min(l_a1, l_a2) / max(l_a1, l_a2)) if max(l_a1, l_a2) > 0 else 1.0
    
    # 10. Heuristic score & rank normalized
    name_score = base_feats[0] # name_token_sort_ratio
    addr_score = base_feats[9] # addr_sort_ratio
    cand_heur = max(name_score, addr_score)
    cand_rank_norm = (rank / total_cands) if total_cands > 1 else 0.0
    
    new_feats = [
        jw_sim,
        containment,
        first_tok_match,
        acro_match,
        translit_ratio,
        cross_script,
        pin_status,
        house_status,
        loc_jaccard,
        addr_len_ratio,
        cand_heur,
        cand_rank_norm
    ]
    return base_feats + new_feats

print(f"Total Features: {len(NEW_FEATURE_NAMES)} (Baseline: {len(BASE_FEATURE_NAMES)}, Added: {len(NEW_FEATURE_NAMES)-len(BASE_FEATURE_NAMES)})", flush=True)

# 4. Compute Feature Matrices
print("\n[Step 3/8] Computing Feature Matrices...", flush=True)
t0 = time.time()

X_train_base = []
X_train_enh = []
y_train = []

for sid, cid, label in train_candidate_pairs:
    s1_rec = s1_records[sid]
    cand_rec = candidates_pool[cid]
    b_feat = compute_base_features(s1_rec, cand_rec)
    e_feat = compute_enhanced_features(s1_rec, cand_rec)
    X_train_base.append(b_feat)
    X_train_enh.append(e_feat)
    y_train.append(label)

X_train_base = np.array(X_train_base, dtype=np.float32)
X_train_enh = np.array(X_train_enh, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)

X_val_base = []
X_val_enh = []
val_pair_keys = []

for sid in val_s1_ids:
    s1_rec = s1_records[sid]
    cands = list(val_candidates[sid])
    for rank, cid in enumerate(cands):
        cand_rec = candidates_pool[cid]
        b_feat = compute_base_features(s1_rec, cand_rec)
        e_feat = compute_enhanced_features(s1_rec, cand_rec, rank=rank, total_cands=len(cands))
        X_val_base.append(b_feat)
        X_val_enh.append(e_feat)
        val_pair_keys.append((sid, cid))

X_val_base = np.array(X_val_base, dtype=np.float32)
X_val_enh = np.array(X_val_enh, dtype=np.float32)
print(f"Features computed in {time.time() - t0:.2f}s", flush=True)

# 5. Train and Compare Model Alternatives
print("\n[Step 4/8] Training and Comparing Model Alternatives...", flush=True)

models = {}

# Model A: Current LightGBM Baseline (18 features)
print("Training Model A: LightGBM Baseline (18 feats)...", flush=True)
model_a = lgb.LGBMClassifier(n_estimators=180, learning_rate=0.09, num_leaves=31, max_depth=6, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1, verbose=-1)
model_a.fit(X_train_base, y_train)
models["Model A (LightGBM Baseline)"] = (model_a, X_val_base)

# Model B: Improved LightGBM (+ Engineered Features)
print("Training Model B: LightGBM + Enhanced Features (30 feats)...", flush=True)
model_b = lgb.LGBMClassifier(n_estimators=220, learning_rate=0.08, num_leaves=31, max_depth=6, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1, verbose=-1)
model_b.fit(X_train_enh, y_train)
models["Model B (LightGBM + Enhanced Feats)"] = (model_b, X_val_enh)

# Model C: LightGBM with Precision Weighting (scale_pos_weight=0.85 to penalize false positives)
print("Training Model C: LightGBM (Weighted for Precision)...", flush=True)
model_c = lgb.LGBMClassifier(n_estimators=220, learning_rate=0.08, num_leaves=31, max_depth=6, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=0.85, random_state=42, n_jobs=-1, verbose=-1)
model_c.fit(X_train_enh, y_train)
models["Model C (LightGBM Precision-Weighted)"] = (model_c, X_val_enh)

# Model D: XGBoost Classifier (30 feats)
print("Training Model D: XGBoost Classifier...", flush=True)
model_d = xgb.XGBClassifier(n_estimators=220, learning_rate=0.08, max_depth=6, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1, eval_metric='logloss')
model_d.fit(X_train_enh, y_train)
models["Model D (XGBoost)"] = (model_d, X_val_enh)

# Model E: HistGradientBoosting (30 feats)
print("Training Model E: HistGradientBoosting...", flush=True)
model_e = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.09, max_leaf_nodes=31, max_depth=6, random_state=42)
model_e.fit(X_train_enh, y_train)
models["Model E (HistGradientBoosting)"] = (model_e, X_val_enh)

# Helper for scoring
def evaluate_model_predictions(probs, threshold=0.85):
    val_s1_scores = defaultdict(list)
    for (sid, cid), p in zip(val_pair_keys, probs):
        val_s1_scores[sid].append((cid, float(p)))
        
    preds = {sid: {cid for cid, p in val_s1_scores[sid] if p >= threshold} for sid in val_s1_ids}
    tp = sum(len(val_ground_truth[sid] & preds[sid]) for sid in val_s1_ids)
    fp = sum(len(preds[sid] - val_ground_truth[sid]) for sid in val_s1_ids)
    fn = sum(len(val_ground_truth[sid] - preds[sid]) for sid in val_s1_ids)
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    macro_res = compute_macro_f05(val_ground_truth, preds)
    exact_acc = (sum(1 for sid in val_s1_ids if val_ground_truth[sid] == preds[sid]) / len(val_s1_ids)) * 100
    
    return {
        "threshold": threshold,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "macro_f05": float(macro_res["macro_f05"]),
        "exact_entity_accuracy": float(exact_acc),
        "singleton_fps": int(macro_res["singleton_false_positives"]),
        "fp_count": fp,
        "fn_count": fn,
        "val_s1_scores": val_s1_scores
    }

# Compare models at th=0.85
model_comparison = {}
for m_name, (m_obj, X_eval) in models.items():
    t_inf = time.time()
    probs = m_obj.predict_proba(X_eval)[:, 1]
    eval_res = evaluate_model_predictions(probs, threshold=0.85)
    eval_res["inference_time"] = float(time.time() - t_inf)
    del eval_res["val_s1_scores"]
    model_comparison[m_name] = eval_res

# 6. Detailed Error Dataset & Hard Negative / False Negative Analysis (on Model B)
print("\n[Step 5/8] Analyzing Hard Negatives & False Negatives...", flush=True)
best_model_probs = model_b.predict_proba(X_val_enh)[:, 1]

diagnostic_records = []
cat_b_records = []
cat_c_records = []

for (sid, cid), prob in zip(val_pair_keys, best_model_probs):
    s1_rec = s1_records[sid]
    cand_rec = candidates_pool[cid]
    is_true = (cid in val_ground_truth[sid])
    pred_label = 1 if prob >= 0.85 else 0
    
    rec = {
        "s1_id": sid,
        "s1_name": s1_rec["name"],
        "s1_addr": s1_rec["clean_addr"],
        "cand_id": cid,
        "cand_name": cand_rec["name"],
        "cand_addr": cand_rec["clean_addr"],
        "is_true_match": is_true,
        "predicted_prob": float(prob),
        "predicted_label": pred_label
    }
    
    # Category B: True match, but rejected (prob < 0.85)
    if is_true and pred_label == 0:
        cat_b_records.append(rec)
    # Category C: Negative candidate, but accepted (prob >= 0.85)
    if not is_true and pred_label == 1:
        cat_c_records.append(rec)

# False negative categorization (the Cat B misses)
fn_breakdown = Counter()
for r in cat_b_records:
    p = r["predicted_prob"]
    name_sim = rapidfuzz.fuzz.token_sort_ratio(r["s1_name"], r["cand_name"])
    addr_sim = rapidfuzz.fuzz.token_sort_ratio(r["s1_addr"], r["cand_addr"]) if (r["s1_addr"] and r["cand_addr"]) else 0
    
    if not r["cand_addr"]:
        fn_breakdown["missing_cand_address"] += 1
    elif 0.70 <= p < 0.85:
        fn_breakdown["marginal_score_near_threshold (0.70-0.85)"] += 1
    elif addr_sim >= 75 and name_sim < 50:
        fn_breakdown["weak_name_strong_address"] += 1
    elif name_sim >= 75 and addr_sim < 40:
        fn_breakdown["strong_name_weak_address"] += 1
    elif any(0x0900 <= ord(c) <= 0x0D7F for c in r["cand_name"]):
        fn_breakdown["native_script_transliteration_gap"] += 1
    else:
        fn_breakdown["moderate_name_and_addr_noise"] += 1

# Hard negative categorization (the Cat C false positives)
cat_c_records.sort(key=lambda x: x["predicted_prob"], reverse=True)
fp_breakdown = Counter()
for r in cat_c_records:
    name_sim = rapidfuzz.fuzz.token_sort_ratio(r["s1_name"], r["cand_name"])
    addr_sim = rapidfuzz.fuzz.token_sort_ratio(r["s1_addr"], r["cand_addr"]) if (r["s1_addr"] and r["cand_addr"]) else 0
    if name_sim >= 90 and addr_sim < 40:
        fp_breakdown["same_or_near_identical_name_different_locality"] += 1
    elif name_sim < 60 and addr_sim >= 85:
        fp_breakdown["different_business_same_address_or_building"] += 1
    elif "hotel" in r["s1_name"].lower() or "trading" in r["s1_name"].lower() or "enterprises" in r["s1_name"].lower():
        fp_breakdown["generic_business_keyword_collision"] += 1
    else:
        fp_breakdown["high_name_and_address_coincidence"] += 1

# 7. Feature Importance & Ablation
print("\n[Step 6/8] Feature Importance & Ablation...", flush=True)
importances = model_b.feature_importances_
feat_ranks = sorted(zip(NEW_FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)

# Ablation experiment: Baseline vs Baseline + Name vs Baseline + Address vs All
ablation_results = {}

# Ablation 1: Baseline (18 feats)
ablation_results["Baseline Features (18)"] = model_comparison["Model A (LightGBM Baseline)"]["macro_f05"]

# Ablation 2: Baseline + Jaro-Winkler + Token Containment + Acronym
enh_name_idx = list(range(18)) + [18, 19, 20, 21, 22, 23]
m_name_ablation = lgb.LGBMClassifier(n_estimators=180, learning_rate=0.09, num_leaves=31, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
m_name_ablation.fit(X_train_enh[:, enh_name_idx], y_train)
p_name = m_name_ablation.predict_proba(X_val_enh[:, enh_name_idx])[:, 1]
ablation_results["Baseline + New Name & Translit Feats (24)"] = evaluate_model_predictions(p_name, 0.85)["macro_f05"]

# Ablation 3: Baseline + Address PIN & Locality
enh_addr_idx = list(range(18)) + [24, 25, 26, 27]
m_addr_ablation = lgb.LGBMClassifier(n_estimators=180, learning_rate=0.09, num_leaves=31, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
m_addr_ablation.fit(X_train_enh[:, enh_addr_idx], y_train)
p_addr = m_addr_ablation.predict_proba(X_val_enh[:, enh_addr_idx])[:, 1]
ablation_results["Baseline + New Address & PIN Feats (22)"] = evaluate_model_predictions(p_addr, 0.85)["macro_f05"]

# Ablation 4: All 30 features
ablation_results["All Features (30)"] = model_comparison["Model B (LightGBM + Enhanced Feats)"]["macro_f05"]

# 8. Threshold Sweep for Model B (0.30 -> 0.95)
print("\n[Step 7/8] Threshold Optimization Sweep (0.30 to 0.95)...", flush=True)
threshold_sweep = []
for th in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]:
    res = evaluate_model_predictions(best_model_probs, threshold=th)
    del res["val_s1_scores"]
    threshold_sweep.append(res)

# 9. Multi-Match Entity-Level Decision Logic
print("\n[Step 8/8] Testing Entity-Level Decision Logic...", flush=True)
val_s1_scores = defaultdict(list)
for (sid, cid), p in zip(val_pair_keys, best_model_probs):
    val_s1_scores[sid].append((cid, float(p)))

# Rule A: Pure Threshold (th = 0.85)
rule_a_preds = {sid: {cid for cid, p in val_s1_scores[sid] if p >= 0.85} for sid in val_s1_ids}
rule_a_res = compute_macro_f05(val_ground_truth, rule_a_preds)

# Rule B: Threshold + Confidence Margin
# If top candidate is very high (>= 0.92), admit secondary candidates with prob >= 0.72 if gap <= 0.20
rule_b_preds = {}
for sid in val_s1_ids:
    scored = sorted(val_s1_scores[sid], key=lambda x: x[1], reverse=True)
    accepted = set()
    if scored:
        top_cid, top_p = scored[0]
        if top_p >= 0.85:
            accepted.add(top_cid)
            for cid, p in scored[1:]:
                if p >= 0.85 or (top_p >= 0.92 and p >= 0.72 and (top_p - p) <= 0.20):
                    accepted.add(cid)
    rule_b_preds[sid] = accepted
rule_b_res = compute_macro_f05(val_ground_truth, rule_b_preds)

# Rule C: Threshold + Strong-Evidence Override
# If candidate has exact core match AND PIN match, admit at prob >= 0.60 even if below 0.85
rule_c_preds = {}
for sid in val_s1_ids:
    s1_rec = s1_records[sid]
    accepted = set()
    for cid, p in val_s1_scores[sid]:
        if p >= 0.85:
            accepted.add(cid)
        elif p >= 0.60:
            cand_rec = candidates_pool[cid]
            exact_core = (s1_rec["core_name"] and s1_rec["core_name"] == cand_rec["core_name"])
            pin1 = extract_pin_codes(s1_rec["addr_tokens"], s1_rec["numbers"])
            pin2 = extract_pin_codes(cand_rec["addr_tokens"], cand_rec["numbers"])
            pin_match = (set(pin1) & set(pin2)) if (pin1 and pin2) else False
            if exact_core or pin_match:
                accepted.add(cid)
    rule_c_preds[sid] = accepted
rule_c_res = compute_macro_f05(val_ground_truth, rule_c_preds)

# Decision logic comparison
decision_rules = {
    "Rule A (Pure Threshold @ 0.85)": {
        "macro_f05": rule_a_res["macro_f05"],
        "exact_accuracy": sum(1 for s in val_s1_ids if val_ground_truth[s] == rule_a_preds[s]) / len(val_s1_ids) * 100,
        "singleton_fps": rule_a_res["singleton_false_positives"]
    },
    "Rule B (Threshold + Confidence Margin)": {
        "macro_f05": rule_b_res["macro_f05"],
        "exact_accuracy": sum(1 for s in val_s1_ids if val_ground_truth[s] == rule_b_preds[s]) / len(val_s1_ids) * 100,
        "singleton_fps": rule_b_res["singleton_false_positives"]
    },
    "Rule C (Threshold + Strong-Evidence Override)": {
        "macro_f05": rule_c_res["macro_f05"],
        "exact_accuracy": sum(1 for s in val_s1_ids if val_ground_truth[s] == rule_c_preds[s]) / len(val_s1_ids) * 100,
        "singleton_fps": rule_c_res["singleton_false_positives"]
    }
}

# Compile complete summary
full_summary = {
    "model_comparison": model_comparison,
    "top_feature_importance": feat_ranks[:15],
    "ablation_results": ablation_results,
    "false_negative_breakdown": dict(fn_breakdown),
    "false_positive_breakdown": dict(fp_breakdown),
    "top_hard_negatives_sample": cat_c_records[:10],
    "threshold_sweep": threshold_sweep,
    "decision_rules": decision_rules
}

def default_serializer(obj):
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, set):
        return list(obj)
    return str(obj)

with open(r"c:\jaishree_projects\amazon_ml_challenge\model_optimization_summary.json", 'w', encoding='utf-8') as f:
    json.dump(full_summary, f, indent=2, default=default_serializer)

print("\n=================================================================", flush=True)
print("MODEL OPTIMIZATION & ERROR ANALYSIS COMPLETE!", flush=True)
print("Saved complete diagnostic data to model_optimization_summary.json", flush=True)
print("=================================================================", flush=True)
