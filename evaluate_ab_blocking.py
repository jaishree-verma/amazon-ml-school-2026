import os
import sys
import csv
import json
import time
import random
import numpy as np
import lightgbm as lgb
from collections import defaultdict, Counter
import rapidfuzz

from src.normalize import normalize_business_name, normalize_address
from src.features import compute_pairwise_features, FEATURE_NAMES
from src.metrics import compute_macro_f05
from src.blocking import BaselineBlocking, EnhancedBlocking, extract_pin_codes

# Exact seed reproducibility
np.random.seed(42)
random.seed(42)

base_dir = r"c:\jaishree_projects\amazon_ml_challenge\student_resource\dataset\train"

print("=================================================================", flush=True)
print("A/B EVALUATION: BASELINE VS ENHANCED BLOCKING", flush=True)
print("=================================================================", flush=True)

# 1. Load Ground Truth
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

print(f"Validation Set: {len(val_s1_ids):,} S1 entities | True Links: {sum(len(s1_gt[s]) for s in val_s1_ids):,}", flush=True)

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
            s1_records[eid] = {
                "id": eid,
                "name": row[1],
                "clean_name": clean_name,
                "core_name": core_name,
                "tokens": set(tokens),
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

# Load candidate pool (identical pool for fair A/B test)
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
                candidates_pool[eid] = {
                    "id": eid,
                    "name": row[1],
                    "clean_name": clean_name,
                    "core_name": core_name,
                    "tokens": set(tokens),
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

# 2. Build Inverted Indexes for Baseline and Enhanced
baseline_blocking = BaselineBlocking()
baseline_blocking.build_indexes(candidates_pool)

enhanced_blocking = EnhancedBlocking()
enhanced_blocking.build_indexes(candidates_pool)

# 3. Train LightGBM model once on baseline training pairs
print("Training standard LightGBM model on training pairs...", flush=True)
X_train = []
y_train = []

for sid in train_s1_ids:
    s1_rec = s1_records[sid]
    true_set = s1_gt[sid]
    cand_set = baseline_blocking.get_candidates(s1_rec, candidates_pool, top_k=20)
    all_cands = cand_set | (true_set & set(candidates_pool.keys()))
    
    for cid in all_cands:
        cand_rec = candidates_pool[cid]
        feats = compute_pairwise_features(s1_rec, cand_rec)
        label = 1 if cid in true_set else 0
        X_train.append(feats)
        y_train.append(label)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)

model = lgb.LGBMClassifier(
    n_estimators=180,
    learning_rate=0.09,
    num_leaves=31,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
model.fit(X_train, y_train)
print("LightGBM Model Ready.", flush=True)

# Evaluation Function
FIXED_THRESHOLD = 0.55

def evaluate_blocking_pipeline(blocking_obj, top_k_val, name="Pipeline"):
    print(f"\nEvaluating {name}...", flush=True)
    t0 = time.time()
    
    val_ground_truth = {sid: s1_gt[sid] for sid in val_s1_ids}
    val_candidates = {}
    val_pair_feats = []
    val_pair_keys = []
    
    blocking_start = time.time()
    for sid in val_s1_ids:
        s1_rec = s1_records[sid]
        cand_set = blocking_obj.get_candidates(s1_rec, candidates_pool, top_k=top_k_val)
        val_candidates[sid] = cand_set
        for cid in cand_set:
            cand_rec = candidates_pool[cid]
            val_pair_feats.append(compute_pairwise_features(s1_rec, cand_rec))
            val_pair_keys.append((sid, cid))
    blocking_and_feat_time = time.time() - blocking_start
    
    # Model inference
    inf_start = time.time()
    X_val = np.array(val_pair_feats, dtype=np.float32)
    val_probs = model.predict_proba(X_val)[:, 1]
    inference_time = time.time() - inf_start
    
    val_s1_scores = defaultdict(list)
    for (sid, cid), prob in zip(val_pair_keys, val_probs):
        val_s1_scores[sid].append((cid, float(prob)))
        
    # Generate predictions at fixed threshold = 0.55
    predictions = {}
    for sid in val_s1_ids:
        predictions[sid] = {cid for cid, prob in val_s1_scores[sid] if prob >= FIXED_THRESHOLD}
        
    total_time = time.time() - t0
    
    # Calculate Candidate Statistics
    candidate_counts = [len(c) for c in val_candidates.values()]
    total_cands = sum(candidate_counts)
    avg_cands = np.mean(candidate_counts)
    med_cands = np.median(candidate_counts)
    max_cands = np.max(candidate_counts)
    
    # Decompose Errors: Cat A, Cat B, Cat C
    total_true_matches = sum(len(m) for m in val_ground_truth.values())
    cat_a_set = set() # (sid, true_id)
    cat_b_set = set()
    cat_c_count = 0
    
    correct_entities = 0
    incorrect_entities = 0
    
    for sid in val_s1_ids:
        true_set = val_ground_truth[sid]
        pred_set = predictions[sid]
        cands = val_candidates[sid]
        
        if true_set == pred_set:
            correct_entities += 1
        else:
            incorrect_entities += 1
            
        for tid in true_set:
            if tid not in cands:
                cat_a_set.add((sid, tid))
            elif tid not in pred_set:
                cat_b_set.add((sid, tid))
                
        for pid in pred_set:
            if pid not in true_set:
                cat_c_count += 1
                
    cat_a = len(cat_a_set)
    cat_b = len(cat_b_set)
    cand_recall = ((total_true_matches - cat_a) / total_true_matches) * 100
    
    # Standard Pair Metrics
    tp = sum(len(val_ground_truth[sid] & predictions[sid]) for sid in val_s1_ids)
    fp = sum(len(predictions[sid] - val_ground_truth[sid]) for sid in val_s1_ids)
    fn = sum(len(val_ground_truth[sid] - predictions[sid]) for sid in val_s1_ids)
    
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
    
    macro_metrics = compute_macro_f05(val_ground_truth, predictions)
    exact_acc = (correct_entities / len(val_s1_ids)) * 100
    
    return {
        "name": name,
        "total_time": total_time,
        "blocking_and_feat_time": blocking_and_feat_time,
        "inference_time": inference_time,
        "total_candidate_pairs": total_cands,
        "avg_candidates": avg_cands,
        "med_candidates": med_cands,
        "max_candidates": max_cands,
        "cand_recall": cand_recall,
        "cat_a": cat_a,
        "cat_b": cat_b,
        "cat_c": cat_c_count,
        "precision": p,
        "recall": r,
        "f1": f1,
        "macro_f05": macro_metrics["macro_f05"],
        "singleton_fps": macro_metrics["singleton_false_positives"],
        "exact_entity_accuracy": exact_acc,
        "correct_entities": correct_entities,
        "incorrect_entities": incorrect_entities,
        "cat_a_pairs": cat_a_set,
        "val_candidates": val_candidates
    }

# Run A and B
res_a = evaluate_blocking_pipeline(baseline_blocking, top_k_val=20, name="Baseline (top_k=20)")
res_b = evaluate_blocking_pipeline(enhanced_blocking, top_k_val=30, name="Enhanced (top_k=30 + PIN/Locality + Protection)")

# 4. CALCULATE COMPARATIVE METRICS
cat_a_recovered = len(res_a["cat_a_pairs"] - res_b["cat_a_pairs"])
new_cands = res_b["total_candidate_pairs"] - res_a["total_candidate_pairs"]
pct_cands_increase = (new_cands / res_a["total_candidate_pairs"]) * 100
pct_time_increase = ((res_b["total_time"] - res_a["total_time"]) / res_a["total_time"]) * 100

# 5. ERROR ATTRIBUTION FOR REMAINING CATEGORY A MISSES IN ENHANCED BLOCKING
remaining_cat_a = res_b["cat_a_pairs"]
attribution_counts = Counter()

for sid, tid in remaining_cat_a:
    s1_rec = s1_records[sid]
    t_rec = candidates_pool.get(tid)
    if not t_rec:
        attribution_counts["unknown_target"] += 1
        continue
        
    s1_name = s1_rec["clean_name"]
    t_name = t_rec["clean_name"]
    s1_addr = s1_rec["clean_addr"]
    t_addr = t_rec["clean_addr"]
    
    # 1. Missing address
    if not t_addr or len(t_addr) < 5:
        attribution_counts["missing_target_address"] += 1
    elif not s1_addr or len(s1_addr) < 5:
        attribution_counts["missing_s1_address"] += 1
    # 2. Cross-script / Transliteration (Indic non-ASCII or phonetically disjoint)
    elif any(0x0900 <= ord(c) <= 0x0D7F for c in t_rec["name"]):
        attribution_counts["cross_script_native_script"] += 1
    # 3. Missing PIN in both
    elif not extract_pin_codes(s1_rec["addr_tokens"], s1_rec["numbers"]) and not extract_pin_codes(t_rec["addr_tokens"], t_rec["numbers"]):
        attribution_counts["missing_postal_pin_code"] += 1
    # 4. Severe name divergence
    elif rapidfuzz.fuzz.token_sort_ratio(s1_name, t_name) < 30:
        attribution_counts["extreme_name_divergence_no_token_overlap"] += 1
    # 5. Locality mismatch
    elif rapidfuzz.fuzz.token_sort_ratio(s1_addr, t_addr) < 40:
        attribution_counts["address_components_disjoint"] += 1
    else:
        attribution_counts["candidate_pruning_rank_gt_30"] += 1

# Export full A/B results
ab_summary = {
    "baseline": {k: v for k, v in res_a.items() if k not in ("cat_a_pairs", "val_candidates")},
    "enhanced": {k: v for k, v in res_b.items() if k not in ("cat_a_pairs", "val_candidates")},
    "comparison": {
        "cat_a_recovered": cat_a_recovered,
        "cat_a_recovery_pct": (cat_a_recovered / res_a["cat_a"]) * 100,
        "new_candidate_pairs": new_cands,
        "pct_candidate_pairs_increase": pct_cands_increase,
        "time_increase_pct": pct_time_increase,
        "macro_f05_diff": res_b["macro_f05"] - res_a["macro_f05"],
        "accuracy_diff": res_b["exact_entity_accuracy"] - res_a["exact_entity_accuracy"]
    },
    "remaining_cat_a_attribution": dict(attribution_counts)
}

def default_serializer(obj):
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, set):
        return list(obj)
    return str(obj)

with open(r"c:\jaishree_projects\amazon_ml_challenge\ab_blocking_summary.json", 'w', encoding='utf-8') as f:
    json.dump(ab_summary, f, indent=2, default=default_serializer)

print("\nA/B Evaluation Completed successfully! Saved to ab_blocking_summary.json", flush=True)
