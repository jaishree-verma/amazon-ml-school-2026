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

from src.normalize import normalize_business_name, normalize_address, get_transliterated_name, has_non_latin
from src.features import compute_pairwise_features, FEATURE_NAMES
from src.metrics import compute_macro_f05
from src.blocking import EnhancedBlocking, TranslitBlocking, AdaptiveBlocking, extract_pin_codes

# Reproducibility
np.random.seed(42)
random.seed(42)

base_dir = r"c:\jaishree_projects\amazon_ml_challenge\student_resource\dataset\train"

print("=================================================================", flush=True)
print("A/B/C EVALUATION: ENHANCED vs TRANSLIT vs ADAPTIVE BLOCKING", flush=True)
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

val_true_matches_total = sum(len(s1_gt[s]) for s in val_s1_ids)
print(f"Validation Entities: {len(val_s1_ids):,} | True Positive Links: {val_true_matches_total:,}", flush=True)

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

# Load candidate pool
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
print(f"Candidates pool loaded: {len(candidates_pool):,} records.", flush=True)

# Train the reference LightGBM model once
print("\nTraining LightGBM model on training pairs...", flush=True)
base_blocker = EnhancedBlocking()
base_blocker.build_indexes(candidates_pool)

X_train = []
y_train = []
for sid in train_s1_ids:
    s1_rec = s1_records[sid]
    true_set = s1_gt[sid]
    cand_set = base_blocker.get_candidates(s1_rec, candidates_pool, top_k=30)
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

# Evaluation Function across thresholds
def run_evaluation(blocking_obj, name):
    print(f"\n--- Running {name} ---", flush=True)
    t0 = time.time()
    
    val_ground_truth = {sid: s1_gt[sid] for sid in val_s1_ids}
    val_candidates = {}
    val_pair_feats = []
    val_pair_keys = []
    
    block_start = time.time()
    for sid in val_s1_ids:
        s1_rec = s1_records[sid]
        cand_set = blocking_obj.get_candidates(s1_rec, candidates_pool)
        val_candidates[sid] = cand_set
        for cid in cand_set:
            cand_rec = candidates_pool[cid]
            val_pair_feats.append(compute_pairwise_features(s1_rec, cand_rec))
            val_pair_keys.append((sid, cid))
    block_feat_time = time.time() - block_start
    
    # Model inference
    inf_start = time.time()
    X_val = np.array(val_pair_feats, dtype=np.float32)
    val_probs = model.predict_proba(X_val)[:, 1]
    inf_time = time.time() - inf_start
    total_time = time.time() - t0
    
    val_s1_scores = defaultdict(list)
    for (sid, cid), prob in zip(val_pair_keys, val_probs):
        val_s1_scores[sid].append((cid, float(prob)))
        
    candidate_counts = [len(c) for c in val_candidates.values()]
    total_cands = sum(candidate_counts)
    avg_cands = float(np.mean(candidate_counts))
    max_cands = int(np.max(candidate_counts))
    
    # Category A misses set
    cat_a_pairs = set()
    for sid in val_s1_ids:
        true_set = val_ground_truth[sid]
        cands = val_candidates[sid]
        for tid in true_set:
            if tid not in cands:
                cat_a_pairs.add((sid, tid))
                
    cat_a_count = len(cat_a_pairs)
    cand_recall = ((val_true_matches_total - cat_a_count) / val_true_matches_total) * 100
    
    def score_at_threshold(th):
        preds = {sid: {cid for cid, p in val_s1_scores[sid] if p >= th} for sid in val_s1_ids}
        tp = sum(len(val_ground_truth[sid] & preds[sid]) for sid in val_s1_ids)
        fp = sum(len(preds[sid] - val_ground_truth[sid]) for sid in val_s1_ids)
        fn = sum(len(val_ground_truth[sid] - preds[sid]) for sid in val_s1_ids)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        macro_res = compute_macro_f05(val_ground_truth, preds)
        exact_correct = sum(1 for sid in val_s1_ids if val_ground_truth[sid] == preds[sid])
        exact_acc = (exact_correct / len(val_s1_ids)) * 100
        
        cat_b = 0
        cat_c = 0
        for sid in val_s1_ids:
            true_set = val_ground_truth[sid]
            cands = val_candidates[sid]
            pred_set = preds[sid]
            for tid in true_set:
                if tid in cands and tid not in pred_set:
                    cat_b += 1
            for pid in pred_set:
                if pid not in true_set:
                    cat_c += 1
                    
        return {
            "threshold": th,
            "cat_b": cat_b,
            "cat_c": cat_c,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "macro_f05": float(macro_res["macro_f05"]),
            "singleton_fps": int(macro_res["singleton_false_positives"]),
            "exact_entity_accuracy": float(exact_acc),
            "correct_entities": int(exact_correct),
            "incorrect_entities": int(len(val_s1_ids) - exact_correct)
        }
        
    res_085 = score_at_threshold(0.85)
    res_055 = score_at_threshold(0.55)
    
    return {
        "name": name,
        "runtime": float(total_time),
        "total_candidate_pairs": int(total_cands),
        "avg_candidates": avg_cands,
        "max_candidates": max_cands,
        "candidate_recall": float(cand_recall),
        "cat_a_count": int(cat_a_count),
        "cat_a_pairs": cat_a_pairs,
        "metrics_085": res_085,
        "metrics_055": res_055
    }

# Build and run Variant A
print("\nBuilding Variant A (Enhanced Baseline)...", flush=True)
blocker_a = EnhancedBlocking()
blocker_a.build_indexes(candidates_pool)
res_a = run_evaluation(blocker_a, "Variant A (Current Enhanced Baseline)")

# Build and run Variant B
print("\nBuilding Variant B (Enhanced + Cross-Script Transliteration)...", flush=True)
blocker_b = TranslitBlocking()
blocker_b.build_indexes(candidates_pool)
res_b = run_evaluation(blocker_b, "Variant B (Enhanced + Transliteration)")

# Build and run Variant C
print("\nBuilding Variant C (Enhanced + Transliteration + Address Fallback / Adaptive)...", flush=True)
blocker_c = AdaptiveBlocking()
blocker_c.build_indexes(candidates_pool)
res_c = run_evaluation(blocker_c, "Variant C (Enhanced + Translit + Address Fallback)")

# 4. MEASURE RECOVERIES OF THE 4 ERROR SEGMENTS (Against Variant A's misses)
misses_a = res_a["cat_a_pairs"]

# Categorize the original 363 misses in Variant A:
cross_script_original = set()
missing_pin_original = set()
missing_addr_original = set()
rank_gt_30_original = set()

for sid, tid in misses_a:
    s1_rec = s1_records[sid]
    t_rec = candidates_pool.get(tid)
    if not t_rec: continue
    
    if not t_rec["clean_addr"] or len(t_rec["clean_addr"]) < 5:
        missing_addr_original.add((sid, tid))
    elif any(0x0900 <= ord(c) <= 0x0D7F for c in t_rec["name"]):
        cross_script_original.add((sid, tid))
    elif not extract_pin_codes(s1_rec["addr_tokens"], s1_rec["numbers"]) and not extract_pin_codes(t_rec["addr_tokens"], t_rec["numbers"]):
        missing_pin_original.add((sid, tid))
    else:
        rank_gt_30_original.add((sid, tid))

# Calculate recovery for B
recov_b_total = len(misses_a - res_b["cat_a_pairs"])
recov_b_cross_script = len(cross_script_original - res_b["cat_a_pairs"])
recov_b_missing_pin = len(missing_pin_original - res_b["cat_a_pairs"])
recov_b_missing_addr = len(missing_addr_original - res_b["cat_a_pairs"])
recov_b_rank = len(rank_gt_30_original - res_b["cat_a_pairs"])

# Calculate recovery for C
recov_c_total = len(misses_a - res_c["cat_a_pairs"])
recov_c_cross_script = len(cross_script_original - res_c["cat_a_pairs"])
recov_c_missing_pin = len(missing_pin_original - res_c["cat_a_pairs"])
recov_c_missing_addr = len(missing_addr_original - res_c["cat_a_pairs"])
recov_c_rank = len(rank_gt_30_original - res_c["cat_a_pairs"])

# Summary Export
summary = {
    "variant_a": {k: v for k, v in res_a.items() if k != "cat_a_pairs"},
    "variant_b": {k: v for k, v in res_b.items() if k != "cat_a_pairs"},
    "variant_c": {k: v for k, v in res_c.items() if k != "cat_a_pairs"},
    "segment_recoveries": {
        "variant_b": {
            "total_recovered": recov_b_total,
            "cross_script_recovered": recov_b_cross_script,
            "cross_script_total": len(cross_script_original),
            "missing_pin_recovered": recov_b_missing_pin,
            "missing_pin_total": len(missing_pin_original),
            "missing_addr_recovered": recov_b_missing_addr,
            "missing_addr_total": len(missing_addr_original),
            "rank_gt_30_recovered": recov_b_rank,
            "rank_gt_30_total": len(rank_gt_30_original)
        },
        "variant_c": {
            "total_recovered": recov_c_total,
            "cross_script_recovered": recov_c_cross_script,
            "cross_script_total": len(cross_script_original),
            "missing_pin_recovered": recov_c_missing_pin,
            "missing_pin_total": len(missing_pin_original),
            "missing_addr_recovered": recov_c_missing_addr,
            "missing_addr_total": len(missing_addr_original),
            "rank_gt_30_recovered": recov_c_rank,
            "rank_gt_30_total": len(rank_gt_30_original)
        }
    }
}

def default_serializer(obj):
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, set):
        return list(obj)
    return str(obj)

with open(r"c:\jaishree_projects\amazon_ml_challenge\abc_blocking_summary.json", 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, default=default_serializer)

print("\n=================================================================", flush=True)
print("A/B/C EVALUATION COMPLETE! Saved to abc_blocking_summary.json", flush=True)
print("=================================================================", flush=True)
