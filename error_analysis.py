import os
import sys
import csv
import json
import random
import numpy as np
import lightgbm as lgb
from collections import defaultdict, Counter
import rapidfuzz

from src.normalize import normalize_business_name, normalize_address
from src.features import compute_pairwise_features, FEATURE_NAMES
from src.metrics import compute_macro_f05

# Reproducibility
np.random.seed(42)
random.seed(42)

base_dir = r"c:\jaishree_projects\amazon_ml_challenge\student_resource\dataset\train"

print("Starting in-depth error analysis...", flush=True)

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

# Build inverted indexes
country_core_idx = defaultdict(lambda: defaultdict(list))
country_prefix_idx = defaultdict(lambda: defaultdict(list))
country_token_idx = defaultdict(lambda: defaultdict(list))
country_2token_idx = defaultdict(lambda: defaultdict(list))
country_addr_idx = defaultdict(lambda: defaultdict(list))

def get_addr_keys(addr_clean, addr_tokens, numbers):
    keys = []
    long_tokens = [t for t in addr_tokens if len(t) >= 4 and not t.isdigit() and t not in ('floor', 'road', 'street', 'avenue', 'near', 'block')]
    for num in numbers[:2]:
        for t in long_tokens[:3]:
            keys.append(f"{num}_{t}")
    for num in numbers:
        if len(num) in (5, 6):
            keys.append(f"pin_{num}")
    return keys

for eid, rec in candidates_pool.items():
    country = rec["country"]
    core = rec["core_name"].replace(" ", "")
    if core:
        country_core_idx[country][core].append(eid)
        if len(core) >= 5:
            country_prefix_idx[country][core[:5]].append(eid)
            
    tokens = list(rec["tokens"])
    if len(tokens) >= 2:
        country_2token_idx[country][f"{tokens[0]}_{tokens[1]}"].append(eid)
        
    for t in tokens:
        if len(t) >= 4 and t not in ("hotel", "company", "services", "enterprises", "india", "pvt", "ltd"):
            country_token_idx[country][t].append(eid)
            
    addr_keys = get_addr_keys(rec["clean_addr"], rec["addr_tokens"], rec["numbers"])
    for ak in addr_keys:
        country_addr_idx[country][ak].append(eid)

def get_candidates(rec, top_k=20):
    country = rec["country"]
    clean_name = rec["clean_name"]
    clean_core = rec["core_name"].replace(" ", "")
    tokens = list(rec["tokens"])
    clean_addr = rec["clean_addr"]
    
    cand_set = set()
    if clean_core in country_core_idx[country]:
        cand_set.update(country_core_idx[country][clean_core])
        
    if len(tokens) >= 2:
        pair_key = f"{tokens[0]}_{tokens[1]}"
        if pair_key in country_2token_idx[country]:
            for cid in country_2token_idx[country][pair_key][:20]:
                cand_set.add(cid)
                
    if len(clean_core) >= 5:
        pref = clean_core[:5]
        if pref in country_prefix_idx[country]:
            for cid in country_prefix_idx[country][pref][:20]:
                cand_set.add(cid)
                    
    for t in tokens:
        if len(t) >= 5 and t in country_token_idx[country]:
            matches_t = country_token_idx[country][t]
            if len(matches_t) <= 35:
                for cid in matches_t[:15]:
                    cand_set.add(cid)
                            
    addr_keys = get_addr_keys(clean_addr, rec["addr_tokens"], rec["numbers"])
    for ak in addr_keys:
        if ak in country_addr_idx[country]:
            matches_ak = country_addr_idx[country][ak]
            if len(matches_ak) <= 25:
                for cid in matches_ak[:10]:
                    cand_set.add(cid)

    if len(cand_set) > top_k:
        scored = []
        for cid in cand_set:
            c_rec = candidates_pool[cid]
            name_score = rapidfuzz.fuzz.token_sort_ratio(clean_name, c_rec["clean_name"])
            addr_score = rapidfuzz.fuzz.token_sort_ratio(clean_addr, c_rec["clean_addr"]) if (clean_addr and c_rec["clean_addr"]) else 0
            scored.append((max(name_score, addr_score), cid))
        scored.sort(reverse=True, key=lambda x: x[0])
        cand_set = {cid for _, cid in scored[:top_k]}
        
    return cand_set

# Train Model
X_train = []
y_train = []

for sid in train_s1_ids:
    s1_rec = s1_records[sid]
    true_set = s1_gt[sid]
    cand_set = get_candidates(s1_rec, top_k=20)
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

# Validation Evaluation & Detailed Error Extraction
val_ground_truth = {sid: s1_gt[sid] for sid in val_s1_ids}
val_candidates = {}
val_pair_feats = []
val_pair_keys = []

for sid in val_s1_ids:
    s1_rec = s1_records[sid]
    cand_set = get_candidates(s1_rec, top_k=20)
    val_candidates[sid] = cand_set
    for cid in cand_set:
        cand_rec = candidates_pool[cid]
        val_pair_feats.append(compute_pairwise_features(s1_rec, cand_rec))
        val_pair_keys.append((sid, cid))

X_val = np.array(val_pair_feats, dtype=np.float32)
val_probs = model.predict_proba(X_val)[:, 1]

val_pair_prob_map = {}
val_s1_scores = defaultdict(list)
for (sid, cid), prob in zip(val_pair_keys, val_probs):
    val_pair_prob_map[(sid, cid)] = float(prob)
    val_s1_scores[sid].append((cid, float(prob)))

# Optimal threshold = 0.55
th = 0.55
val_predictions = {}
for sid in val_s1_ids:
    val_predictions[sid] = {cid for cid, prob in val_s1_scores[sid] if prob >= th}

# 1. ENTITY-LEVEL ERROR CLASSIFICATION
entity_error_counts = Counter()
entity_errors_detail = []

for sid in val_s1_ids:
    s1_rec = s1_records[sid]
    true_set = val_ground_truth[sid]
    pred_set = val_predictions[sid]
    
    if true_set == pred_set:
        continue  # exact match
        
    is_singleton = (len(true_set) == 0)
    
    if is_singleton:
        # Predicted something for singleton
        category = "singleton_false_positive"
    else:
        if len(pred_set) == 0:
            category = "complete_miss"
        else:
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            if tp == 0:
                category = "pure_false_positive"
            elif fp == 0 and fn > 0:
                category = "pure_false_negative (partial miss)"
            elif fp > 0 and fn == 0:
                category = "over_prediction (all true + extra FP)"
            else:
                category = "mixed_partial (some TP, some FP, some FN)"
                
    entity_error_counts[category] += 1
    entity_errors_detail.append({
        "sid": sid,
        "category": category,
        "country": s1_rec["country"],
        "true_count": len(true_set),
        "pred_count": len(pred_set),
        "true_set": list(true_set),
        "pred_set": list(pred_set)
    })

# 2. PAIR-LEVEL ERROR DECOMPOSITION: Category A, B, C
# Total true pairs
total_true_pairs = sum(len(m) for m in val_ground_truth.values())

cat_a = 0  # True match NEVER entered candidate set (blocking failure)
cat_b = 0  # True match was a candidate but rejected by LightGBM (classifier false negative)
cat_c = 0  # Incorrect candidate accepted by LightGBM (classifier false positive)

cat_a_by_country = Counter()
cat_b_by_country = Counter()
cat_c_by_country = Counter()

cat_a_by_source = Counter()
cat_b_by_source = Counter()
cat_c_by_source = Counter()

cat_a_name_scores = []
cat_b_name_scores = []
cat_a_addr_scores = []
cat_b_addr_scores = []

for sid, true_set in val_ground_truth.items():
    s1_rec = s1_records[sid]
    cands = val_candidates[sid]
    preds = val_predictions[sid]
    country = s1_rec["country"]
    
    # Check false negatives
    for true_id in true_set:
        src = "S2" if true_id.startswith("S2-") else "S3"
        t_rec = candidates_pool.get(true_id)
        name_sim = rapidfuzz.fuzz.token_sort_ratio(s1_rec["clean_name"], t_rec["clean_name"]) if t_rec else 0
        addr_sim = rapidfuzz.fuzz.token_sort_ratio(s1_rec["clean_addr"], t_rec["clean_addr"]) if (t_rec and s1_rec["clean_addr"] and t_rec["clean_addr"]) else 0
        
        if true_id not in cands:
            cat_a += 1
            cat_a_by_country[country] += 1
            cat_a_by_source[src] += 1
            cat_a_name_scores.append(name_sim)
            cat_a_addr_scores.append(addr_sim)
        else:
            if true_id not in preds:
                cat_b += 1
                cat_b_by_country[country] += 1
                cat_b_by_source[src] += 1
                cat_b_name_scores.append(name_sim)
                cat_b_addr_scores.append(addr_sim)
                
    # Check false positives
    for pred_id in preds:
        if pred_id not in true_set:
            src = "S2" if pred_id.startswith("S2-") else "S3"
            cat_c += 1
            cat_c_by_country[country] += 1
            cat_c_by_source[src] += 1

# Blocking metrics
candidate_counts = [len(c) for c in val_candidates.values()]
avg_cands = np.mean(candidate_counts)
med_cands = np.median(candidate_counts)
max_cands = np.max(candidate_counts)
cand_recall = ((total_true_pairs - cat_a) / total_true_pairs) * 100

total_potential_pairs = len(val_s1_ids) * len(candidates_pool)
total_actual_candidates = sum(candidate_counts)
cand_reduction = (1.0 - (total_actual_candidates / total_potential_pairs)) * 100

# 3. Threshold Analysis
threshold_table = []
for test_th in [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    temp_preds = {}
    for sid in val_s1_ids:
        temp_preds[sid] = {cid for cid, prob in val_s1_scores[sid] if prob >= test_th}
    m = compute_macro_f05(val_ground_truth, temp_preds)
    # Pair-level P & R
    tp = sum(len(val_ground_truth[sid] & temp_preds[sid]) for sid in val_s1_ids)
    fp = sum(len(temp_preds[sid] - val_ground_truth[sid]) for sid in val_s1_ids)
    fn = sum(len(val_ground_truth[sid] - temp_preds[sid]) for sid in val_s1_ids)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
    threshold_table.append({
        "threshold": test_th,
        "macro_f05": m["macro_f05"],
        "precision": p,
        "recall": r,
        "f1": f1,
        "singleton_fps": m["singleton_false_positives"]
    })

# Export error analysis summary to JSON
summary = {
    "total_validation_entities": len(val_s1_ids),
    "correct_entities": len(val_s1_ids) - len(entity_errors_detail),
    "incorrect_entities": len(entity_errors_detail),
    "entity_error_classification": dict(entity_error_counts),
    "pair_errors": {
        "total_true_matches": total_true_pairs,
        "cat_a_missed_by_blocking": cat_a,
        "cat_b_rejected_by_lightgbm": cat_b,
        "cat_c_accepted_incorrect_cands": cat_c,
        "cat_a_by_country": dict(cat_a_by_country),
        "cat_b_by_country": dict(cat_b_by_country),
        "cat_c_by_country": dict(cat_c_by_country),
        "cat_a_by_source": dict(cat_a_by_source),
        "cat_b_by_source": dict(cat_b_by_source),
        "cat_c_by_source": dict(cat_c_by_source),
        "cat_a_avg_name_sim": float(np.mean(cat_a_name_scores)) if cat_a_name_scores else 0,
        "cat_b_avg_name_sim": float(np.mean(cat_b_name_scores)) if cat_b_name_scores else 0,
        "cat_a_avg_addr_sim": float(np.mean(cat_a_addr_scores)) if cat_a_addr_scores else 0,
        "cat_b_avg_addr_sim": float(np.mean(cat_b_addr_scores)) if cat_b_addr_scores else 0
    },
    "blocking_stats": {
        "candidate_recall": float(cand_recall),
        "avg_candidates": float(avg_cands),
        "med_candidates": float(med_cands),
        "max_candidates": int(max_cands),
        "reduction_ratio": float(cand_reduction)
    },
    "threshold_table": threshold_table
}

with open(r"c:\jaishree_projects\amazon_ml_challenge\error_analysis_summary.json", 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print("Error analysis completed and saved to error_analysis_summary.json!", flush=True)
