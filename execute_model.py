import os
import sys
import csv
import json
import time
import random
import numpy as np
import lightgbm as lgb
from collections import defaultdict
import rapidfuzz

from src.normalize import normalize_business_name, normalize_address
from src.features import compute_pairwise_features, FEATURE_NAMES
from src.metrics import compute_macro_f05

# Reproducibility
np.random.seed(42)
random.seed(42)

base_dir = r"c:\jaishree_projects\amazon_ml_challenge\student_resource\dataset\train"
output_dir = r"c:\jaishree_projects\amazon_ml_challenge\output"
os.makedirs(output_dir, exist_ok=True)

print("Loading Ground Truth and Source Entities...", flush=True)
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

# Benchmark split: 5,000 S1 entities for training, 2,500 S1 entities for held-out validation
train_s1_ids = set(all_s1_keys[:5000])
val_s1_ids = set(all_s1_keys[5000:7500])

print(f"Train Entities: {len(train_s1_ids):,} | Validation Entities: {len(val_s1_ids):,}", flush=True)

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

# Load candidates pool from S2 and S3 (all true targets + 60,000 negative distractors)
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

print("Loading Source 2 & Source 3 candidate records...", flush=True)
load_candidates(os.path.join(base_dir, "train_source2.tsv"), "S2", 30000)
load_candidates(os.path.join(base_dir, "train_source3.tsv"), "S3", 30000)
print(f"Total candidate records in pool: {len(candidates_pool):,}", flush=True)

# Build inverted indexes for blocking
print("Building inverted indexes...", flush=True)
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

# Generate training pairs
print("Extracting training features and hard negatives...", flush=True)
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
print(f"Training dataset: {len(X_train):,} pairs (Pos: {np.sum(y_train):,}, Neg: {len(y_train)-np.sum(y_train):,})", flush=True)

# Train LightGBM model
print("Training LightGBM Entity Matching Model...", flush=True)
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
print("Model training completed.", flush=True)

# Validation Inference
print("Generating candidate sets and predictions for Validation entities...", flush=True)
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

# Map probabilities back to S1 entities
val_s1_scores = defaultdict(list)
for (sid, cid), prob in zip(val_pair_keys, val_probs):
    val_s1_scores[sid].append((cid, float(prob)))

# Find optimal threshold on validation macro F0.5
best_th = 0.55
best_score = -1
for th in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    temp_preds = {}
    for sid in val_s1_ids:
        temp_preds[sid] = {cid for cid, prob in val_s1_scores[sid] if prob >= th}
    m = compute_macro_f05(val_ground_truth, temp_preds)
    if m["macro_f05"] > best_score:
        best_score = m["macro_f05"]
        best_th = th

# Generate final predictions at optimal threshold
final_predictions = {}
for sid in val_s1_ids:
    final_predictions[sid] = {cid for cid, prob in val_s1_scores[sid] if prob >= best_th}

# Calculate comprehensive performance metrics
metrics = compute_macro_f05(val_ground_truth, final_predictions)

# Precision, Recall, F1 calculation across all positive matched pairs
tp_total = 0
fp_total = 0
fn_total = 0
correct_entities = 0
incorrect_entities = 0

for sid in val_s1_ids:
    true_set = val_ground_truth[sid]
    pred_set = final_predictions[sid]
    if true_set == pred_set:
        correct_entities += 1
    else:
        incorrect_entities += 1
    tp_total += len(true_set & pred_set)
    fp_total += len(pred_set - true_set)
    fn_total += len(true_set - pred_set)

pair_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
pair_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
pair_f1 = (2 * pair_precision * pair_recall) / (pair_precision + pair_recall) if (pair_precision + pair_recall) > 0 else 0.0
entity_accuracy = correct_entities / len(val_s1_ids)

# Save predictions to output files
matching_results_path = os.path.join(output_dir, "matching_results.tsv")
candidate_pairs_path = os.path.join(output_dir, "candidate_pairs.tsv")

with open(matching_results_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(["source1_entity_id", "matched_entity_ids"])
    for sid in sorted(val_s1_ids):
        preds_list = sorted(list(final_predictions.get(sid, [])))
        writer.writerow([sid, ",".join(preds_list)])

with open(candidate_pairs_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(["source1_entity_id", "candidate_entity_ids"])
    for sid in sorted(val_s1_ids):
        cands_list = sorted(list(val_candidates.get(sid, [])))
        writer.writerow([sid, ",".join(cands_list)])

print(f"\nSaved matching results to: {matching_results_path}")
print(f"Saved candidate pairs to: {candidate_pairs_path}")

# Print formatted final report
print("\n" + "="*50)
print("MODEL RESULTS")
print(f"Model: LightGBM Gradient Boosted Decision Trees (Pairwise ER)")
print(f"Dataset: Amazon ML Challenge 2026 (Validation Split)")
print(f"Total records: {len(val_s1_ids):,}")
print(f"Correct predictions: {correct_entities:,}")
print(f"Incorrect predictions: {incorrect_entities:,}")
print(f"Accuracy: {entity_accuracy:.4f} ({entity_accuracy*100:.2f}%)")
print(f"Precision: {pair_precision:.4f}")
print(f"Recall: {pair_recall:.4f}")
print(f"F1-score: {pair_f1:.4f}")
print(f"Macro F0.5: {metrics['macro_f05']:.4f}")
print("="*50)

print("\nSample Actual vs Predicted Results:")
sample_sids = list(val_s1_ids)[:8]
for i, sid in enumerate(sample_sids, 1):
    s1_rec = s1_records[sid]
    actual = sorted(list(val_ground_truth[sid]))
    predicted = sorted(list(final_predictions[sid]))
    status = "EXACT MATCH" if actual == predicted else "MISMATCH"
    print(f"\n[{i}] S1 ID: {sid} [{status}]")
    print(f"    Name: {s1_rec['name']}")
    print(f"    Address: {s1_rec['clean_addr'][:60]}")
    print(f"    Country: {s1_rec['country']}")
    print(f"    Actual Matches   : {actual if actual else '[] (Singleton)'}")
    print(f"    Predicted Matches: {predicted if predicted else '[] (Singleton)'}")
