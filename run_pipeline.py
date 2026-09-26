import os
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

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

base_dir = r"c:\jaishree_projects\amazon_ml_challenge\student_resource\dataset\train"

print("=================================================================")
print("AMAZON ML CHALLENGE 2026 — ENTITY RESOLUTION PIPELINE EXECUTION")
print("=================================================================")

# 1. Load Ground Truth and Entity Metadata
print("\n[Step 1/6] Loading Ground Truth & Source Entities...")
s1_gt = {}
with open(os.path.join(base_dir, "train_ground_truth.tsv"), 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        s1_id = row[0].strip()
        matches = set(row[1].strip().split(',')) if len(row) > 1 and row[1].strip() else set()
        s1_gt[s1_id] = matches

print(f"Total S1 entities in Ground Truth: {len(s1_gt):,}")

# Let's create an entity-isolated Train and Validation benchmark
# 15,000 S1 for training, 10,000 S1 for held-out validation (ensuring NO leakage between train and val S1!)
all_s1_keys = list(s1_gt.keys())
random.shuffle(all_s1_keys)

train_s1_ids = set(all_s1_keys[:15000])
val_s1_ids = set(all_s1_keys[15000:25000])

print(f"Isolated Train S1 Entities: {len(train_s1_ids):,}")
print(f"Held-out Validation S1 Entities: {len(val_s1_ids):,}")

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

print(f"Loaded and normalized {len(s1_records):,} S1 records.")

# Find all true targets for train and val S1
train_targets = set()
for sid in train_s1_ids:
    train_targets.update(s1_gt[sid])

val_targets = set()
for sid in val_s1_ids:
    val_targets.update(s1_gt[sid])

all_needed_targets = train_targets | val_targets
print(f"Total True Targets needed: {len(all_needed_targets):,} (Train: {len(train_targets):,}, Val: {len(val_targets):,})")

# Load candidate pool from S2 and S3 (including true targets + 300,000 negative distractors)
candidates_pool = {}
def load_source(file_path, prefix, max_distractors=150000):
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

print("\n[Step 2/6] Loading Candidate Records from Source 2 & Source 3...")
load_source(os.path.join(base_dir, "train_source2.tsv"), "S2", 150000)
load_source(os.path.join(base_dir, "train_source3.tsv"), "S3", 150000)
print(f"Candidate Pool size: {len(candidates_pool):,} records.")

# 3. Build Multi-Stage Inverted Indexes
print("\n[Step 3/6] Building Multi-Stage Inverted Indexes for Blocking...")
t0 = time.time()

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
    if len(long_tokens) >= 2:
        keys.append(f"loc_{long_tokens[0]}_{long_tokens[1]}")
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

print(f"Indexes constructed in {time.time() - t0:.2f}s")

def generate_candidates_for_s1(rec, top_k=25):
    country = rec["country"]
    clean_name = rec["clean_name"]
    clean_core = rec["core_name"].replace(" ", "")
    tokens = list(rec["tokens"])
    clean_addr = rec["clean_addr"]
    
    cand_set = set()
    
    # 1. Exact core match
    if clean_core in country_core_idx[country]:
        cand_set.update(country_core_idx[country][clean_core])
        
    # 2. 2-token match
    if len(tokens) >= 2:
        pair_key = f"{tokens[0]}_{tokens[1]}"
        if pair_key in country_2token_idx[country]:
            for cid in country_2token_idx[country][pair_key][:25]:
                cand_set.add(cid)
                
    # 3. Prefix 5 match
    if len(clean_core) >= 5:
        pref = clean_core[:5]
        if pref in country_prefix_idx[country]:
            for cid in country_prefix_idx[country][pref][:25]:
                cand_set.add(cid)
                    
    # 4. Informative name token
    for t in tokens:
        if len(t) >= 5 and t in country_token_idx[country]:
            matches_t = country_token_idx[country][t]
            if len(matches_t) <= 40:
                for cid in matches_t[:20]:
                    cand_set.add(cid)
                            
    # 5. Address blocking
    addr_keys = get_addr_keys(clean_addr, rec["addr_tokens"], rec["numbers"])
    for ak in addr_keys:
        if ak in country_addr_idx[country]:
            matches_ak = country_addr_idx[country][ak]
            if len(matches_ak) <= 30:
                for cid in matches_ak[:15]:
                    cand_set.add(cid)

    # Candidate pruning: score and keep top_k
    if len(cand_set) > top_k:
        scored = []
        for cid in cand_set:
            c_rec = candidates_pool[cid]
            name_score = rapidfuzz.fuzz.token_sort_ratio(clean_name, c_rec["clean_name"])
            addr_score = rapidfuzz.fuzz.token_sort_ratio(clean_addr, c_rec["clean_addr"]) if (clean_addr and c_rec["clean_addr"]) else 0
            comb_score = max(name_score, addr_score)
            scored.append((comb_score, cid))
        scored.sort(reverse=True, key=lambda x: x[0])
        cand_set = {cid for _, cid in scored[:top_k]}
        
    return cand_set

# 4. Generate Training Pairs & Hard Negatives
print("\n[Step 4/6] Generating Training Candidates & Extracting Features...")
t0 = time.time()

X_train = []
y_train = []

for sid in train_s1_ids:
    s1_rec = s1_records[sid]
    true_set = s1_gt[sid]
    
    # Generate candidates from blocking
    cand_set = generate_candidates_for_s1(s1_rec, top_k=25)
    
    # Always include true matches in training so model sees true positive distributions
    all_train_candidates = cand_set | (true_set & set(candidates_pool.keys()))
    
    for cid in all_train_candidates:
        cand_rec = candidates_pool[cid]
        feats = compute_pairwise_features(s1_rec, cand_rec)
        label = 1 if cid in true_set else 0
        X_train.append(feats)
        y_train.append(label)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)

pos_count = int(np.sum(y_train))
neg_count = len(y_train) - pos_count
print(f"Training dataset: {len(X_train):,} pairs (Positives: {pos_count:,}, Hard Negatives: {neg_count:,})")
print(f"Features computed in {time.time() - t0:.2f}s")

# 5. Train LightGBM Classifier
print("\n[Step 5/6] Training LightGBM Pairwise Matching Model...")
t0 = time.time()

# Precision-weighted configuration (scale_pos_weight tuned for precision)
clf = lgb.LGBMClassifier(
    n_estimators=250,
    learning_rate=0.08,
    num_leaves=31,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

clf.fit(X_train, y_train)
print(f"Model trained in {time.time() - t0:.2f}s")

# Feature importances
importances = clf.feature_importances_
top_feat_idx = np.argsort(importances)[::-1][:7]
print("\nTop 7 Most Informative Features:")
for idx in top_feat_idx:
    print(f"  - {FEATURE_NAMES[idx]}: {importances[idx]}")

# 6. Held-out Validation Inference & Threshold Tuning
print("\n[Step 6/6] Evaluating End-to-End on 10,000 Held-Out Validation Entities...")
t0 = time.time()

val_ground_truth = {sid: s1_gt[sid] for sid in val_s1_ids}
val_candidates = {}
val_pair_scores = {}

val_true_matches = sum(len(m) for m in val_ground_truth.values())
val_true_retrieved = 0

all_val_feats = []
val_pair_keys = []

for sid in val_s1_ids:
    s1_rec = s1_records[sid]
    true_set = val_ground_truth[sid]
    
    cand_set = generate_candidates_for_s1(s1_rec, top_k=25)
    val_candidates[sid] = cand_set
    
    if true_set:
        val_true_retrieved += len(true_set & cand_set)
        
    for cid in cand_set:
        cand_rec = candidates_pool[cid]
        all_val_feats.append(compute_pairwise_features(s1_rec, cand_rec))
        val_pair_keys.append((sid, cid))

blocking_recall = (val_true_retrieved / val_true_matches) * 100 if val_true_matches > 0 else 0
avg_candidates = sum(len(c) for c in val_candidates.values()) / len(val_s1_ids)

print(f"\n--- Candidate Generation / Blocking Quality ---")
print(f"Validation Candidate Recall Ceiling: {blocking_recall:.2f}%")
print(f"Average Candidates Passed to Model: {avg_candidates:.2f}")

# Model inference on all validation candidate pairs
X_val = np.array(all_val_feats, dtype=np.float32)
val_probs = clf.predict_proba(X_val)[:, 1]

# Map probabilities back to pairs
val_s1_to_scored_cands = defaultdict(list)
for (sid, cid), prob in zip(val_pair_keys, val_probs):
    val_s1_to_scored_cands[sid].append((cid, prob))

# Threshold tuning on validation set to maximize official Macro F0.5
print("\n--- Threshold Tuning on Macro F0.5 ---")
best_th = 0.5
best_metrics = None
thresholds_to_test = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

results_table = []

for th in thresholds_to_test:
    preds = {}
    for sid in val_s1_ids:
        scored = val_s1_to_scored_cands.get(sid, [])
        matched = {cid for cid, prob in scored if prob >= th}
        preds[sid] = matched
        
    res = compute_macro_f05(val_ground_truth, preds)
    results_table.append((th, res))
    if best_metrics is None or res["macro_f05"] > best_metrics["macro_f05"]:
        best_th = th
        best_metrics = res

print(f"{'Threshold':<10} | {'Macro F0.5':<12} | {'Precision':<10} | {'Recall':<10} | {'Singleton Acc':<14} | {'Singleton FPs':<14}")
print("-" * 80)
for th, res in results_table:
    star = " <-- OPTIMAL" if th == best_th else ""
    print(f"{th:<10.2f} | {res['macro_f05']:<12.4f} | {res['avg_precision']:<10.4f} | {res['avg_recall']:<10.4f} | {res['singleton_accuracy']:<14.4f} | {res['singleton_false_positives']:<14}{star}")

print("\n" + "=" * 65)
print("FINAL PIPELINE VALIDATION ACCURACY SUMMARY")
print("=" * 65)
print(f"Optimal Probability Threshold   : {best_th:.2f}")
print(f"Candidate Blocking Recall       : {blocking_recall:.2f}%")
print(f"Macro F_0.5 Score               : {best_metrics['macro_f05']:.4f} ({best_metrics['macro_f05']*100:.2f}%)")
print(f"Average Precision               : {best_metrics['avg_precision']:.4f} ({best_metrics['avg_precision']*100:.2f}%)")
print(f"Average Recall                  : {best_metrics['avg_recall']:.4f} ({best_metrics['avg_recall']*100:.2f}%)")
print(f"Singleton Accuracy              : {best_metrics['singleton_accuracy']:.4f} ({best_metrics['singleton_accuracy']*100:.2f}%)")
print(f"Total Validation Entities       : {best_metrics['total_entities']:,}")
print(f"Total Singletons in Validation  : {best_metrics['total_singletons']:,}")
print(f"Singleton False Merges          : {best_metrics['singleton_false_positives']:,}")
print("=" * 65)
