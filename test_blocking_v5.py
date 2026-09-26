import os
import csv
import json
import time
from collections import defaultdict
import rapidfuzz
from src.normalize import normalize_business_name, normalize_address

base_dir = r"c:\jaishree_projects\amazon_ml_challenge\student_resource\dataset\train"

print("Loading 10,000 validation S1 records...")
val_s1 = {}
val_gt = {}

with open(os.path.join(base_dir, "train_ground_truth.tsv"), 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for i, row in enumerate(reader):
        s1_id = row[0].strip()
        matches = set(row[1].strip().split(',')) if len(row) > 1 and row[1].strip() else set()
        val_gt[s1_id] = matches
        if len(val_gt) >= 10000:
            break

s1_ids = set(val_gt.keys())
with open(os.path.join(base_dir, "train_source1.tsv"), 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        if row[0] in s1_ids:
            val_s1[row[0]] = {
                "id": row[0],
                "name": row[1],
                "addr": row[2],
                "country": row[3]
            }

true_targets = set()
for m_set in val_gt.values():
    true_targets.update(m_set)

candidates_pool = {}
val_countries = {v["country"] for v in val_s1.values()}

def load_pool(file_path, prefix, max_distractors=150000):
    distractors = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            eid = row[0]
            if eid in true_targets or (distractors < max_distractors and row[3] in val_countries):
                candidates_pool[eid] = {
                    "id": eid,
                    "name": row[1],
                    "addr": row[2],
                    "country": row[3],
                    "source": prefix
                }
                if eid not in true_targets:
                    distractors += 1

load_pool(os.path.join(base_dir, "train_source2.tsv"), "S2")
load_pool(os.path.join(base_dir, "train_source3.tsv"), "S3")

print("Extracting features and building inverted indexes...")
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
    clean_name, core_name, tokens = normalize_business_name(rec["name"])
    clean_addr, addr_tokens, numbers = normalize_address(rec["addr"])
    
    rec["clean_name"] = clean_name
    rec["core_name"] = core_name
    rec["clean_addr"] = clean_addr
    rec["tokens"] = set(tokens)
    
    clean_core = core_name.replace(" ", "")
    if clean_core:
        country_core_idx[country][clean_core].append(eid)
        if len(clean_core) >= 5:
            country_prefix_idx[country][clean_core[:5]].append(eid)
            
    if len(tokens) >= 2:
        pair_key = f"{tokens[0]}_{tokens[1]}"
        country_2token_idx[country][pair_key].append(eid)
        
    for t in tokens:
        if len(t) >= 4 and t not in ("hotel", "company", "services", "enterprises", "india", "pvt", "ltd"):
            country_token_idx[country][t].append(eid)
            
    addr_keys = get_addr_keys(clean_addr, addr_tokens, numbers)
    for ak in addr_keys:
        country_addr_idx[country][ak].append(eid)

print(f"Indexes built in {time.time() - t0:.2f}s")

# Run candidate generation
print("Running candidate generation with enhanced keys...")
t0 = time.time()

candidates_per_s1 = {}
true_matches_found = 0
total_true_matches = sum(len(m) for m in val_gt.values())

for s1_id, rec in val_s1.items():
    country = rec["country"]
    clean_name, core_name, tokens = normalize_business_name(rec["name"])
    clean_addr, addr_tokens, numbers = normalize_address(rec["addr"])
    clean_core = core_name.replace(" ", "")
    
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
    addr_keys = get_addr_keys(clean_addr, addr_tokens, numbers)
    for ak in addr_keys:
        if ak in country_addr_idx[country]:
            matches_ak = country_addr_idx[country][ak]
            if len(matches_ak) <= 30:
                for cid in matches_ak[:15]:
                    cand_set.add(cid)

    # Candidate pruning: score and keep top 25
    if len(cand_set) > 25:
        scored = []
        for cid in cand_set:
            c_rec = candidates_pool[cid]
            name_score = rapidfuzz.fuzz.token_sort_ratio(clean_name, c_rec["clean_name"])
            addr_score = rapidfuzz.fuzz.token_sort_ratio(clean_addr, c_rec["clean_addr"]) if (clean_addr and c_rec["clean_addr"]) else 0
            comb_score = max(name_score, addr_score)
            scored.append((comb_score, cid))
        scored.sort(reverse=True, key=lambda x: x[0])
        cand_set = {cid for _, cid in scored[:25]}
        
    candidates_per_s1[s1_id] = cand_set
    
    true_m = val_gt[s1_id]
    if true_m:
        found = len(true_m & cand_set)
        true_matches_found += found

gen_time = time.time() - t0
cand_recall = true_matches_found / total_true_matches if total_true_matches > 0 else 0
avg_cands = sum(len(c) for c in candidates_per_s1.values()) / len(val_s1)
med_cands = sorted(len(c) for c in candidates_per_s1.values())[len(val_s1)//2]

print("="*50)
print(f"ENHANCED BLOCKING RESULTS ON 10,000 VALIDATION S1 ENTITIES:")
print(f"Time Taken: {gen_time:.2f}s ({len(val_s1)/gen_time:.1f} S1/sec)")
print(f"Total True Matches: {total_true_matches}")
print(f"True Matches Retrieved: {true_matches_found}")
print(f"Candidate Recall: {cand_recall*100:.2f}%")
print(f"Average Candidates per S1: {avg_cands:.2f}")
print(f"Median Candidates per S1: {med_cands}")
print("="*50)
