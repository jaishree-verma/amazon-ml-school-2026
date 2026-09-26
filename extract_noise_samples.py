import os
import csv
import json

base_dir = r"c:\jaishree_projects\amazon_ml_challenge\student_resource\dataset\train"

# Let's read ground truth matches for first 5000 S1 entities
print("Loading sample ground truth...")
gt_samples = {}
with open(os.path.join(base_dir, "train_ground_truth.tsv"), 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for i, row in enumerate(reader):
        if not row: continue
        s1 = row[0].strip()
        matches = [m.strip() for m in row[1].split(',') if m.strip()]
        if matches:
            gt_samples[s1] = matches
        if len(gt_samples) >= 20000:
            break

s1_needed = set(gt_samples.keys())
s2_needed = set()
s3_needed = set()
for matches in gt_samples.values():
    for m in matches:
        if m.startswith("S2-"): s2_needed.add(m)
        elif m.startswith("S3-"): s3_needed.add(m)

print(f"Needed: {len(s1_needed)} S1, {len(s2_needed)} S2, {len(s3_needed)} S3")

s1_data = {}
with open(os.path.join(base_dir, "train_source1.tsv"), 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        if row[0] in s1_needed:
            s1_data[row[0]] = (row[1], row[2], row[3])
            if len(s1_data) == len(s1_needed): break

s2_data = {}
with open(os.path.join(base_dir, "train_source2.tsv"), 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        if row[0] in s2_needed:
            s2_data[row[0]] = (row[1], row[2], row[3])

s3_data = {}
with open(os.path.join(base_dir, "train_source3.tsv"), 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        if row[0] in s3_needed:
            s3_data[row[0]] = (row[1], row[2], row[3])

print("Finding noise patterns...")
noise_examples = []

for s1_id, matches in gt_samples.items():
    if s1_id not in s1_data: continue
    s1_name, s1_addr, s1_country = s1_data[s1_id]
    
    for m_id in matches:
        target = s2_data.get(m_id) if m_id.startswith("S2-") else s3_data.get(m_id)
        if not target: continue
        m_name, m_addr, m_country = target
        
        noise_examples.append({
            "s1_id": s1_id,
            "s1_name": s1_name,
            "s1_addr": s1_addr,
            "s1_country": s1_country,
            "match_id": m_id,
            "match_name": m_name,
            "match_addr": m_addr,
            "match_country": m_country
        })

with open(r"c:\jaishree_projects\amazon_ml_challenge\noise_samples.json", 'w', encoding='utf-8') as out_f:
    json.dump(noise_examples[:1000], out_f, indent=2)

print(f"Extracted {len(noise_examples)} pairs.")
