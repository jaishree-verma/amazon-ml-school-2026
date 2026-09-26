import os
import sys
import json
import csv
from collections import Counter, defaultdict

base_dir = r"c:\jaishree_projects\amazon_ml_challenge\student_resource\dataset"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

def analyze_source_file(file_path):
    print(f"Analyzing {os.path.basename(file_path)}...")
    size_bytes = os.path.getsize(file_path)
    
    row_count = 0
    missing_name = 0
    missing_addr = 0
    missing_country = 0
    empty_name = 0
    empty_addr = 0
    empty_country = 0
    
    countries = Counter()
    name_lengths = []
    addr_lengths = []
    unique_names = set()
    unique_ids = set()
    duplicate_ids = 0
    malformed_rows = 0
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)
        col_count = len(header) if header else 0
        
        for i, row in enumerate(reader):
            row_count += 1
            if len(row) != 4:
                malformed_rows += 1
                continue
            
            eid, name, addr, country = row[0], row[1], row[2], row[3]
            
            if eid in unique_ids:
                duplicate_ids += 1
            else:
                unique_ids.add(eid)
                
            if not name.strip():
                empty_name += 1
            else:
                if len(unique_names) < 500000: # avoid memory blowup
                    unique_names.add(name.strip())
                name_lengths.append(len(name.strip()))
                
            if not addr.strip():
                empty_addr += 1
            else:
                addr_lengths.append(len(addr.strip()))
                
            if not country.strip():
                empty_country += 1
            else:
                countries[country.strip()] += 1
                
    # Calculate stats
    avg_name_len = sum(name_lengths) / len(name_lengths) if name_lengths else 0
    name_lengths.sort()
    med_name_len = name_lengths[len(name_lengths)//2] if name_lengths else 0
    
    avg_addr_len = sum(addr_lengths) / len(addr_lengths) if addr_lengths else 0
    addr_lengths.sort()
    med_addr_len = addr_lengths[len(addr_lengths)//2] if addr_lengths else 0
    
    return {
        "filename": os.path.basename(file_path),
        "size_mb": round(size_bytes / (1024*1024), 2),
        "rows": row_count,
        "columns": header,
        "col_count": col_count,
        "duplicate_ids": duplicate_ids,
        "malformed_rows": malformed_rows,
        "empty_name": empty_name,
        "empty_addr": empty_addr,
        "empty_country": empty_country,
        "countries": dict(countries),
        "avg_name_len": round(avg_name_len, 2),
        "med_name_len": med_name_len,
        "avg_addr_len": round(avg_addr_len, 2),
        "med_addr_len": med_addr_len,
        "sample_unique_names_count": len(unique_names)
    }

def analyze_ground_truth(file_path):
    print(f"Analyzing {os.path.basename(file_path)}...")
    size_bytes = os.path.getsize(file_path)
    
    total_s1 = 0
    duplicate_s1 = 0
    seen_s1 = set()
    
    match_counts = Counter()
    s2_matches = 0
    s3_matches = 0
    other_matches = 0
    
    singletons = 0
    exact_one = 0
    multi_match = 0
    
    # Store some interesting match examples to fetch noise patterns later
    sample_matches = []
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)
        
        for row in reader:
            if not row:
                continue
            total_s1 += 1
            s1_id = row[0].strip()
            if s1_id in seen_s1:
                duplicate_s1 += 1
            seen_s1.add(s1_id)
            
            raw_matches = row[1].strip() if len(row) > 1 else ""
            if not raw_matches:
                match_counts[0] += 1
                singletons += 1
            else:
                m_list = [m.strip() for m in raw_matches.split(',') if m.strip()]
                n = len(m_list)
                match_counts[n] += 1
                if n == 1:
                    exact_one += 1
                else:
                    multi_match += 1
                
                for m in m_list:
                    if m.startswith("S2-"):
                        s2_matches += 1
                    elif m.startswith("S3-"):
                        s3_matches += 1
                    else:
                        other_matches += 1
                        
                if len(sample_matches) < 200:
                    sample_matches.append((s1_id, m_list))
                    
    return {
        "filename": os.path.basename(file_path),
        "size_mb": round(size_bytes / (1024*1024), 2),
        "total_s1": total_s1,
        "columns": header,
        "duplicate_s1": duplicate_s1,
        "singletons": singletons,
        "singleton_pct": round(singletons / total_s1 * 100, 2) if total_s1 else 0,
        "exact_one": exact_one,
        "exact_one_pct": round(exact_one / total_s1 * 100, 2) if total_s1 else 0,
        "multi_match": multi_match,
        "multi_match_pct": round(multi_match / total_s1 * 100, 2) if total_s1 else 0,
        "match_counts": dict(sorted(match_counts.items())),
        "total_matches": s2_matches + s3_matches + other_matches,
        "s2_matches": s2_matches,
        "s3_matches": s3_matches,
        "other_matches": other_matches,
        "sample_matches": sample_matches
    }

print("Starting analysis...")
results = {}

for f in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
    results[f] = analyze_source_file(os.path.join(train_dir, f))

results["train_ground_truth.tsv"] = analyze_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))

for f in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
    results[f] = analyze_source_file(os.path.join(test_dir, f))

# Save results
out_path = r"c:\jaishree_projects\amazon_ml_challenge\dataset_stats.json"
with open(out_path, 'w', encoding='utf-8') as out_f:
    json.dump(results, out_f, indent=2)

print("Finished! Saved stats to", out_path)
