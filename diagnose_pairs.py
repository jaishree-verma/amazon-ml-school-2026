import json
from src.normalize import normalize_business_name, normalize_address

with open('noise_samples.json', 'r', encoding='utf-8') as f:
    samples = json.load(f)

print(f"Loaded {len(samples)} samples.")
for i, p in enumerate(samples[:15]):
    c1, core1, t1 = normalize_business_name(p['s1_name'])
    c2, core2, t2 = normalize_business_name(p['match_name'])
    print(f"[{i+1}] S1: {p['s1_name']}")
    print(f"     Clean: '{c1}' | Core: '{core1}' | Tokens: {t1}")
    print(f"    Match: {p['match_name']}")
    print(f"     Clean: '{c2}' | Core: '{core2}' | Tokens: {t2}")
    print(f"    Addr1: {p['s1_addr']}")
    print(f"    Addr2: {p['match_addr']}")
    print("-" * 50)
