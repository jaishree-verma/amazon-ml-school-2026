import numpy as np
import rapidfuzz
from src.blocking import extract_pin_codes, extract_locality_tokens

FEATURE_NAMES = [
    # 1. Base Name Features (0-8)
    "name_sort_ratio",
    "name_set_ratio",
    "name_ratio",
    "name_partial_ratio",
    "core_exact",
    "token_jaccard",
    "token_intersection",
    "name_len_diff",
    "name_len_ratio",
    # 2. Base Address Features (9-15)
    "addr_sort_ratio",
    "addr_set_ratio",
    "addr_ratio",
    "addr_jaccard",
    "num_overlap",
    "has_common_num",
    "cand_addr_empty",
    # 3. Source Features (16-17)
    "source_is_s2",
    "source_is_s3",
    # 4. Enhanced Name & Cross-Script Features (18-23)
    "jaro_winkler_similarity",
    "token_containment",
    "first_token_match",
    "acronym_match",
    "translit_name_ratio",
    "cross_script_flag",
    # 5. Enhanced Address & Locality Features (24-27)
    "pin_match_status",
    "house_num_status",
    "locality_jaccard",
    "addr_len_ratio",
    # 6. Candidate Ranking / Heuristic Features (28-29)
    "cand_heuristic_score",
    "cand_rank_normalized"
]

def compute_pairwise_features(s1_rec, cand_rec, rank=0, total_cands=1):
    """
    Computes the validated 30-dimensional dense feature vector between an S1 entity and a candidate.
    """
    s1_name_clean = s1_rec["clean_name"]
    cand_name_clean = cand_rec["clean_name"]
    
    s1_core = s1_rec["core_name"]
    cand_core = cand_rec["core_name"]
    
    s1_tokens = s1_rec["tokens"]
    cand_tokens = cand_rec["tokens"]
    
    # 1. Base Name Features
    name_sort_ratio = rapidfuzz.fuzz.token_sort_ratio(s1_name_clean, cand_name_clean) / 100.0
    name_set_ratio = rapidfuzz.fuzz.token_set_ratio(s1_name_clean, cand_name_clean) / 100.0
    name_ratio = rapidfuzz.fuzz.ratio(s1_name_clean, cand_name_clean) / 100.0
    name_partial_ratio = rapidfuzz.fuzz.partial_ratio(s1_name_clean, cand_name_clean) / 100.0
    
    core_exact = 1.0 if (s1_core and cand_core and s1_core == cand_core) else 0.0
    
    intersection = len(s1_tokens & cand_tokens)
    union = len(s1_tokens | cand_tokens)
    token_jaccard = intersection / union if union > 0 else 0.0
    
    l1 = len(s1_name_clean)
    l2 = len(cand_name_clean)
    name_len_diff = abs(l1 - l2)
    name_len_ratio = min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 1.0
    
    # 2. Base Address Features
    s1_addr = s1_rec["clean_addr"]
    cand_addr = cand_rec["clean_addr"]
    cand_addr_empty = 1.0 if not cand_addr else 0.0
    
    if s1_addr and cand_addr:
        addr_sort_ratio = rapidfuzz.fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
        addr_set_ratio = rapidfuzz.fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0
        addr_ratio = rapidfuzz.fuzz.ratio(s1_addr, cand_addr) / 100.0
        
        a_t1 = set(s1_addr.split())
        a_t2 = set(cand_addr.split())
        a_inter = len(a_t1 & a_t2)
        a_union = len(a_t1 | a_t2)
        addr_jaccard = a_inter / a_union if a_union > 0 else 0.0
        
        n1 = set(s1_rec.get("numbers", []))
        n2 = set(cand_rec.get("numbers", []))
        num_overlap = len(n1 & n2)
        has_common_num = 1.0 if num_overlap > 0 else 0.0
    else:
        addr_sort_ratio = 0.0
        addr_set_ratio = 0.0
        addr_ratio = 0.0
        addr_jaccard = 0.0
        num_overlap = 0.0
        has_common_num = 0.0
        
    # 3. Source Features
    source_is_s2 = 1.0 if cand_rec.get("source") == "S2" or cand_rec["id"].startswith("S2-") else 0.0
    source_is_s3 = 1.0 if cand_rec.get("source") == "S3" or cand_rec["id"].startswith("S3-") else 0.0
    
    # 4. Enhanced Name & Cross-Script Features
    jw_sim = rapidfuzz.distance.JaroWinkler.similarity(s1_name_clean, cand_name_clean)
    min_t = min(len(s1_tokens), len(cand_tokens))
    containment = (intersection / min_t) if min_t > 0 else 0.0
    
    t1_s1 = s1_name_clean.split()[0] if s1_name_clean else ""
    t1_cand = cand_name_clean.split()[0] if cand_name_clean else ""
    first_tok_match = 1.0 if (t1_s1 and t1_cand and t1_s1 == t1_cand) else 0.0
    
    s1_words = [w for w in s1_name_clean.split() if w]
    cand_words = [w for w in cand_name_clean.split() if w]
    s1_acro = "".join(w[0] for w in s1_words[:4])
    cand_acro = "".join(w[0] for w in cand_words[:4])
    acro_match = 1.0 if (s1_acro and cand_acro and (s1_acro == cand_name_clean or cand_acro == s1_name_clean or s1_acro == cand_acro)) else 0.0
    
    translit_ratio = 0.0
    cross_script = 0.0
    if cand_rec.get("trans_clean"):
        cross_script = 1.0
        translit_ratio = max(translit_ratio, rapidfuzz.fuzz.token_sort_ratio(s1_name_clean, cand_rec["trans_clean"]) / 100.0)
    if s1_rec.get("trans_clean"):
        cross_script = 1.0
        translit_ratio = max(translit_ratio, rapidfuzz.fuzz.token_sort_ratio(s1_rec["trans_clean"], cand_name_clean) / 100.0)
        
    # 5. Enhanced Address & PIN Features
    p1 = extract_pin_codes(s1_rec.get("addr_tokens", []), s1_rec.get("numbers", []))
    p2 = extract_pin_codes(cand_rec.get("addr_tokens", []), cand_rec.get("numbers", []))
    if p1 and p2:
        pin_status = 1.0 if set(p1) & set(p2) else -1.0
    else:
        pin_status = 0.0
        
    num1 = s1_rec.get("numbers", [])
    num2 = cand_rec.get("numbers", [])
    if num1 and num2:
        house_status = 1.0 if num1[0] == num2[0] else -1.0
    else:
        house_status = 0.0
        
    loc1 = set(extract_locality_tokens(s1_rec.get("addr_tokens", [])))
    loc2 = set(extract_locality_tokens(cand_rec.get("addr_tokens", [])))
    loc_inter = len(loc1 & loc2)
    loc_union = len(loc1 | loc2)
    loc_jaccard = (loc_inter / loc_union) if loc_union > 0 else 0.0
    
    l_a1 = len(s1_addr)
    l_a2 = len(cand_addr)
    addr_len_ratio = (min(l_a1, l_a2) / max(l_a1, l_a2)) if max(l_a1, l_a2) > 0 else 1.0
    
    # 6. Candidate Ranking / Heuristic Score
    cand_heur = max(name_sort_ratio, addr_sort_ratio)
    cand_rank_norm = (rank / total_cands) if total_cands > 1 else 0.0
    
    return [
        name_sort_ratio,
        name_set_ratio,
        name_ratio,
        name_partial_ratio,
        core_exact,
        token_jaccard,
        intersection,
        name_len_diff,
        name_len_ratio,
        addr_sort_ratio,
        addr_set_ratio,
        addr_ratio,
        addr_jaccard,
        num_overlap,
        has_common_num,
        cand_addr_empty,
        source_is_s2,
        source_is_s3,
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
