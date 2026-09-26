import numpy as np
import rapidfuzz

def compute_pairwise_features(s1_rec, cand_rec):
    """
    Computes a vector of dense similarity features between an S1 entity and a candidate record.
    """
    s1_name_clean = s1_rec["clean_name"]
    cand_name_clean = cand_rec["clean_name"]
    
    s1_core = s1_rec["core_name"]
    cand_core = cand_rec["core_name"]
    
    s1_tokens = s1_rec["tokens"]
    cand_tokens = cand_rec["tokens"]
    
    # 1. Name Features
    name_sort_ratio = rapidfuzz.fuzz.token_sort_ratio(s1_name_clean, cand_name_clean) / 100.0
    name_set_ratio = rapidfuzz.fuzz.token_set_ratio(s1_name_clean, cand_name_clean) / 100.0
    name_ratio = rapidfuzz.fuzz.ratio(s1_name_clean, cand_name_clean) / 100.0
    name_partial_ratio = rapidfuzz.fuzz.partial_ratio(s1_name_clean, cand_name_clean) / 100.0
    
    # Core name match
    core_exact = 1.0 if (s1_core and cand_core and s1_core == cand_core) else 0.0
    
    # Token Jaccard
    intersection = len(s1_tokens & cand_tokens)
    union = len(s1_tokens | cand_tokens)
    token_jaccard = intersection / union if union > 0 else 0.0
    
    # Length features
    l1 = len(s1_name_clean)
    l2 = len(cand_name_clean)
    name_len_diff = abs(l1 - l2)
    name_len_ratio = min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 1.0
    
    # 2. Address Features
    s1_addr = s1_rec["clean_addr"]
    cand_addr = cand_rec["clean_addr"]
    cand_addr_empty = 1.0 if not cand_addr else 0.0
    
    if s1_addr and cand_addr:
        addr_sort_ratio = rapidfuzz.fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
        addr_set_ratio = rapidfuzz.fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0
        addr_ratio = rapidfuzz.fuzz.ratio(s1_addr, cand_addr) / 100.0
        
        # Address token Jaccard
        a_t1 = set(s1_addr.split())
        a_t2 = set(cand_addr.split())
        a_inter = len(a_t1 & a_t2)
        a_union = len(a_t1 | a_t2)
        addr_jaccard = a_inter / a_union if a_union > 0 else 0.0
        
        # Numeric tokens overlap (house numbers, postal codes)
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
        source_is_s3
    ]

FEATURE_NAMES = [
    "name_sort_ratio",
    "name_set_ratio",
    "name_ratio",
    "name_partial_ratio",
    "core_exact",
    "token_jaccard",
    "token_intersection",
    "name_len_diff",
    "name_len_ratio",
    "addr_sort_ratio",
    "addr_set_ratio",
    "addr_ratio",
    "addr_jaccard",
    "num_overlap",
    "has_common_num",
    "cand_addr_empty",
    "source_is_s2",
    "source_is_s3"
]
