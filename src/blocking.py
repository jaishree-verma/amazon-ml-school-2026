import re
from collections import defaultdict
import rapidfuzz
from src.normalize import has_non_latin, get_transliterated_name

ADDR_STOPWORDS = {
    'floor', 'road', 'street', 'avenue', 'near', 'block', 'lane', 'drive',
    'opposite', 'behind', 'beside', 'suite', 'unit', 'building', 'tower',
    'flat', 'shop', 'plot', 'sector', 'nagar', 'city', 'town', 'post', 'rue',
    'boulevard', 'chemin', 'impasse', 'place', 'allee', 'cedex'
}

def extract_pin_codes(addr_tokens, numbers):
    pins = []
    for num in numbers:
        if len(num) in (5, 6):
            pins.append(num)
    return pins

def extract_locality_tokens(addr_tokens):
    return [t for t in addr_tokens if len(t) >= 4 and not t.isdigit() and t not in ADDR_STOPWORDS]

# ==============================================================================
# VARIANT A: ENHANCED BLOCKING (Current Baseline from previous step)
# ==============================================================================

def get_enhanced_addr_keys(clean_name, addr_clean, addr_tokens, numbers):
    keys = []
    long_tokens = extract_locality_tokens(addr_tokens)
    pins = extract_pin_codes(addr_tokens, numbers)
    
    for num in numbers[:2]:
        for t in long_tokens[:3]:
            keys.append(f"{num}_{t}")
            
    for pin in pins:
        keys.append(f"pin_{pin}")
        
    if len(long_tokens) >= 2:
        for i in range(min(2, len(long_tokens) - 1)):
            keys.append(f"loc_{long_tokens[i]}_{long_tokens[i+1]}")
            
    name_clean_alnum = re.sub(r'[^a-z0-9]', '', clean_name.lower())
    if len(name_clean_alnum) >= 2:
        name_p2 = name_clean_alnum[:2]
        for pin in pins[:1]:
            keys.append(f"np_{name_p2}_{pin}")
        for t in long_tokens[:2]:
            keys.append(f"nl_{name_p2}_{t}")
            
    return keys

class EnhancedBlocking:
    """Variant A: Baseline Enhanced Blocking with top_k=30 and address protection."""
    def __init__(self):
        self.country_core_idx = defaultdict(lambda: defaultdict(list))
        self.country_prefix_idx = defaultdict(lambda: defaultdict(list))
        self.country_token_idx = defaultdict(lambda: defaultdict(list))
        self.country_2token_idx = defaultdict(lambda: defaultdict(list))
        self.country_addr_idx = defaultdict(lambda: defaultdict(list))

    def build_indexes(self, candidates_pool):
        for eid, rec in candidates_pool.items():
            country = rec["country"]
            core = rec["core_name"].replace(" ", "")
            if core:
                self.country_core_idx[country][core].append(eid)
                if len(core) >= 5:
                    self.country_prefix_idx[country][core[:5]].append(eid)
                    
            tokens = list(rec["tokens"])
            if len(tokens) >= 2:
                self.country_2token_idx[country][f"{tokens[0]}_{tokens[1]}"].append(eid)
                
            for t in tokens:
                if len(t) >= 4 and t not in ("hotel", "company", "services", "enterprises", "india", "pvt", "ltd"):
                    self.country_token_idx[country][t].append(eid)
                    
            addr_keys = get_enhanced_addr_keys(rec["clean_name"], rec["clean_addr"], rec["addr_tokens"], rec["numbers"])
            for ak in addr_keys:
                self.country_addr_idx[country][ak].append(eid)

    def get_candidates(self, rec, candidates_pool, top_k=30):
        country = rec["country"]
        clean_name = rec["clean_name"]
        clean_core = rec["core_name"].replace(" ", "")
        tokens = list(rec["tokens"])
        clean_addr = rec["clean_addr"]
        
        cand_set = set()
        if clean_core in self.country_core_idx[country]:
            cand_set.update(self.country_core_idx[country][clean_core])
            
        if len(tokens) >= 2:
            pair_key = f"{tokens[0]}_{tokens[1]}"
            if pair_key in self.country_2token_idx[country]:
                for cid in self.country_2token_idx[country][pair_key][:25]:
                    cand_set.add(cid)
                    
        if len(clean_core) >= 5:
            pref = clean_core[:5]
            if pref in self.country_prefix_idx[country]:
                for cid in self.country_prefix_idx[country][pref][:25]:
                    cand_set.add(cid)
                        
        for t in tokens:
            if len(t) >= 5 and t in self.country_token_idx[country]:
                matches_t = self.country_token_idx[country][t]
                if len(matches_t) <= 40:
                    for cid in matches_t[:20]:
                        cand_set.add(cid)
                                
        addr_keys = get_enhanced_addr_keys(clean_name, clean_addr, rec["addr_tokens"], rec["numbers"])
        for ak in addr_keys:
            if ak in self.country_addr_idx[country]:
                matches_ak = self.country_addr_idx[country][ak]
                if len(matches_ak) <= 30:
                    for cid in matches_ak[:15]:
                        cand_set.add(cid)

        if len(cand_set) > top_k:
            scored = []
            for cid in cand_set:
                c_rec = candidates_pool[cid]
                name_score = rapidfuzz.fuzz.token_sort_ratio(clean_name, c_rec["clean_name"])
                addr_score = rapidfuzz.fuzz.token_sort_ratio(clean_addr, c_rec["clean_addr"]) if (clean_addr and c_rec["clean_addr"]) else 0
                is_protected = (40 <= name_score <= 55 and addr_score >= 75)
                comb_score = max(name_score, addr_score)
                scored.append((comb_score, cid, is_protected))
                
            scored.sort(reverse=True, key=lambda x: x[0])
            selected = set()
            for _, cid, is_prot in scored:
                if is_prot and len(selected) < 8:
                    selected.add(cid)
            for _, cid, _ in scored:
                if cid not in selected:
                    selected.add(cid)
                if len(selected) >= top_k:
                    break
            cand_set = selected
            
        return cand_set

# ==============================================================================
# VARIANT B: ENHANCED + CROSS-SCRIPT TRANSLITERATION BLOCKING
# ==============================================================================

class TranslitBlocking(EnhancedBlocking):
    """
    Variant B: Enhanced Blocking + Cross-Script Transliteration layer.
    Indexes transliterated representations alongside original names and
    uses transliterated similarity during candidate ranking.
    """
    def __init__(self):
        super().__init__()
        self.country_trans_core_idx = defaultdict(lambda: defaultdict(list))
        self.country_trans_token_idx = defaultdict(lambda: defaultdict(list))

    def build_indexes(self, candidates_pool):
        super().build_indexes(candidates_pool)
        for eid, rec in candidates_pool.items():
            country = rec["country"]
            # Extract transliteration if candidate has non-Latin script
            t_clean, t_core, t_tokens = get_transliterated_name(rec["name"])
            rec["trans_clean"] = t_clean
            rec["trans_core"] = t_core
            rec["trans_tokens"] = set(t_tokens)
            
            if t_core:
                clean_tcore = t_core.replace(" ", "")
                self.country_trans_core_idx[country][clean_tcore].append(eid)
                if len(clean_tcore) >= 5:
                    self.country_prefix_idx[country][clean_tcore[:5]].append(eid)
                for t in t_tokens:
                    if len(t) >= 4 and t not in ("hotel", "company", "services", "enterprises", "india", "pvt", "ltd"):
                        self.country_trans_token_idx[country][t].append(eid)
                        
                # Transliterated prefix + PIN / locality
                name_clean_alnum = re.sub(r'[^a-z0-9]', '', t_clean)
                if len(name_clean_alnum) >= 2:
                    p2 = name_clean_alnum[:2]
                    pins = extract_pin_codes(rec["addr_tokens"], rec["numbers"])
                    long_tokens = extract_locality_tokens(rec["addr_tokens"])
                    for pin in pins[:1]:
                        self.country_addr_idx[country][f"np_{p2}_{pin}"].append(eid)
                    for lt in long_tokens[:2]:
                        self.country_addr_idx[country][f"nl_{p2}_{lt}"].append(eid)

    def get_candidates(self, rec, candidates_pool, top_k=30):
        # First retrieve standard enhanced candidates
        cand_set = super().get_candidates(rec, candidates_pool, top_k=999999) # get full union before pruning
        country = rec["country"]
        
        # Check if S1 or targets have transliteration
        s1_t_clean, s1_t_core, s1_t_tokens = get_transliterated_name(rec["name"])
        rec["trans_clean"] = s1_t_clean
        
        clean_core = rec["core_name"].replace(" ", "")
        tokens = list(rec["tokens"])
        
        # Query transliteration indexes
        if clean_core in self.country_trans_core_idx[country]:
            cand_set.update(self.country_trans_core_idx[country][clean_core])
            
        for t in tokens:
            if len(t) >= 5 and t in self.country_trans_token_idx[country]:
                matches_t = self.country_trans_token_idx[country][t]
                if len(matches_t) <= 35:
                    for cid in matches_t[:15]:
                        cand_set.add(cid)
                        
        if s1_t_core:
            clean_s1_tcore = s1_t_core.replace(" ", "")
            if clean_s1_tcore in self.country_core_idx[country]:
                cand_set.update(self.country_core_idx[country][clean_s1_tcore])
            if clean_s1_tcore in self.country_trans_core_idx[country]:
                cand_set.update(self.country_trans_core_idx[country][clean_s1_tcore])
            for t in s1_t_tokens:
                if len(t) >= 5 and t in self.country_token_idx[country]:
                    for cid in self.country_token_idx[country][t][:15]:
                        cand_set.add(cid)
                        
        # Pruning with transliteration awareness
        if len(cand_set) > top_k:
            clean_name = rec["clean_name"]
            clean_addr = rec["clean_addr"]
            scored = []
            
            for cid in cand_set:
                c_rec = candidates_pool[cid]
                # Compare original names
                score_orig = rapidfuzz.fuzz.token_sort_ratio(clean_name, c_rec["clean_name"])
                # Compare transliterated names if either is transliterated
                score_trans = 0
                if c_rec.get("trans_clean"):
                    score_trans = max(score_trans, rapidfuzz.fuzz.token_sort_ratio(clean_name, c_rec["trans_clean"]))
                if s1_t_clean:
                    score_trans = max(score_trans, rapidfuzz.fuzz.token_sort_ratio(s1_t_clean, c_rec["clean_name"]))
                name_score = max(score_orig, score_trans)
                
                addr_score = rapidfuzz.fuzz.token_sort_ratio(clean_addr, c_rec["clean_addr"]) if (clean_addr and c_rec["clean_addr"]) else 0
                is_protected = (40 <= name_score <= 55 and addr_score >= 75)
                comb_score = max(name_score, addr_score)
                scored.append((comb_score, cid, is_protected))
                
            scored.sort(reverse=True, key=lambda x: x[0])
            selected = set()
            for _, cid, is_prot in scored:
                if is_prot and len(selected) < 8:
                    selected.add(cid)
            for _, cid, _ in scored:
                if cid not in selected:
                    selected.add(cid)
                if len(selected) >= top_k:
                    break
            cand_set = selected
            
        return cand_set

# ==============================================================================
# VARIANT C: ENHANCED + TRANSLIT + ADDRESS FALLBACK & ADAPTIVE RETRIEVAL
# ==============================================================================

def get_address_fallback_keys(clean_name, addr_tokens, numbers):
    """
    Conservative address fallback keys for records where postal PIN is missing:
    - number + locality
    - locality bigrams
    - name_token + number
    """
    keys = []
    long_tokens = extract_locality_tokens(addr_tokens)
    # 1. number + locality
    for num in numbers[:2]:
        for t in long_tokens[:2]:
            keys.append(f"num_loc_{num}_{t}")
            
    # 2. name token + number
    name_clean_tokens = [t for t in clean_name.split() if len(t) >= 4 and t not in ('hotel', 'company', 'services', 'enterprises', 'private', 'limited', 'pvt', 'ltd')]
    for nt in name_clean_tokens[:2]:
        for num in numbers[:2]:
            keys.append(f"nt_num_{nt}_{num}")
            
    return keys

class AdaptiveBlocking(TranslitBlocking):
    """
    Variant C: Enhanced + Cross-Script + Address Fallback with Adaptive Secondary Retrieval.
    Keeps top_k=30 as standard, but allows controlled expansion up to 40 for difficult entities
    (missing PIN or low candidate density).
    """
    def __init__(self):
        super().__init__()
        self.country_fallback_idx = defaultdict(lambda: defaultdict(list))

    def build_indexes(self, candidates_pool):
        super().build_indexes(candidates_pool)
        for eid, rec in candidates_pool.items():
            country = rec["country"]
            f_keys = get_address_fallback_keys(rec["clean_name"], rec["addr_tokens"], rec["numbers"])
            for fk in f_keys:
                self.country_fallback_idx[country][fk].append(eid)

    def get_candidates(self, rec, candidates_pool, top_k=30):
        # Stage 1: Get standard candidate set
        cand_set = super().get_candidates(rec, candidates_pool, top_k=top_k)
        
        # Stage 2: Adaptive expansion ONLY if entity has missing PIN or weak candidate yield
        pins = extract_pin_codes(rec["addr_tokens"], rec["numbers"])
        needs_adaptive = (len(pins) == 0 or len(cand_set) < 10 or has_non_latin(rec["name"]))
        
        if needs_adaptive:
            country = rec["country"]
            adaptive_target_k = 40  # strict budget ceiling
            
            f_keys = get_address_fallback_keys(rec["clean_name"], rec["addr_tokens"], rec["numbers"])
            for fk in f_keys:
                if fk in self.country_fallback_idx[country]:
                    matches = self.country_fallback_idx[country][fk]
                    if len(matches) <= 25:
                        for cid in matches[:10]:
                            cand_set.add(cid)
                            if len(cand_set) >= adaptive_target_k:
                                break
                                
            # If still over budget, prune to adaptive_target_k
            if len(cand_set) > adaptive_target_k:
                clean_name = rec["clean_name"]
                clean_addr = rec["clean_addr"]
                scored = []
                for cid in cand_set:
                    c_rec = candidates_pool[cid]
                    name_score = rapidfuzz.fuzz.token_sort_ratio(clean_name, c_rec["clean_name"])
                    if c_rec.get("trans_clean"):
                        name_score = max(name_score, rapidfuzz.fuzz.token_sort_ratio(clean_name, c_rec["trans_clean"]))
                    addr_score = rapidfuzz.fuzz.token_sort_ratio(clean_addr, c_rec["clean_addr"]) if (clean_addr and c_rec["clean_addr"]) else 0
                    scored.append((max(name_score, addr_score), cid))
                scored.sort(reverse=True, key=lambda x: x[0])
                cand_set = {cid for _, cid in scored[:adaptive_target_k]}
                
        return cand_set
