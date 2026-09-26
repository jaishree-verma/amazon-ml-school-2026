import json
import re

with open(r"c:\jaishree_projects\amazon_ml_challenge\noise_samples.json", 'r', encoding='utf-8') as f:
    pairs = json.load(f)

categories = {
    "abbreviations": [],
    "legal_suffix": [],
    "typos": [],
    "punctuation": [],
    "word_order": [],
    "transliteration": [],
    "address_abbreviations": [],
    "missing_address_components": [],
    "postal_pin_variation": [],
    "house_number_variation": [],
    "landmark_based": []
}

legal_terms = {"inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited", "pvt", "private", "co", "company", "llp"}
addr_abbrevs = [("rd", "road"), ("st", "street"), ("ave", "avenue"), ("blvd", "boulevard"), ("dr", "drive"), ("fl", "floor"), ("ste", "suite"), ("ln", "lane")]

for p in pairs:
    s1_name = p["s1_name"]
    m_name = p["match_name"]
    s1_addr = p["s1_addr"]
    m_addr = p["match_addr"]
    
    # 1. Punctuation
    s1_clean = re.sub(r'[^\w\s]', '', s1_name).lower()
    m_clean = re.sub(r'[^\w\s]', '', m_name).lower()
    if ('&' in s1_name or '&' in m_name or '-' in s1_name or '-' in m_name) and (s1_name != m_name):
        if len(categories["punctuation"]) < 5:
            categories["punctuation"].append(p)
            
    # 2. Legal suffix
    s1_words = set(s1_clean.split())
    m_words = set(m_clean.split())
    diff = (s1_words ^ m_words) & legal_terms
    if diff and len(categories["legal_suffix"]) < 5:
        categories["legal_suffix"].append(p)
        
    # 3. Word order
    if sorted(s1_words) == sorted(m_words) and s1_words and s1_clean.split() != m_clean.split():
        if len(categories["word_order"]) < 5:
            categories["word_order"].append(p)

    # 4. Typos (small edit distance, not exact)
    if 0 < abs(len(s1_clean) - len(m_clean)) <= 2 and s1_clean != m_clean:
        # Check if words are nearly identical
        if len(s1_words) == len(m_words) and len(s1_words & m_words) >= max(1, len(s1_words)-1):
            if len(categories["typos"]) < 5:
                categories["typos"].append(p)

    # 5. Abbreviations
    if ("corp" in s1_name.lower() and "corporation" in m_name.lower()) or ("pvt" in s1_name.lower() and "private" in m_name.lower()) or ("mfg" in s1_name.lower() or "intl" in s1_name.lower()):
        if len(categories["abbreviations"]) < 5:
            categories["abbreviations"].append(p)

    # 6. Address abbreviations
    s1_addr_lower = s1_addr.lower()
    m_addr_lower = m_addr.lower()
    for ab, full in addr_abbrevs:
        if (re.search(r'\b' + ab + r'\b', s1_addr_lower) and re.search(r'\b' + full + r'\b', m_addr_lower)) or \
           (re.search(r'\b' + full + r'\b', s1_addr_lower) and re.search(r'\b' + ab + r'\b', m_addr_lower)):
            if len(categories["address_abbreviations"]) < 5:
                categories["address_abbreviations"].append(p)
                break

    # 7. Missing address components
    if len(s1_addr) > 0 and len(m_addr) > 0 and (len(s1_addr) > 2 * len(m_addr) or len(m_addr) > 2 * len(s1_addr)):
        if len(categories["missing_address_components"]) < 5:
            categories["missing_address_components"].append(p)

    # 8. Landmark based
    landmarks = ["near", "opp", "opposite", "behind", "beside", "adjacent", "metro station", "bus stand", "temple"]
    if any(re.search(r'\b' + lm + r'\b', s1_addr_lower) or re.search(r'\b' + lm + r'\b', m_addr_lower) for lm in landmarks):
        if len(categories["landmark_based"]) < 5:
            categories["landmark_based"].append(p)

    # 9. Postal/PIN variation
    s1_nums = set(re.findall(r'\b\d{5,6}\b', s1_addr))
    m_nums = set(re.findall(r'\b\d{5,6}\b', m_addr))
    if s1_nums and m_nums and s1_nums != m_nums:
        if len(categories["postal_pin_variation"]) < 5:
            categories["postal_pin_variation"].append(p)

    # 10. House number variation
    s1_h = re.findall(r'#\s*\d+|no\.?\s*\d+|\b\d+[-/]\w+', s1_addr_lower)
    m_h = re.findall(r'#\s*\d+|no\.?\s*\d+|\b\d+[-/]\w+', m_addr_lower)
    if (s1_h or m_h) and (s1_h != m_h) and len(categories["house_number_variation"]) < 5:
        categories["house_number_variation"].append(p)

print("Writing categories...")
with open(r"c:\jaishree_projects\amazon_ml_challenge\noise_categories.json", 'w', encoding='utf-8') as out_f:
    json.dump(categories, out_f, indent=2)

for cat, lst in categories.items():
    print(f"{cat}: {len(lst)} examples found")
