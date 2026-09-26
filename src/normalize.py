import re
import unicodedata

# Common legal business suffixes across US, India, and France
LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'co', 'company', 'llp', 'pllc', 'sarl', 'sasu', 'sas',
    'eurl', 'sa', 'gmbh', 'bv', 'nv', 'assoc', 'association', 'partners',
    'services', 'enterprises', 'holdings', 'group', 'industries', 'international',
    'intl', 'technologies', 'solutions'
}

# Address abbreviations mapping
ADDR_ABBREVS = {
    r'\brd\b': 'road',
    r'\bst\b': 'street',
    r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bfl\b': 'floor',
    r'\bste\b': 'suite',
    r'\bapt\b': 'apartment',
    r'\bpkwy\b': 'parkway',
    r'\bhwy\b': 'highway',
    r'\bbd\b': 'boulevard',
    r'\bstr\b': 'street'
}

def strip_accents(text: str) -> str:
    """Converts accented unicode characters (é, à, ñ, etc.) to ASCII base form."""
    if not text:
        return ""
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def normalize_text(text: str) -> str:
    """Standard text normalization: lowercase, strip accents, clean punctuation."""
    if not text:
        return ""
    text = strip_accents(text).lower()
    text = text.replace('&', ' and ')
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

import anyascii

def has_non_latin(text: str) -> bool:
    """Checks if text contains non-Latin scripts (Devanagari, Tamil, Bengali, etc.)."""
    if not text:
        return False
    return any(ord(c) > 0x024F for c in text if c.isalpha())

def get_transliterated_name(name: str):
    """
    If name contains non-Latin scripts, transliterates to Latin phonetics and normalizes.
    Returns (trans_clean, trans_core, trans_tokens). If already Latin, returns ('', '', []).
    """
    if not has_non_latin(name):
        return "", "", []
    trans_raw = anyascii.anyascii(name)
    trans_clean = normalize_text(trans_raw)
    trans_tokens = trans_clean.split()
    trans_core_tokens = [t for t in trans_tokens if t not in LEGAL_SUFFIXES]
    if not trans_core_tokens:
        trans_core_tokens = trans_tokens
    trans_core = " ".join(trans_core_tokens)
    return trans_clean, trans_core, trans_tokens

def normalize_business_name(name: str):
    """
    Returns (clean_name, core_name, tokens).
    - clean_name: normalized text with punctuation removed
    - core_name: text with leading/trailing legal suffixes removed
    - tokens: list of unique significant tokens
    """
    clean = normalize_text(name)
    tokens = clean.split()
    
    # Filter out legal suffixes from the core name
    core_tokens = [t for t in tokens if t not in LEGAL_SUFFIXES]
    if not core_tokens:
        core_tokens = tokens  # fallback if name was only legal suffixes
        
    core = " ".join(core_tokens)
    return clean, core, tokens

def normalize_address(addr: str):
    """
    Returns (clean_addr, addr_tokens, numeric_tokens).
    """
    if not addr:
        return "", [], []
    clean = normalize_text(addr)
    for pattern, repl in ADDR_ABBREVS.items():
        clean = re.sub(pattern, repl, clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    tokens = clean.split()
    numbers = [t for t in tokens if t.isdigit()]
    return clean, tokens, numbers
