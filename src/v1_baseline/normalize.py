"""Text normalization utilities for business names and addresses.

All logic here is purely algorithmic / rule based and derived only from the
domain knowledge described in the problem statement (no external lookups).
"""
import re
import unicodedata

# --- legal-suffix / abbreviation expansion (name) ---------------------------
NAME_ABBREV = {
    "corp": "corporation", "corporation": "corporation",
    "inc": "incorporated", "incorporated": "incorporated",
    "ltd": "limited", "limited": "limited",
    "llc": "llc", "llp": "llp",
    "co": "company", "company": "company",
    "pvt": "private", "private": "private",
    "plc": "plc",
    "intl": "international", "international": "international",
    "assoc": "associates", "associates": "associates",
    "bros": "brothers", "brothers": "brothers",
    "mfg": "manufacturing", "manufacturing": "manufacturing",
    "grp": "group", "group": "group",
    "svcs": "services", "svc": "service", "services": "services",
    "dept": "department",
    "&": "and",
}
# tokens that carry little discriminative power for blocking purposes
NAME_STOP = {
    "the", "a", "an", "and", "of", "for", "at", "in", "on", "&",
    "corporation", "incorporated", "limited", "llc", "llp", "company",
    "private", "plc", "international", "associates", "brothers",
    "group", "services", "service", "co", "corp", "inc", "ltd", "pvt",
}

# --- address abbreviation expansion -----------------------------------------
ADDR_ABBREV = {
    "rd": "road", "road": "road",
    "st": "street", "str": "street", "street": "street",
    "ave": "avenue", "av": "avenue", "avenue": "avenue",
    "dr": "drive", "drive": "drive",
    "ln": "lane", "lane": "lane",
    "blvd": "boulevard", "boulevard": "boulevard",
    "apt": "apartment", "apartment": "apartment",
    "ste": "suite", "suite": "suite",
    "hwy": "highway", "highway": "highway",
    "pkwy": "parkway", "parkway": "parkway",
    "ct": "court", "court": "court",
    "sq": "square", "square": "square",
    "no": "number", "number": "number",
    "flr": "floor", "floor": "floor",
    "bldg": "building", "building": "building",
    "near": "near", "opp": "opposite", "opposite": "opposite",
}
ADDR_STOP = {
    "road", "street", "avenue", "drive", "lane", "boulevard", "apartment",
    "suite", "highway", "parkway", "court", "square", "number", "floor",
    "building", "near", "opposite", "the", "of", "and", "at", "in", "on",
}

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def basic_clean(s):
    """Unicode-normalize, lowercase, replace punctuation with spaces."""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = s.replace("&", " and ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _expand(tokens, mapping):
    return [mapping.get(t, t) for t in tokens]


def name_tokens(s, drop_stop=True):
    s = basic_clean(s)
    toks = s.split(" ") if s else []
    toks = _expand(toks, NAME_ABBREV)
    if drop_stop:
        toks = [t for t in toks if t and t not in NAME_STOP]
    else:
        toks = [t for t in toks if t]
    return toks


def addr_tokens(s, drop_stop=True):
    s = basic_clean(s)
    toks = s.split(" ") if s else []
    toks = _expand(toks, ADDR_ABBREV)
    if drop_stop:
        toks = [t for t in toks if t and t not in ADDR_STOP]
    else:
        toks = [t for t in toks if t]
    return toks


def normalized_name(s):
    """Canonical normalized name string (for ratio-based similarity)."""
    return " ".join(name_tokens(s, drop_stop=False))


def normalized_addr(s):
    return " ".join(addr_tokens(s, drop_stop=False))
