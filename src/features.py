# src/features.py
# URL -> 51 URL-only features

import re
import math
from urllib.parse import urlparse
import pandas as pd

URL_ONLY_KEEP = [
    "length_url",
    "length_hostname",
    "ip",
    "nb_dots",
    "nb_hyphens",
    "nb_at",
    "nb_qm",
    "nb_and",
    "nb_or",
    "nb_eq",
    "nb_underscore",
    "nb_tilde",
    "nb_percent",
    "nb_slash",
    "nb_star",
    "nb_colon",
    "nb_comma",
    "nb_semicolumn",
    "nb_dollar",
    "nb_space",
    "nb_www",
    "nb_com",
    "nb_dslash",
    "http_in_path",
    "https_token",
    "ratio_digits_url",
    "ratio_digits_host",
    "punycode",
    "port",
    "tld_in_path",
    "tld_in_subdomain",
    "abnormal_subdomain",
    "nb_subdomains",
    "prefix_suffix",
    "random_domain",
    "shortening_service",
    "path_extension",
    "nb_redirection",
    "nb_external_redirection",
    "length_words_raw",
    "char_repeat",
    "shortest_words_raw",
    "shortest_word_host",
    "shortest_word_path",
    "longest_words_raw",
    "longest_word_host",
    "longest_word_path",
    "avg_words_raw",
    "avg_word_host",
    "avg_word_path",
    "phish_hints",
]

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "buff.ly", "is.gd",
    "cutt.ly", "bit.do", "rebrand.ly", "shorte.st", "adf.ly", "trib.al",
    "tiny.cc", "lnkd.in", "s.id", "rb.gy", "t.ly"
}

PHISH_HINT_WORDS = [
    "login", "signin", "sign-in", "verify", "verification", "secure", "security",
    "account", "update", "confirm", "password", "billing", "invoice", "support", "webscr",
    "redirect", "credential", "reset", "unlock", "limited", "suspend", "alert"
]

PATH_EXTENSIONS = (
    ".php", ".html", ".htm", ".asp", ".aspx", ".jsp", ".cgi", ".exe", ".zip",
    ".rar", ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".js"
)

def _ensure_scheme(raw_url: str) -> str:
    u = raw_url.strip()
    if not u:
        return ""
    if "://" not in u:
        u = "http://" + u
    return u

def _is_ip(host: str) -> int:
    # IPv4
    if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host or ""):
        parts = host.split(".")
        try:
            return int(all(0 <= int(p) <= 255 for p in parts))
        except ValueError:
            return 0
    # very light IPv6 check
    if ":" in (host or "") and re.fullmatch(r"[0-9a-fA-F:]+", host):
        return 1
    return 0

def _count_digits(s: str) -> int:
    return sum(ch.isdigit() for ch in (s or ""))

def _ratio_digits(s: str) -> float:
    s = s or ""
    return _count_digits(s) / len(s) if len(s) > 0 else 0.0

def _tokens_alnum(s: str) -> list[str]:
    # Split into alphanumeric tokens
    return [t for t in re.split(r"[^A-Za-z0-9]+", s or "") if t]

def _min_len(tokens: list[str]) -> int:
    return min((len(t) for t in tokens), default=0)

def _max_len(tokens: list[str]) -> int:
    return max((len(t) for t in tokens), default=0)

def _avg_len(tokens: list[str]) -> float:
    return (sum(len(t) for t in tokens) / len(tokens)) if tokens else 0.0

def _max_consecutive_repeat(s: str) -> int:
    # Maximum run length of any repeating character (e.g. "ccccc" -> 5)
    s = s or ""
    if not s:
        return 0
    best = 1
    run = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best

def _tld_from_host(host: str) -> str:
    host = (host or "").strip(".").lower()
    if not host or _is_ip(host):
        return ""
    parts = host.split(".")
    return parts[-1] if len(parts) >= 2 else ""

def _subdomain_part(host: str) -> str:
    host = (host or "").strip(".").lower()
    if not host or _is_ip(host):
        return ""
    parts = host.split(".")
    if len(parts) <= 2:
        return ""
    return ".".join(parts[:-2])

def _domain_label(host: str) -> str:
    # second-level domain label (example.com -> "example")
    host = (host or "").strip(".").lower()
    if not host or _is_ip(host):
        return ""
    parts = host.split(".")
    return parts[-2] if len(parts) >= 2 else ""

def _entropy(s: str) -> float:
    # Shannon entropy
    s = s or ""
    if not s:
        return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    ent = 0.0
    for c in counts.values():
        p = c / len(s)
        ent -= p * math.log2(p)
    return ent

def _is_random_domain(host: str) -> int:
    # Simple heuristic: high entropy + longer domain label
    label = _domain_label(host)
    if len(label) < 10:
        return 0
    return 1 if _entropy(label) >= 3.5 else 0

def _has_path_extension(path: str) -> int:
    path = (path or "").lower()
    return 1 if any(path.endswith(ext) for ext in PATH_EXTENSIONS) else 0

def _phish_hints_count(url: str) -> int:
    u = (url or "").lower()
    return sum(u.count(w) for w in PHISH_HINT_WORDS)

def extract_features(raw_url: str) -> pd.DataFrame:
    # Returns a 1-row DataFrame with columns exactly in URL_ONLY_KEEP order
    url = _ensure_scheme(raw_url)
    parsed = urlparse(url)

    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    full = url or ""

    # Basic counts (character-level)
    features = {}
    features["length_url"] = len(full)
    features["length_hostname"] = len(host)

    features["ip"] = _is_ip(host)

    features["nb_dots"] = full.count(".")
    features["nb_hyphens"] = full.count("-")
    features["nb_at"] = full.count("@")
    features["nb_qm"] = full.count("?")
    features["nb_and"] = full.count("&")
    features["nb_or"] = full.lower().count("or")
    features["nb_eq"] = full.count("=")
    features["nb_underscore"] = full.count("_")
    features["nb_tilde"] = full.count("~")
    features["nb_percent"] = full.count("%")
    features["nb_slash"] = full.count("/")
    features["nb_star"] = full.count("*")
    features["nb_colon"] = full.count(":")
    features["nb_comma"] = full.count(",")
    features["nb_semicolumn"] = full.count(";")
    features["nb_dollar"] = full.count("$")
    features["nb_space"] = full.count(" ")

    features["nb_www"] = full.lower().count("www")
    features["nb_com"] = full.lower().count(".com")

    # double-slash occurrences
    features["nb_dslash"] = full.count("//")

    # Tokens
    features["http_in_path"] = 1 if "http" in (path.lower()) else 0
    # "https" token appearing in URL beyond scheme (common phishing trick like https-login.example.com)
    scheme = parsed.scheme.lower() if parsed.scheme else ""
    remainder = full.lower()
    if scheme in ("http", "https") and remainder.startswith(scheme + "://"):
        remainder = remainder[len(scheme + "://"):]
    features["https_token"] = 1 if "https" in remainder else 0

    features["ratio_digits_url"] = _ratio_digits(full)
    features["ratio_digits_host"] = _ratio_digits(host)

    features["punycode"] = 1 if "xn--" in host else 0
    features["port"] = 1 if parsed.port is not None else 0

    tld = _tld_from_host(host)
    subdomain = _subdomain_part(host)

    features["tld_in_path"] = 1 if (tld and tld in path.lower()) else 0
    features["tld_in_subdomain"] = 1 if (tld and tld in subdomain) else 0

    # "abnormal_subdomain" heuristic: scheme-like tokens inside hostname/subdomain
    features["abnormal_subdomain"] = 1 if ("http" in subdomain or "https" in subdomain or "//" in host) else 0

    # number of subdomain labels
    if host and not _is_ip(host):
        parts = host.strip(".").split(".")
        features["nb_subdomains"] = max(len(parts) - 2, 0)
    else:
        features["nb_subdomains"] = 0

    # prefix-suffix: hyphen within domain label (common dataset definition)
    features["prefix_suffix"] = 1 if "-" in _domain_label(host) else 0

    features["random_domain"] = _is_random_domain(host)

    features["shortening_service"] = 1 if host in SHORTENERS else 0

    features["path_extension"] = _has_path_extension(path)

    # redirection heuristics: count occurrences of "http" inside path/query (beyond the scheme)
    tail = (path + "?" + query).lower()
    features["nb_redirection"] = tail.count("http")
    # external redirection: presence of "http" in tail suggests redirecting to another domain
    features["nb_external_redirection"] = 1 if "http" in tail else 0

    # Word-length stats
    raw_tokens = _tokens_alnum(full)
    host_tokens = _tokens_alnum(host)
    path_tokens = _tokens_alnum(path)

    features["length_words_raw"] = len(raw_tokens)
    features["char_repeat"] = _max_consecutive_repeat(full)

    features["shortest_words_raw"] = _min_len(raw_tokens)
    features["shortest_word_host"] = _min_len(host_tokens)
    features["shortest_word_path"] = _min_len(path_tokens)

    features["longest_words_raw"] = _max_len(raw_tokens)
    features["longest_word_host"] = _max_len(host_tokens)
    features["longest_word_path"] = _max_len(path_tokens)

    features["avg_words_raw"] = _avg_len(raw_tokens)
    features["avg_word_host"] = _avg_len(host_tokens)
    features["avg_word_path"] = _avg_len(path_tokens)

    features["phish_hints"] = _phish_hints_count(full)

    # Return as DataFrame with correct column order
    row = {col: features.get(col, 0) for col in URL_ONLY_KEEP}
    return pd.DataFrame([row], columns=URL_ONLY_KEEP)

if __name__ == "__main__":
    # Test
    test_url = "https://secure-login.example.com/account/verify?redirect=http://evil.com"
    df = extract_features(test_url)
    print(df.head(1).T)