"""Extract phishing-related URL, page-content, and domain features."""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Mapping
from urllib.parse import parse_qs, urlparse


FEATURE_NAMES = [
    "having_IP_Address",
    "URL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "SSLfinal_State",
    "Domain_registeration_length",
    "Favicon",
    "port",
    "HTTPS_token",
    "Request_URL",
    "URL_of_Anchor",
    "Links_in_tags",
    "SFH",
    "Submitting_to_email",
    "Abnormal_URL",
    "Redirect",
    "on_mouseover",
    "RightClick",
    "popUpWidnow",
    "Iframe",
    "age_of_domain",
    "DNSRecord",
    "web_traffic",
    "Page_Rank",
    "Google_Index",
    "Links_pointing_to_page",
    "Statistical_report",
]

_SHORTENERS = {
    "bit.ly",
    "goo.gl",
    "ow.ly",
    "tinyurl.com",
    "t.co",
    "is.gd",
    "buff.ly",
    "cutt.ly",
    "rebrand.ly",
}
_SUSPICIOUS_TLDS = {".zip", ".review", ".country", ".kim", ".top", ".xyz"}


def _binary(condition: bool) -> int:
    return 1 if condition else -1


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _domain_depth(host: str) -> int:
    return max(0, len([part for part in host.split(".") if part]) - 2)


def _external_ratio(html: str, base_host: str, tag: str, attribute: str) -> float:
    values = re.findall(
        rf"<{tag}\b[^>]*\b{attribute}\s*=\s*[\"']([^\"']+)",
        html,
        flags=re.IGNORECASE,
    )
    if not values:
        return 0.0
    external = 0
    for value in values:
        parsed = urlparse(value if "://" in value else "https://" + base_host)
        if parsed.netloc and parsed.netloc.lower() != base_host:
            external += 1
    return external / len(values)


def extract_features(url: str, html: str | None = None) -> dict[str, int | float]:
    """Return features matching the UCI dataset's feature schema.

    ``html`` is optional. Content-derived features use conservative defaults
    when page source is unavailable, which keeps API prediction lightweight.
    """
    normalized_url = url.strip()
    parsed = urlparse(
        normalized_url if re.match(r"^[a-z][a-z0-9+.-]*://", normalized_url, re.I)
        else "https://" + normalized_url
    )
    host = (parsed.hostname or "").lower()
    path_and_query = (parsed.path or "") + ("?" + parsed.query if parsed.query else "")
    page = html or ""
    lower_page = page.lower()
    query_params = parse_qs(parsed.query)
    suspicious_tld = any(host.endswith(tld) for tld in _SUSPICIOUS_TLDS)
    html_links = re.findall(r"<a\b[^>]*\bhref\s*=", lower_page)

    features: dict[str, int | float] = {
        "having_IP_Address": _binary(_is_ip(host)),
        "URL_Length": _binary(len(normalized_url) < 54),
        "Shortining_Service": _binary(host in _SHORTENERS),
        "having_At_Symbol": _binary("@" in normalized_url),
        "double_slash_redirecting": _binary("//" in path_and_query),
        "Prefix_Suffix": _binary("-" in host),
        "having_Sub_Domain": _binary(_domain_depth(host) == 0),
        "SSLfinal_State": _binary(parsed.scheme == "https" and not suspicious_tld),
        "Domain_registeration_length": 1,
        "Favicon": _binary(bool(re.search(r"<link[^>]+(?:favicon|icon)", lower_page))),
        "port": _binary(parsed.port is None or parsed.port in {80, 443}),
        "HTTPS_token": _binary("https" not in host),
        "Request_URL": _binary(_external_ratio(page, host, "img", "src") < 0.22),
        "URL_of_Anchor": _binary(_external_ratio(page, host, "a", "href") < 0.31),
        "Links_in_tags": _binary(_external_ratio(page, host, "link", "href") < 0.17),
        "SFH": _binary(not re.search(r"<form\b[^>]*\baction\s*=\s*[\"']https?://", lower_page)),
        "Submitting_to_email": _binary("mailto:" not in lower_page),
        "Abnormal_URL": _binary(not suspicious_tld and bool(host)),
        "Redirect": _binary("window.location" not in lower_page and "http-equiv=\"refresh" not in lower_page),
        "on_mouseover": _binary("onmouseover" not in lower_page),
        "RightClick": _binary("event.button==2" not in lower_page.replace(" ", "")),
        "popUpWidnow": _binary("window.open" not in lower_page),
        "Iframe": _binary("<iframe" not in lower_page),
        "age_of_domain": 1,
        "DNSRecord": _binary(bool(host)),
        "web_traffic": 1,
        "Page_Rank": 1,
        "Google_Index": 1,
        "Links_pointing_to_page": _binary(len(html_links) < 50),
        "Statistical_report": _binary(not any(token in normalized_url.lower() for token in ("login", "verify", "secure", "account")) and not query_params),
    }
    return {name: features[name] for name in FEATURE_NAMES}


def feature_vector(url: str, html: str | None = None) -> list[int | float]:
    """Return features in the stable order expected by the trained model."""
    features = extract_features(url, html)
    return [features[name] for name in FEATURE_NAMES]


def features_from_mapping(values: Mapping[str, int | float]) -> list[int | float]:
    """Convert a named feature mapping into the stable model input order."""
    return [values[name] for name in FEATURE_NAMES]