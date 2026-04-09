import json
import os
import re
from typing import Optional, Dict, List
import requests

from aegis.core import compute_router


DRUG_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "drug_cache.json")

_drug_cache = None
_live_cache: Dict[str, Dict] = {}


def _live_lookup_enabled() -> bool:
    raw = os.environ.get("AEGIS_ENABLE_LIVE_DRUG_API", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _should_use_live_lookup() -> bool:
    if not _live_lookup_enabled():
        return False

    status = compute_router.refresh()
    return bool(status.get("is_online"))


def load_cache():
    global _drug_cache
    if _drug_cache is not None:
        return

    with open(DRUG_CACHE_PATH, "r", encoding="utf-8") as file_handle:
        _drug_cache = json.load(file_handle)

    print(f"Drug cache loaded: {len(_drug_cache)} medicines")


def search_drug(query: str) -> Optional[Dict]:
    load_cache()
    normalized_query = query.strip().lower()
    if not normalized_query:
        return None

    for medicine in _drug_cache.values():
        generic_name = medicine.get("generic_name", "").lower()
        if normalized_query in generic_name:
            return medicine

        for brand_name in medicine.get("brand_names", []):
            if normalized_query in brand_name.lower():
                return medicine

    if _should_use_live_lookup():
        live_result = _search_drug_live(normalized_query)
        if live_result is not None:
            return live_result

    return None


def _extract_text_field(value: Optional[List[str]]) -> str:
    if not value:
        return "Unknown"
    first = value[0] if isinstance(value, list) and value else ""
    if isinstance(first, str) and first.strip():
        return first.strip().replace("\n", " ")
    return "Unknown"


def _extract_list_field(value: Optional[List[str]], limit: int = 5) -> List[str]:
    if not value:
        return []
    cleaned: List[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        token = item.strip().replace("\n", " ")
        if token:
            cleaned.append(token)
        if len(cleaned) >= limit:
            break
    return cleaned


def _normalize_openfda_result(result: Dict, query: str) -> Dict:
    openfda = result.get("openfda", {}) if isinstance(result, dict) else {}
    generic_names = _extract_list_field(openfda.get("generic_name"), limit=3)
    brand_names = _extract_list_field(openfda.get("brand_name"), limit=5)
    categories = _extract_list_field(openfda.get("pharm_class_epc"), limit=3)

    interactions = _extract_list_field(result.get("drug_interactions"), limit=3)
    contraindications = _extract_list_field(result.get("contraindications"), limit=3)

    return {
        "generic_name": generic_names[0] if generic_names else query.title(),
        "brand_names": brand_names,
        "category": categories[0] if categories else "Unknown",
        "common_uses": _extract_text_field(result.get("indications_and_usage")),
        "dosage_adult": _extract_text_field(result.get("dosage_and_administration")),
        "major_interactions": interactions,
        "contraindications": contraindications,
        "warnings": _extract_text_field(result.get("warnings")),
        "source": "openfda_live",
    }


def _search_drug_live(query: str) -> Optional[Dict]:
    if query in _live_cache:
        return _live_cache[query]

    base_url = os.environ.get("AEGIS_OPENFDA_URL", "https://api.fda.gov/drug/label.json").strip()
    timeout_raw = os.environ.get("AEGIS_OPENFDA_TIMEOUT_SEC", "4")
    try:
        timeout = float(timeout_raw)
    except ValueError:
        timeout = 4.0

    for field in ["openfda.generic_name", "openfda.brand_name"]:
        params = {
            "search": f'{field}:"{query}"',
            "limit": 1,
        }
        try:
            response = requests.get(base_url, params=params, timeout=max(1.0, timeout))
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results", [])
            if not results:
                continue

            normalized = _normalize_openfda_result(results[0], query)
            _live_cache[query] = normalized
            return normalized
        except Exception:
            continue

    return None


def extract_drug_names(text: str) -> List[str]:
    load_cache()
    lower_text = text.lower()
    matches: List[str] = []

    for medicine in _drug_cache.values():
        generic_name = medicine.get("generic_name", "")
        generic_tokens = [token.strip() for token in re.split(r"[()/]", generic_name) if token.strip()]
        for token in generic_tokens:
            if token.lower() and token.lower() in lower_text:
                matches.append(token)
                break

        for brand in medicine.get("brand_names", []):
            if brand.lower() in lower_text:
                matches.append(brand)
                break

    # Preserve order while deduplicating.
    seen = set()
    unique_matches = []
    for name in matches:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            unique_matches.append(name)

    return unique_matches


def get_drug_context(text: str) -> str:
    found_names = extract_drug_names(text)
    if not found_names and _should_use_live_lookup():
        # If no cached medicine matched, try the full query against live data.
        found_names = [text.strip()] if text and text.strip() else []

    if not found_names:
        return ""

    context_parts = ["DRUG INFORMATION:\n"]
    for name in found_names:
        medicine = search_drug(name)
        if medicine is None:
            continue

        context_parts.append(f"Medicine: {medicine.get('generic_name', 'Unknown')}\n")
        context_parts.append(f"Category: {medicine.get('category', 'Unknown')}\n")
        context_parts.append(f"Common uses: {medicine.get('common_uses', 'Unknown')}\n")
        context_parts.append(f"Standard adult dosage: {medicine.get('dosage_adult', 'Unknown')}\n")

        interactions = medicine.get("major_interactions", [])
        contraindications = medicine.get("contraindications", [])
        context_parts.append(f"Major interactions: {', '.join(interactions) if interactions else 'None listed'}\n")
        context_parts.append(
            f"Contraindications: {', '.join(contraindications) if contraindications else 'None listed'}\n"
        )
        context_parts.append(f"Warnings: {medicine.get('warnings', 'None listed')}\n")
        source = medicine.get("source", "local_cache")
        context_parts.append(f"Source: {source}\n")
        context_parts.append("---\n")

    return "".join(context_parts).strip()


if __name__ == "__main__":
    print("--- Test 1: search_drug('paracetamol') ---")
    print(search_drug("paracetamol"))

    print("\n--- Test 2: search_drug('tylenol') ---")
    print(search_drug("tylenol"))

    print("\n--- Test 3: extract_drug_names(...) ---")
    print(extract_drug_names("I took some ibuprofen and paracetamol for my headache"))

    print("\n--- Test 4: get_drug_context(...) ---")
    print(get_drug_context("Can I take amoxicillin with metformin?"))

    print("\n--- Test 5: search_drug('xyz_fake_drug') ---")
    print(search_drug("xyz_fake_drug"))
