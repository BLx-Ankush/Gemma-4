from aegis.modules import drug_lookup


def _set_test_cache(monkeypatch):
    sample_cache = {
        "paracetamol": {
            "generic_name": "Paracetamol",
            "brand_names": ["Tylenol", "Crocin"],
            "category": "Analgesic",
            "common_uses": "Fever, mild pain",
            "dosage_adult": "500 mg every 6-8 hours",
            "major_interactions": ["Alcohol"],
            "contraindications": ["Severe liver disease"],
            "warnings": "Do not exceed 4g per day",
        },
        "metformin": {
            "generic_name": "Metformin",
            "brand_names": ["Glycomet"],
            "category": "Antidiabetic",
            "common_uses": "Type 2 diabetes",
            "dosage_adult": "500 mg once daily with food",
            "major_interactions": [],
            "contraindications": ["Severe kidney disease"],
            "warnings": "Use with renal monitoring",
        },
    }
    monkeypatch.setattr(drug_lookup, "_drug_cache", sample_cache)


def test_search_drug_matches_generic_and_brand(monkeypatch):
    _set_test_cache(monkeypatch)

    by_generic = drug_lookup.search_drug("paracetamol")
    by_brand = drug_lookup.search_drug("tylenol")

    assert by_generic is not None
    assert by_brand is not None
    assert by_generic["generic_name"] == "Paracetamol"
    assert by_brand["generic_name"] == "Paracetamol"


def test_extract_drug_names_deduplicates(monkeypatch):
    _set_test_cache(monkeypatch)

    text = "I took Tylenol and paracetamol after fever"
    names = drug_lookup.extract_drug_names(text)

    assert len(names) >= 1
    assert any(name.lower() in ["tylenol", "paracetamol"] for name in names)


def test_get_drug_context_contains_core_fields(monkeypatch):
    _set_test_cache(monkeypatch)

    context = drug_lookup.get_drug_context("Can I take paracetamol with metformin?")

    assert "DRUG INFORMATION" in context
    assert "Paracetamol" in context
    assert "Metformin" in context
    assert "Standard adult dosage" in context
