from aegis.app import create_app
from aegis.core import mirror
from aegis.core import gemma_core
from aegis.modules import drug_lookup, veda


def test_infer_uses_cloud_when_online_and_configured(monkeypatch):
    monkeypatch.setattr(gemma_core.compute_router, "get_status", lambda: {"mode": "online", "is_online": True, "ram_gb": 8.0})
    monkeypatch.setattr(gemma_core, "_cloud_is_configured", lambda: True)
    monkeypatch.setattr(gemma_core, "_infer_cloud", lambda *_args, **_kwargs: "cloud-response")

    local_called = {"value": False}

    def _local(*_args, **_kwargs):
        local_called["value"] = True
        return "local-response"

    monkeypatch.setattr(gemma_core, "_infer_local", _local)

    response = gemma_core.infer("system", "user")

    assert response.response == "cloud-response"
    assert local_called["value"] is False


def test_infer_falls_back_to_local_when_cloud_fails(monkeypatch):
    monkeypatch.setattr(gemma_core.compute_router, "get_status", lambda: {"mode": "online", "is_online": True, "ram_gb": 8.0})
    monkeypatch.setattr(gemma_core, "_cloud_is_configured", lambda: True)

    def _raise(*_args, **_kwargs):
        raise RuntimeError("cloud down")

    monkeypatch.setattr(gemma_core, "_infer_cloud", _raise)
    monkeypatch.setattr(gemma_core, "_infer_local", lambda *_args, **_kwargs: "local-fallback")

    response = gemma_core.infer("system", "user")

    assert response.response == "local-fallback"


def test_search_drug_keeps_offline_cache_priority(monkeypatch):
    monkeypatch.setattr(
        drug_lookup,
        "_drug_cache",
        {
            "paracetamol": {
                "generic_name": "Paracetamol",
                "brand_names": ["Tylenol"],
                "category": "Analgesic",
                "common_uses": "Fever",
                "dosage_adult": "500mg",
                "major_interactions": [],
                "contraindications": [],
                "warnings": "None",
            }
        },
    )
    monkeypatch.setattr(drug_lookup, "_should_use_live_lookup", lambda: True)

    live_called = {"value": False}

    def _live_lookup(_query):
        live_called["value"] = True
        return None

    monkeypatch.setattr(drug_lookup, "_search_drug_live", _live_lookup)

    result = drug_lookup.search_drug("tylenol")

    assert result is not None
    assert result["generic_name"] == "Paracetamol"
    assert live_called["value"] is False


def test_search_drug_uses_live_when_not_in_cache(monkeypatch):
    monkeypatch.setattr(drug_lookup, "_drug_cache", {})
    monkeypatch.setattr(drug_lookup, "_should_use_live_lookup", lambda: True)
    monkeypatch.setattr(
        drug_lookup,
        "_search_drug_live",
        lambda _query: {
            "generic_name": "Ibuprofen",
            "brand_names": ["Advil"],
            "category": "NSAID",
            "common_uses": "Pain",
            "dosage_adult": "200mg",
            "major_interactions": [],
            "contraindications": [],
            "warnings": "None",
            "source": "openfda_live",
        },
    )

    result = drug_lookup.search_drug("ibuprofen")

    assert result is not None
    assert result["source"] == "openfda_live"


def test_route_aliases_status_veda_and_mirror_log(monkeypatch):
    app = create_app()
    app.config.update(TESTING=True)
    client = app.test_client()

    monkeypatch.setattr("aegis.app.compute_router.force_refresh", lambda: {"mode": "offline", "ram_gb": 8.0, "is_online": False})
    monkeypatch.setattr("aegis.app.gemma_core.get_model_info", lambda: {"loaded": True, "active_backend": "local"})

    fake_report = mirror.MirrorReport(
        confidence_score=90,
        flags=[],
        reasoning_trace="safe",
        verdict="pass",
    )
    fake_result = veda.VedaResult(
        transcript="",
        detected_language="English",
        query_text="Need first-aid advice",
        response_text="Use clean water and monitor symptoms.",
        raw_response="Use clean water and monitor symptoms.",
        mirror_report=fake_report,
        drug_context="",
        audio_response_path="",
        processing_time_ms=45,
        image_path="",
    )

    monkeypatch.setattr("aegis.app.veda.process_medical_query", lambda **_kwargs: fake_result)
    monkeypatch.setattr("aegis.app.mirror.get_audit_log", lambda limit=20: [{"report": {"verdict": "pass"}, "limit": limit}])

    status_response = client.get("/status")
    veda_response = client.post("/veda", data={"text": "Need first-aid advice", "language": "English"})
    mirror_response = client.get("/mirror/log?limit=5")

    assert status_response.status_code == 200
    assert status_response.get_json()["ok"] is True

    assert veda_response.status_code == 200
    assert veda_response.get_json()["ok"] is True

    mirror_payload = mirror_response.get_json()
    assert mirror_response.status_code == 200
    assert mirror_payload["ok"] is True
    assert mirror_payload["data"][0]["limit"] == 5