from aegis.app import create_app
from aegis.core import mirror
from aegis.modules import veda


def _client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_health_endpoint(monkeypatch):
    client = _client()

    monkeypatch.setattr("aegis.app.compute_router.refresh", lambda: {"mode": "offline", "ram_gb": 8.0, "is_online": False})
    monkeypatch.setattr("aegis.app.gemma_core.get_model_info", lambda: {"loaded": False, "has_vision": True})

    response = client.get("/api/health")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["compute"]["mode"] == "offline"


def test_query_endpoint_requires_text():
    client = _client()
    response = client.post("/api/query", data={"language": "English"})

    assert response.status_code == 400
    assert response.get_json()["ok"] is False


def test_query_endpoint_success(monkeypatch):
    client = _client()

    fake_report = mirror.MirrorReport(
        confidence_score=85,
        flags=[],
        reasoning_trace="safe",
        verdict="pass",
    )
    fake_result = veda.VedaResult(
        transcript="",
        detected_language="English",
        query_text="I have a mild burn",
        response_text="Cool the burn with running water.",
        raw_response="Cool the burn with running water.",
        mirror_report=fake_report,
        drug_context="",
        audio_response_path="",
        processing_time_ms=50,
        image_path="",
    )

    monkeypatch.setattr("aegis.app.veda.process_medical_query", lambda **_kwargs: fake_result)

    response = client.post("/api/query", data={"text": "I have a mild burn", "language": "English"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["data"]["response_text"].startswith("Cool the burn")
    assert payload["data"]["mirror"]["verdict"] == "pass"
