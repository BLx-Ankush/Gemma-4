import requests

from aegis.core import gemma_core


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_extract_chat_text():
    payload = {
        "choices": [
            {
                "message": {
                    "content": "Safe response",
                }
            }
        ]
    }

    assert gemma_core._extract_chat_text(payload) == "Safe response"


def test_local_infer_uses_chat_endpoint(monkeypatch):
    monkeypatch.setenv("AEGIS_LOCAL_LLM_URL", "http://127.0.0.1:8080")

    def _fake_post(url, headers=None, data=None, timeout=None):
        assert url.endswith("/v1/chat/completions")
        return _FakeResponse({"choices": [{"message": {"content": "hello from local"}}]})

    monkeypatch.setattr(gemma_core.requests, "post", _fake_post)

    response = gemma_core._infer_local("sys", "user")

    assert response == "hello from local"


def test_local_infer_falls_back_to_completion(monkeypatch):
    monkeypatch.setenv("AEGIS_LOCAL_LLM_URL", "http://127.0.0.1:8080")

    calls = {"n": 0}

    def _fake_post(url, headers=None, data=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.ConnectionError("chat endpoint down")
        assert url.endswith("/completion")
        return _FakeResponse({"content": "completion fallback"})

    monkeypatch.setattr(gemma_core.requests, "post", _fake_post)

    response = gemma_core._infer_local("sys", "user")

    assert response == "completion fallback"


def test_infer_online_cloud_success(monkeypatch):
    monkeypatch.setattr(gemma_core.compute_router, "get_status", lambda: {"mode": "online", "is_online": True, "ram_gb": 8.0})
    monkeypatch.setattr(gemma_core, "_cloud_is_configured", lambda: True)
    monkeypatch.setattr(gemma_core, "_infer_cloud", lambda *_args, **_kwargs: "cloud ok")
    monkeypatch.setattr(gemma_core, "_infer_local", lambda *_args, **_kwargs: "local")

    assert gemma_core.infer("sys", "user").response == "cloud ok"


def test_infer_online_cloud_fallback_to_local(monkeypatch):
    monkeypatch.setattr(gemma_core.compute_router, "get_status", lambda: {"mode": "online", "is_online": True, "ram_gb": 8.0})
    monkeypatch.setattr(gemma_core, "_cloud_is_configured", lambda: True)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("cloud failed")

    monkeypatch.setattr(gemma_core, "_infer_cloud", _boom)
    monkeypatch.setattr(gemma_core, "_infer_local", lambda *_args, **_kwargs: "local fallback")

    assert gemma_core.infer("sys", "user").response == "local fallback"


def test_load_model_returns_unreachable(monkeypatch):
    def _fake_get(*_args, **_kwargs):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(gemma_core.requests, "get", _fake_get)

    status = gemma_core.load_model()

    assert status["reachable"] is False
