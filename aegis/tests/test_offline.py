from aegis.core import mirror
from aegis.modules import veda


def test_offline_path_skips_cloud(monkeypatch):
    from aegis.core import gemma_core

    monkeypatch.setattr(gemma_core.compute_router, "get_status", lambda: {"mode": "offline", "is_online": False, "ram_gb": 4.0})

    cloud_called = {"value": False}

    def _cloud(*_args, **_kwargs):
        cloud_called["value"] = True
        return "cloud"

    monkeypatch.setattr(gemma_core, "_infer_cloud", _cloud)
    monkeypatch.setattr(gemma_core, "_infer_local", lambda *_args, **_kwargs: "local")

    result = gemma_core.infer("sys", "user")

    assert result.response == "local"
    assert cloud_called["value"] is False


def test_veda_returns_safe_fallback_on_pipeline_error(monkeypatch):
    monkeypatch.setattr(veda.gemma_core, "infer", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("engine down")))
    monkeypatch.setattr(veda.voice_utils, "speak_to_file", lambda _text: "tmp/fallback.wav")

    result = veda.process_text_only("My child has trouble breathing", language="English")

    assert result.audio_response_path == "tmp/fallback.wav"
    assert result.mirror_report.verdict == "warn"
    assert "system_unavailable" in result.mirror_report.flags
    assert len(result.response_text) > 0


def test_offline_mirror_block_still_works():
    block_report = mirror.MirrorReport(confidence_score=5, flags=["potential_harm"], reasoning_trace="high risk", verdict="block")
    safe = mirror.apply_verdict("unsafe", block_report)

    assert "not confident enough" in safe
