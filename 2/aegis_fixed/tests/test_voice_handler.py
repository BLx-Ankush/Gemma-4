from aegis.core import mirror
from aegis.modules import voice_handler, veda


def test_create_and_clear_session():
    session_id = voice_handler.create_session()

    session = voice_handler.get_session(session_id)
    assert session is not None
    assert session.session_id == session_id

    cleared = voice_handler.clear_session(session_id)
    assert cleared is True
    assert voice_handler.get_session(session_id) is None


def test_process_voice_turn_updates_history(monkeypatch):
    report = mirror.MirrorReport(confidence_score=75, flags=["potential_harm"], reasoning_trace="watchful", verdict="warn")
    fake_result = veda.VedaResult(
        transcript="my child has fever",
        detected_language="English",
        query_text="my child has fever",
        response_text="Keep the child hydrated and monitor temperature.",
        raw_response="Keep the child hydrated and monitor temperature.",
        mirror_report=report,
        drug_context="",
        audio_response_path="tmp/reply.wav",
        processing_time_ms=123,
        image_path="",
    )
    monkeypatch.setattr(voice_handler.veda, "process_medical_query", lambda **_kwargs: fake_result)

    payload = voice_handler.process_voice_turn(audio_path="dummy.wav")

    assert "session_id" in payload
    assert payload["transcript"] == "my child has fever"
    assert payload["mirror"]["risk_tags"] == ["potential_harm"]
    assert payload["mirror"]["notes"] == "watchful"
    assert payload["turn_count"] == 1

    history = voice_handler.get_session_history(payload["session_id"])
    assert len(history) == 1
    assert history[0]["assistant_text"].startswith("Keep the child hydrated")
