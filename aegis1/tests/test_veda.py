from pathlib import Path

import pytest

from aegis.core import gemma_core, mirror
from aegis.modules import veda, voice_utils


def test_build_system_prompt_includes_language_and_drug_context():
    prompt = veda.build_system_prompt(language="Hindi", drug_context="DRUG INFORMATION: Paracetamol")

    assert "Hindi" in prompt
    assert "Relevant drug information" in prompt
    assert "Paracetamol" in prompt


def test_process_text_only_pipeline(monkeypatch):
    monkeypatch.setattr(veda.drug_lookup, "get_drug_context", lambda _q: "DRUG INFORMATION: Paracetamol")
    monkeypatch.setattr(
        veda.gemma_core,
        "infer",
        lambda *_args, **_kwargs: gemma_core.InferResult(
            response="Use cool water and rest",
            model_used="cloud-model",
            mode="online",
            latency_ms=42,
            has_vision=False,
        ),
    )
    monkeypatch.setattr(
        veda.mirror,
        "audit",
        lambda **_kwargs: mirror.MirrorReport(confidence_score=80, flags=[], reasoning_trace="ok", verdict="pass"),
    )
    monkeypatch.setattr(veda.mirror, "apply_verdict", lambda original_response, *_args, **_kwargs: original_response)
    monkeypatch.setattr(veda.voice_utils, "speak_to_file", lambda _text: "tmp/reply.wav")

    result = veda.process_text_only("I burned my hand")

    assert result.query_text == "I burned my hand"
    assert result.response_text == "Use cool water and rest"
    assert result.audio_response_path == "tmp/reply.wav"
    assert result.mirror_report.verdict == "pass"


def test_process_medical_query_audio_path(monkeypatch, tmp_path):
    audio_file = tmp_path / "voice.wav"
    audio_file.write_bytes(b"RIFF")

    monkeypatch.setattr(
        veda.voice_utils,
        "transcribe",
        lambda _path: voice_utils.TranscriptionResult(text="help me", language="en", confidence=0.95, duration_ms=100),
    )
    monkeypatch.setattr(veda.drug_lookup, "get_drug_context", lambda _q: "")
    monkeypatch.setattr(
        veda.gemma_core,
        "infer",
        lambda *_args, **_kwargs: gemma_core.InferResult(
            response="Stay calm and seek care",
            model_used="local-model",
            mode="offline",
            latency_ms=37,
            has_vision=False,
        ),
    )
    monkeypatch.setattr(
        veda.mirror,
        "audit",
        lambda **_kwargs: mirror.MirrorReport(confidence_score=70, flags=[], reasoning_trace="ok", verdict="warn"),
    )
    monkeypatch.setattr(
        veda.mirror,
        "apply_verdict",
        lambda original_response, *_args, **_kwargs: original_response + " [verified]",
    )
    monkeypatch.setattr(veda.voice_utils, "speak_to_file", lambda _text: "tmp/audio.wav")

    result = veda.process_medical_query(audio_path=str(audio_file))

    assert result.transcript == "help me"
    assert result.detected_language == "English"
    assert result.response_text.endswith("[verified]")


def test_process_medical_query_raises_when_no_input():
    with pytest.raises(ValueError):
        veda.process_medical_query(audio_path=None, text_query=None)
