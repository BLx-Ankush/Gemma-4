from pathlib import Path

import pytest

from aegis.core import gemma_core, mirror
from aegis.modules import veda, voice_utils


def test_build_system_prompt_includes_language_and_drug_context():
    prompt = veda.build_system_prompt(language="Hindi", drug_context="DRUG INFORMATION: Paracetamol")

    assert "Hindi" in prompt
    assert "Devanagari script only" in prompt
    assert "Relevant drug information" in prompt
    assert "Paracetamol" in prompt


def test_process_text_only_pipeline(monkeypatch):
    monkeypatch.setattr(veda, "ENABLE_TEXT_TTS", True)
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
    monkeypatch.setattr(veda, "ENABLE_VOICE_TTS", True)
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


def test_process_text_only_strips_internal_scaffolding(monkeypatch):
    monkeypatch.setattr(veda, "ENABLE_TEXT_TTS", False)
    monkeypatch.setattr(veda.drug_lookup, "get_drug_context", lambda _q: "")

    leaked = (
        '*   User message: "hand burn"\n'
        "*   Persona: AEGIS\n"
        "*   Goal: Provide first-aid guidance.\n"
        "*   Constraints:\n"
        "    1. Acknowledge\n"
        "*   English?: Yes\n\n"
        "It sounds like you have a burn on your hand.\n\n"
        "1. Run cool tap water over the burn for at least 20 minutes.\n"
        "2. Remove rings and tight items before swelling starts.\n"
        "3. Cover loosely with a clean, non-stick dressing."
    )

    monkeypatch.setattr(
        veda.gemma_core,
        "infer",
        lambda *_args, **_kwargs: gemma_core.InferResult(
            response=leaked,
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

    result = veda.process_text_only("hand burn")

    assert "User message:" not in result.response_text
    assert "Persona:" not in result.response_text
    assert "English?: Yes" not in result.response_text
    assert result.response_text.startswith("It sounds like you have a burn on your hand.")


def test_sanitize_model_response_strips_simple_checklist_line():
    raw = (
        "*   Check against \"Output Rules\": No analysis, no labels.\n"
        "*   Language correct? Yes.\n"
        "It sounds like you have a minor hand burn.\n"
        "1. Cool the area under running water for 20 minutes."
    )

    cleaned = veda._sanitize_model_response(raw)

    assert "Check against \"Output Rules\"" not in cleaned
    assert "Language correct? Yes" not in cleaned
    assert cleaned.startswith("It sounds like you have a minor hand burn.")


def test_sanitize_model_response_drops_leading_bullet_rubric_line():
    raw = (
        '*   Check "Specific actions": "20 minutes" included.\n'
        "It sounds like you have a hand burn.\n"
        "1. Cool the burn under running water for 20 minutes."
    )

    cleaned = veda._sanitize_model_response(raw)

    assert "Check \"Specific actions\"" not in cleaned
    assert cleaned.startswith("It sounds like you have a hand burn.")


def test_sanitize_model_response_strips_internal_reasoning_block():
    raw = (
        "* The user hasn't asked for help with a medical issue.\n"
        "* Wait, the prompt says return only final answer.\n"
        "* Refined Plan: acknowledge and invite details.\n"
        "* Final decision: provide user-facing response.\n\n"
        "Hello! I am AEGIS, your trusted first-aid assistant.\n"
        "Please describe your symptoms so I can give immediate steps."
    )

    cleaned = veda._sanitize_model_response(raw)

    assert "The user hasn't" not in cleaned
    assert "Refined Plan" not in cleaned
    assert cleaned.startswith("Hello! I am AEGIS")


def test_adjust_detected_language_uses_transcript_script_for_hindi_urdu_boundary():
    adjusted = veda._adjust_detected_language_from_transcript("Hindi", "مجھے تیز بخار ہے")
    assert adjusted == "Urdu"


def test_reconcile_detected_language_with_response_updates_badge_for_script_mismatch():
    reconciled = veda._reconcile_detected_language_with_response("Hindi", "آپ پانی پیئیں اور آرام کریں۔")
    assert reconciled == "Urdu"


def test_process_text_only_rewrites_when_hindi_response_is_arabic_script(monkeypatch):
    monkeypatch.setattr(veda, "ENABLE_TEXT_TTS", False)
    monkeypatch.setattr(veda.drug_lookup, "get_drug_context", lambda _q: "")
    monkeypatch.setattr(
        veda.mirror,
        "audit",
        lambda **_kwargs: mirror.MirrorReport(confidence_score=88, flags=[], reasoning_trace="ok", verdict="pass"),
    )
    monkeypatch.setattr(veda.mirror, "apply_verdict", lambda original_response, *_args, **_kwargs: original_response)

    calls = {"count": 0}

    def fake_infer(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return gemma_core.InferResult(
                response="آپ پانی پیئیں اور آرام کریں۔",
                model_used="cloud-model",
                mode="online",
                latency_ms=40,
                has_vision=False,
            )
        return gemma_core.InferResult(
            response="आप पानी पिएं और आराम करें।",
            model_used="cloud-model",
            mode="online",
            latency_ms=22,
            has_vision=False,
        )

    monkeypatch.setattr(veda.gemma_core, "infer", fake_infer)

    result = veda.process_text_only("मुझे चक्कर आ रहा है", language="Hindi")

    assert calls["count"] == 2
    assert "आप" in result.response_text
    assert "آ" not in result.response_text
