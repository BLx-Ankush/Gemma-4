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


def test_process_text_only_test_query_returns_clean_ready_message(monkeypatch):
    monkeypatch.setattr(veda, "ENABLE_TEXT_TTS", False)
    monkeypatch.setattr(veda.drug_lookup, "get_drug_context", lambda _q: "")
    monkeypatch.setattr(
        veda.drug_lookup,
        "get_drug_context_metadata",
        lambda _q: {
            "query": _q,
            "has_context": False,
            "match_count": 0,
            "matches": [],
            "sources": [],
            "live_lookup_enabled": False,
            "live_fallback_used": False,
        },
    )

    def _infer_should_not_run(*_args, **_kwargs):
        raise AssertionError("Primary infer should not run for non-medical test queries")

    monkeypatch.setattr(veda.gemma_core, "infer", _infer_should_not_run)
    monkeypatch.setattr(
        veda.mirror,
        "audit",
        lambda **_kwargs: mirror.MirrorReport(confidence_score=90, flags=[], reasoning_trace="ok", verdict="pass"),
    )
    monkeypatch.setattr(veda.mirror, "apply_verdict", lambda original_response, *_args, **_kwargs: original_response)

    result = veda.process_text_only("hello hello mike testing 123")

    assert result.response_text.startswith("I hear you loud and clear.")
    assert "the user isn't" not in result.response_text.lower()
    assert "cause: n/a" not in result.response_text.lower()
    assert "warning: n/a" not in result.response_text.lower()


def test_process_text_only_audio_uses_exact_final_response_text(monkeypatch):
    monkeypatch.setattr(veda, "ENABLE_TEXT_TTS", True)
    monkeypatch.setattr(veda.drug_lookup, "get_drug_context", lambda _q: "")
    monkeypatch.setattr(
        veda.drug_lookup,
        "get_drug_context_metadata",
        lambda _q: {
            "query": _q,
            "has_context": False,
            "match_count": 0,
            "matches": [],
            "sources": [],
            "live_lookup_enabled": False,
            "live_fallback_used": False,
        },
    )
    monkeypatch.setattr(
        veda.gemma_core,
        "infer",
        lambda *_args, **_kwargs: gemma_core.InferResult(
            response="Use cool water and keep the area clean.",
            model_used="cloud-model",
            mode="online",
            latency_ms=42,
            has_vision=False,
        ),
    )
    monkeypatch.setattr(
        veda.mirror,
        "audit",
        lambda **_kwargs: mirror.MirrorReport(confidence_score=88, flags=[], reasoning_trace="ok", verdict="pass"),
    )
    monkeypatch.setattr(veda.mirror, "apply_verdict", lambda original_response, *_args, **_kwargs: original_response)

    captured = {}

    def _capture_tts(text: str) -> str:
        captured["text"] = text
        return "tmp/reply.wav"

    monkeypatch.setattr(veda.voice_utils, "speak_to_file", _capture_tts)

    result = veda.process_text_only("I burned my hand")

    assert result.audio_response_path == "tmp/reply.wav"
    assert captured["text"] == result.response_text


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


def test_sanitize_model_response_strips_acknowledge_steps_scaffold_lines():
    raw = (
        "*   *Acknowledge:* It is worrying when your child has fever.\n"
        "*   *Steps:*\n"
        "1. Keep your child hydrated.\n"
        "2. Use lukewarm sponging if needed.\n"
        "3. Seek urgent care if breathing trouble or seizures occur."
    )

    cleaned = veda._sanitize_model_response(raw)

    assert "Acknowledge:" not in cleaned
    assert "*Steps:*" not in cleaned
    assert cleaned.startswith("1. Keep your child hydrated")


def test_sanitize_model_response_strips_numbered_rubric_lines():
    raw = (
        "1. Acknowledge experience (1 short sentence).\n"
        "2. 3-5 clear, numbered first-aid steps.\n"
        "3. Most likely cause if clear.\n"
        "4. One warning sign.\n"
        "5. Professional care at the end.\n"
        "\n"
        "It sounds like you may have an infection.\n"
        "1. Drink fluids and rest."
    )

    cleaned = veda._sanitize_model_response(raw)

    assert "3-5 clear" not in cleaned
    assert "Most likely cause" not in cleaned
    assert cleaned.startswith("It sounds like you may have an infection")


def test_sanitize_model_response_strips_mirror_planning_leak_patterns():
    raw = (
        '* But applying a medical structure to a "test" message is nonsensical.\n'
        "2. Steps: (How to use me?) 1. Describe the injury. 2. Upload a photo.\n"
        "3. Cause: This appears to be a system test.\n"
        "2. Explain how to get the best help from me (the \"steps\").\n"
        '* No analysis/planning.\n'
        '* No labels.\n'
        '* *Revised Plan:*\n'
        '* "Hello! I am AEGIS, your first-aid assistant. Please tell me your symptoms and I will provide immediate steps."\n'
    )

    cleaned = veda._sanitize_model_response(raw)

    assert "medical structure" not in cleaned.lower()
    assert "how to use me" not in cleaned.lower()
    assert "revised plan" not in cleaned.lower()
    assert "no analysis/planning" not in cleaned.lower()
    assert "no labels" not in cleaned.lower()
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


def test_process_text_only_fails_closed_when_mirror_audit_errors(monkeypatch):
    monkeypatch.setattr(veda, "ENABLE_TEXT_TTS", False)
    monkeypatch.setattr(veda.drug_lookup, "get_drug_context", lambda _q: "")
    monkeypatch.setattr(
        veda.gemma_core,
        "infer",
        lambda *_args, **_kwargs: gemma_core.InferResult(
            response="Drink water and monitor symptoms.",
            model_used="cloud-model",
            mode="online",
            latency_ms=55,
            has_vision=False,
        ),
    )
    def _raise_mirror_timeout(**_kwargs):
        raise RuntimeError("mirror timeout")

    monkeypatch.setattr(
        veda.mirror,
        "audit",
        _raise_mirror_timeout,
    )

    result = veda.process_text_only("I feel dizzy")

    assert result.mirror_report.verdict == "block"
    assert "system_unavailable" in result.mirror_report.flags
    assert result.response_text.startswith("I cannot safely process this request")


def test_process_text_only_uses_offline_mirror_token_budget(monkeypatch):
    monkeypatch.setattr(veda, "ENABLE_TEXT_TTS", False)
    monkeypatch.setattr(veda.drug_lookup, "get_drug_context", lambda _q: "")
    monkeypatch.setattr(
        veda.drug_lookup,
        "get_drug_context_metadata",
        lambda _q: {
            "query": _q,
            "has_context": False,
            "match_count": 0,
            "matches": [],
            "sources": [],
            "live_lookup_enabled": False,
            "live_fallback_used": False,
        },
    )
    monkeypatch.setattr(veda.gemma_core.compute_router, "get_status", lambda: {"mode": "offline"})
    monkeypatch.setattr(
        veda.gemma_core,
        "infer",
        lambda *_args, **_kwargs: gemma_core.InferResult(
            response="Use cool running water for 20 minutes.",
            model_used="local-model",
            mode="offline",
            latency_ms=120,
            has_vision=False,
        ),
    )

    captured = {"max_tokens": None}

    def _capture_audit(**kwargs):
        captured["max_tokens"] = kwargs.get("max_tokens")
        return mirror.MirrorReport(confidence_score=90, flags=[], reasoning_trace="ok", verdict="pass")

    monkeypatch.setattr(veda.mirror, "audit", _capture_audit)
    monkeypatch.setattr(veda.mirror, "apply_verdict", lambda original_response, *_args, **_kwargs: original_response)

    result = veda.process_text_only("I burned my hand")

    assert result.mirror_report.verdict == "pass"
    assert captured["max_tokens"] == veda.MIRROR_AUDIT_MAX_TOKENS_OFFLINE
