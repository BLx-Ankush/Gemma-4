from dataclasses import dataclass
from typing import Optional
from aegis.core import gemma_core, mirror
from aegis.core.prompts import build_veda_prompt
from . import drug_lookup, voice_utils
import os
import time


@dataclass
class VedaResult:
    transcript: str
    detected_language: str
    query_text: str
    response_text: str
    raw_response: str
    mirror_report: mirror.MirrorReport
    drug_context: str
    audio_response_path: str
    processing_time_ms: int
    image_path: str


LANGUAGE_MAP = {
    "en": "English", "hi": "Hindi", "ta": "Tamil",
    "es": "Spanish", "fr": "French", "sw": "Swahili",
    "ar": "Arabic", "pt": "Portuguese", "zh": "Chinese",
    "bn": "Bengali", "ur": "Urdu", "te": "Telugu",
    "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
}

SAFE_FALLBACK_MESSAGES = {
    "English": "I cannot safely process this request right now. Please seek help from a nearby health worker or emergency service immediately.",
    "Hindi": "Main is samay surakshit roop se is prashn ko process nahi kar pa raha hoon. Kripya turant kisi health worker ya emergency seva se sampark karein.",
    "Kannada": "Iga ee vinantiyannu surakshitavagi prakriye maadalu sadyavilla. Dayavittu hatra iruva aroghya karmikarannu athava aapathkaaleena seveyannu takshan samparkisi.",
}


def _safe_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _is_env_enabled(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


# FIX 8: VEDA_MAX_TOKENS raised from 32 to 256.
# 32 tokens = 3-4 words. Medical guidance needs at least 150-250 tokens.
VEDA_MAX_TOKENS = max(64, min(_safe_int_env("AEGIS_VEDA_MAX_TOKENS", 256), 512))

ENABLE_TEXT_TTS = _is_env_enabled("AEGIS_ENABLE_TEXT_TTS", False)
ENABLE_VOICE_TTS = _is_env_enabled("AEGIS_ENABLE_VOICE_TTS", True)
TTS_MAX_CHARS = max(120, _safe_int_env("AEGIS_TTS_MAX_CHARS", 500))


def _safe_fallback_message(language: str) -> str:
    return SAFE_FALLBACK_MESSAGES.get(language, SAFE_FALLBACK_MESSAGES["English"])


def build_system_prompt(language: str = "English", drug_context: str = "") -> str:
    return build_veda_prompt(language=language, drug_context=drug_context)


def _should_generate_tts(audio_path: Optional[str]) -> bool:
    return ENABLE_VOICE_TTS if audio_path else ENABLE_TEXT_TTS


def _trim_for_tts(text: str) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= TTS_MAX_CHARS:
        return cleaned
    return cleaned[:TTS_MAX_CHARS].rstrip() + "..."


def process_medical_query(
    image_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    text_query: Optional[str] = None,
    language: str = "English",
    session_id: Optional[str] = None,
) -> VedaResult:
    start_time = time.time()

    if not (audio_path or text_query):
        raise ValueError("Either audio_path or text_query must be provided")

    transcript = ""
    detected_language = language
    query_text = text_query or ""
    drug_context = ""
    raw_response = ""
    model_latency_ms = 0

    try:
        if audio_path and os.path.exists(audio_path):
            transcription = voice_utils.transcribe(audio_path)
            transcript = transcription.text
            query_text = transcript
            detected_language = LANGUAGE_MAP.get(transcription.language, transcription.language)

        drug_context = drug_lookup.get_drug_context(query_text)
        system_prompt = build_system_prompt(detected_language, drug_context)

        if image_path:
            user_message = f"[I am showing you an image: {image_path}]\n\n{query_text}"
        else:
            user_message = query_text

        infer_result = gemma_core.infer(
            system_prompt,
            user_message,
            image_path,
            max_tokens=VEDA_MAX_TOKENS,
        )
        raw_response = infer_result.response
        model_latency_ms = infer_result.latency_ms

        # FIX 9: MIRROR always runs. Removed the SYNC_MIRROR_MAX_MODEL_MS deferral.
        # This is the core guarantee of AEGIS. No exceptions.
        image_context = f"Image: {image_path}" if image_path else "No image provided"
        mirror_report = mirror.audit(
            query=query_text,
            response=raw_response,
            image_context=image_context
        )
        response_text = mirror.apply_verdict(raw_response, mirror_report, detected_language)

    except Exception as exc:
        response_text = _safe_fallback_message(detected_language)
        raw_response = response_text
        mirror_report = mirror.MirrorReport(
            confidence_score=0,
            flags=["system_unavailable"],
            reasoning_trace=str(exc),
            verdict="warn",
        )

    audio_response_path = ""
    if _should_generate_tts(audio_path):
        try:
            audio_response_path = voice_utils.speak_to_file(_trim_for_tts(response_text))
        except Exception:
            audio_response_path = ""

    processing_time_ms = int((time.time() - start_time) * 1000)
    return VedaResult(
        transcript=transcript,
        detected_language=detected_language,
        query_text=query_text,
        response_text=response_text,
        raw_response=raw_response,
        mirror_report=mirror_report,
        drug_context=drug_context,
        audio_response_path=audio_response_path,
        processing_time_ms=processing_time_ms,
        image_path=image_path or "",
    )


def process_text_only(text_query: str, language: str = "English") -> VedaResult:
    return process_medical_query(
        image_path=None, audio_path=None,
        text_query=text_query, language=language, session_id=None,
    )
