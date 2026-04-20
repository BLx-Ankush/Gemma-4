from dataclasses import dataclass
from typing import Dict, Optional
from aegis.core import gemma_core, mirror
from aegis.core.prompts import build_veda_prompt
from . import drug_lookup, voice_utils
import os
import re
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
    drug_metadata: Dict[str, object]
    audio_response_path: str
    model_latency_ms: int
    processing_time_ms: int
    image_path: str


LANGUAGE_MAP = {
    "en": "English", "hi": "Hindi", "ta": "Tamil",
    "es": "Spanish", "fr": "French", "sw": "Swahili",
    "ar": "Arabic", "pt": "Portuguese", "zh": "Chinese",
    "bn": "Bengali", "ur": "Urdu", "te": "Telugu",
    "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
}

LANGUAGE_ALIASES = {
    "english": "English",
    "hindi": "Hindi",
    "kannada": "Kannada",
    "tamil": "Tamil",
    "telugu": "Telugu",
    "bengali": "Bengali",
    "marathi": "Marathi",
    "gujarati": "Gujarati",
    "malayalam": "Malayalam",
    "urdu": "Urdu",
    "arabic": "Arabic",
    "spanish": "Spanish",
    "french": "French",
    "swahili": "Swahili",
    "portuguese": "Portuguese",
    "chinese": "Chinese",
}

SCRIPT_REGEX = {
    "arabic": re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"),
    "devanagari": re.compile(r"[\u0900-\u097F]"),
    "latin": re.compile(r"[A-Za-z]"),
}

INTERNAL_REASONING_HINTS = (
    "the user has",
    "the user hasn't",
    "the user isn't",
    "the user has not",
    "the prompt",
    "system instructions",
    "output rules",
    "self-correction",
    "refined plan",
    "final decision",
    "alternative interpretation",
    "hidden medical request",
    "system check",
    "system test",
    "test message",
    "how to use me",
    "best help from me",
    "no analysis/planning",
    "no labels",
    "applying a medical structure",
    "medical structure",
    "no medical issue",
    "no medical problem",
    "non-existent injury",
    "testing the bot",
    "if the user is just testing",
    "cause: n/a",
    "warning: n/a",
    "let's double-check",
    "that's too robotic",
    "nonsensical",
    "revised plan",
    "refining based on",
    "persona",
    "alternative:",
    "response:",
    "let's think",
    "i should return",
    "i need to follow",
    "i will provide user-facing",
    "wait,",
    "*wait*",
)

SAFE_FALLBACK_MESSAGES = {
    "English": "I cannot safely process this request right now. Please seek help from a nearby health worker or emergency service immediately.",
    "Hindi": "Main is samay surakshit roop se is prashn ko process nahi kar pa raha hoon. Kripya turant kisi health worker ya emergency seva se sampark karein.",
    "Kannada": "Iga ee vinantiyannu surakshitavagi prakriye maadalu sadyavilla. Dayavittu hatra iruva aroghya karmikarannu athava aapathkaaleena seveyannu takshan samparkisi.",
}


NON_MEDICAL_TEST_PATTERN = re.compile(
    r"\b(hello|hi|hey|test|testing|check|checking|ping|mic|microphone|audio|voice|1\s*,?\s*2\s*,?\s*3|123)\b",
    flags=re.IGNORECASE,
)

MEDICAL_SIGNAL_PATTERN = re.compile(
    r"\b(pain|fever|burn|bleed|bleeding|injur|wound|cut|fracture|vomit|nausea|diarrhea|breath|breathing|cough|chest|allerg|rash|headache|dizz|unconscious|faint|seizure|swelling|poison|overdose|pregnan|child|baby|medicine|dose|tablet|sick|symptom)\b",
    flags=re.IGNORECASE,
)

READY_MESSAGE_EN = (
    "I hear you loud and clear. I am AEGIS, your first-aid assistant, and I am ready to provide immediate, practical guidance for any injuries or health concerns. "
    "Please let me know what is happening or describe your symptoms so I can give you actionable steps to take."
)


def _safe_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _is_env_enabled(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


VEDA_MAX_TOKENS = max(96, min(_safe_int_env("AEGIS_VEDA_MAX_TOKENS", 192), 384))
MIRROR_AUDIT_MAX_TOKENS = max(96, min(_safe_int_env("AEGIS_MIRROR_MAX_TOKENS", 192), 384))
MIRROR_AUDIT_MAX_TOKENS_OFFLINE = max(64, min(_safe_int_env("AEGIS_MIRROR_MAX_TOKENS_OFFLINE", 96), 256))

ENABLE_TEXT_TTS = _is_env_enabled("AEGIS_ENABLE_TEXT_TTS", True)
ENABLE_VOICE_TTS = _is_env_enabled("AEGIS_ENABLE_VOICE_TTS", True)
TTS_MAX_CHARS = max(120, _safe_int_env("AEGIS_TTS_MAX_CHARS", 500))
ENABLE_RESPONSE_LANGUAGE_REWRITE = _is_env_enabled("AEGIS_ENABLE_RESPONSE_LANGUAGE_REWRITE", True)
ENABLE_VOICE_LANGUAGE_REWRITE = _is_env_enabled("AEGIS_ENABLE_VOICE_LANGUAGE_REWRITE", True)
LANGUAGE_REWRITE_MAX_PRIMARY_LATENCY_MS = max(
    0,
    _safe_int_env("AEGIS_LANGUAGE_REWRITE_MAX_PRIMARY_LATENCY_MS", 18000),
)


def _safe_fallback_message(language: str) -> str:
    return SAFE_FALLBACK_MESSAGES.get(language, SAFE_FALLBACK_MESSAGES["English"])


def _looks_like_non_medical_test_query(query_text: str) -> bool:
    text = (query_text or "").strip().lower()
    if not text:
        return False
    if MEDICAL_SIGNAL_PATTERN.search(text):
        return False

    tokens = re.findall(r"[a-z0-9]+", text)
    if len(tokens) > 24:
        return False

    return bool(NON_MEDICAL_TEST_PATTERN.search(text))


def _normalize_language_name(language: str) -> str:
    raw = (language or "").strip()
    if not raw:
        return "English"

    lowered = raw.lower()
    if lowered in LANGUAGE_MAP:
        return LANGUAGE_MAP[lowered]
    if lowered in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[lowered]
    return raw


def _script_counts(text: str) -> dict:
    sample = text or ""
    return {
        "arabic": len(SCRIPT_REGEX["arabic"].findall(sample)),
        "devanagari": len(SCRIPT_REGEX["devanagari"].findall(sample)),
        "latin": len(SCRIPT_REGEX["latin"].findall(sample)),
    }


def _adjust_detected_language_from_transcript(language: str, transcript: str) -> str:
    normalized = _normalize_language_name(language)
    counts = _script_counts(transcript)

    # Whisper can sometimes classify Urdu/Hindustani speech as Hindi while transcript is Arabic-script.
    if normalized == "Hindi" and counts["arabic"] >= 4 and counts["arabic"] > counts["devanagari"]:
        return "Urdu"

    return normalized


def _has_language_script_mismatch(text: str, language: str) -> bool:
    normalized = _normalize_language_name(language)
    counts = _script_counts(text)

    if normalized == "Hindi":
        return counts["arabic"] >= max(6, counts["devanagari"] * 2)
    if normalized in {"Urdu", "Arabic"}:
        return counts["devanagari"] >= max(6, counts["arabic"] * 2)
    return False


def _reconcile_detected_language_with_response(language: str, response_text: str) -> str:
    normalized = _normalize_language_name(language)
    counts = _script_counts(response_text)

    if normalized == "Hindi" and counts["arabic"] >= 6 and counts["arabic"] > counts["devanagari"]:
        return "Urdu"
    if normalized in {"Urdu", "Arabic"} and counts["devanagari"] >= 6 and counts["devanagari"] > counts["arabic"]:
        return "Hindi"
    return normalized


def _rewrite_response_in_language(response_text: str, language: str) -> str:
    target_language = _normalize_language_name(language)
    rewrite_system_prompt = (
        "You are AEGIS translation guard. Rewrite the guidance below in "
        f"{target_language}. Keep all medical meaning, dosage numbers, and step order unchanged. "
        "Return only the rewritten user-facing answer."
    )

    if target_language == "Hindi":
        rewrite_system_prompt += " Use Devanagari script only."
    elif target_language == "Urdu":
        rewrite_system_prompt += " Use Urdu (Perso-Arabic) script."
    elif target_language == "Arabic":
        rewrite_system_prompt += " Use Arabic script."

    infer_result = gemma_core.infer(
        rewrite_system_prompt,
        response_text,
        image_path=None,
        max_tokens=VEDA_MAX_TOKENS,
    )
    return _sanitize_model_response(infer_result.response)


def _build_ready_response(language: str) -> str:
    target_language = _normalize_language_name(language)
    if target_language == "English":
        return READY_MESSAGE_EN

    try:
        rewritten = _rewrite_response_in_language(READY_MESSAGE_EN, target_language)
        rewritten = _sanitize_model_response(rewritten)
        return rewritten or READY_MESSAGE_EN
    except Exception:
        return READY_MESSAGE_EN


def _enforce_response_language(response_text: str, language: str) -> str:
    if not response_text:
        return response_text
    if not _has_language_script_mismatch(response_text, language):
        return response_text

    try:
        rewritten = _rewrite_response_in_language(response_text, language)
    except Exception:
        return response_text

    if rewritten and not _has_language_script_mismatch(rewritten, language):
        return rewritten
    return rewritten or response_text


def build_system_prompt(language: str = "English", drug_context: str = "") -> str:
    return build_veda_prompt(language=language, drug_context=drug_context)


def _should_generate_tts(audio_path: Optional[str]) -> bool:
    return ENABLE_VOICE_TTS if audio_path else ENABLE_TEXT_TTS


def _trim_for_tts(text: str) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= TTS_MAX_CHARS:
        return cleaned
    return cleaned[:TTS_MAX_CHARS].rstrip() + "..."


def _is_meta_line(line: str) -> bool:
    stripped = (line or "").strip()
    if not stripped:
        return False

    lowered = stripped.lower().lstrip("* ")
    meta_prefixes = [
        "user message:",
        "persona:",
        "goal:",
        "constraints:",
        "language:",
        "acknowledge:",
        "steps:",
        "revised plan:",
        "final answer:",
        "draft answer:",
        "mirror view:",
        "step 1:",
        "step 2:",
        "step 3:",
        "step 4:",
        "step 5:",
    ]
    if any(lowered.startswith(prefix) for prefix in meta_prefixes):
        return True

    if "output rules" in lowered or "check against" in lowered:
        return True

    if re.match(
        r'^[*\-\s]*check\s+["\']?(specific actions|likely cause|warning sign|professional care|3-5 numbered steps|short acknowledgment|english)["\']?\s*:',
        lowered,
        flags=re.IGNORECASE,
    ):
        return True

    if re.search(
        r"\b(3-5\s+clear|most likely cause|likely cause|warning sign|professional care at the end|short acknowledgment)\b",
        lowered,
        flags=re.IGNORECASE,
    ):
        return True

    if re.match(
        r"^[*\-\s]*\d+\.\s*(acknowledge|3-5\s+clear|most likely cause|likely cause|warning sign|professional care)",
        lowered,
        flags=re.IGNORECASE,
    ):
        return True

    if re.match(r"^[*\-\s]*\d+\.\s*steps?\s*:", lowered, flags=re.IGNORECASE):
        return True

    if re.match(
        r"^[*\-\s]*\d+\.\s*(explain how to|get the best help from me|identify the\s+\"?cause\"?\s+as\s+a\s+test)",
        lowered,
        flags=re.IGNORECASE,
    ):
        return True

    if "no analysis/planning" in lowered or "no labels" in lowered:
        return True

    if "applying a medical structure" in lowered or "test message" in lowered:
        return True

    if re.match(r"^[*\-\s]*\d+\.\s*(cause|warning)\s*:\s*n\/?a\.?$", lowered, flags=re.IGNORECASE):
        return True

    if lowered.startswith("example:"):
        return True

    if "no analysis" in lowered and "no labels" in lowered:
        return True

    if re.match(
        r'^(english\?|short acknowledgment\?|3-5 numbered steps\?|specific actions\?|likely cause\?|warning sign\?|professional care at the end\?|no "?i cannot"?)\s*:\s*yes\.?$',
        lowered,
        flags=re.IGNORECASE,
    ):
        return True

    if re.match(
        r"^[*\-\s]*[a-z0-9 ,/\"'()\-]{3,70}\?\s*(yes|no)\.?$",
        stripped,
        flags=re.IGNORECASE,
    ):
        return True

    if re.match(r"^[*\-\s]*\*{1,2}[a-z][a-z0-9\s\-/]{2,45}\*{1,2}\s*:", stripped, flags=re.IGNORECASE):
        return True

    if any(hint in lowered for hint in INTERNAL_REASONING_HINTS):
        return True

    if re.match(r"^[*\-\s]*\*?(wait|actually|self-correction|refined plan|final decision|alternative interpretation)\*?[:\-]?", stripped, flags=re.IGNORECASE):
        return True

    if re.match(r'^["\']\s*hello[^"\']+["\']\s*$', stripped, flags=re.IGNORECASE):
        return True

    return False


def _clean_user_facing_line(line: str) -> str:
    cleaned = (line or "").strip()
    if not cleaned:
        return ""

    # When models leak quoted final text as a markdown bullet, keep the text and drop wrappers.
    if re.match(r'^[*\-]\s*["\'].+["\']\s*$', cleaned):
        cleaned = re.sub(r'^[*\-]\s*', "", cleaned)
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
            cleaned = cleaned[1:-1].strip()

    return cleaned


def _sanitize_model_response(text: str) -> str:
    cleaned = (text or "").replace("\r\n", "\n").strip()
    if not cleaned:
        return ""

    lines = cleaned.split("\n")
    has_meta_lines = any(_is_meta_line(line) for line in lines)
    leaked_markers = ("user message:", "persona:", "constraints:", "goal:")
    has_leaked_markers = any(marker in cleaned.lower() for marker in leaked_markers)
    if not has_leaked_markers and not has_meta_lines:
        return cleaned

    # If a compliance checklist exists, start from the first line after the checklist block.
    checklist_indices = []
    if has_leaked_markers:
        checklist_indices = [
            i for i, line in enumerate(lines)
            if re.search(
                r'english\?|short acknowledgment\?|3-5 numbered steps\?|specific actions\?|likely cause\?|warning sign\?|professional care at the end\?|no "?i cannot"?',
                line,
                flags=re.IGNORECASE,
            )
        ]
    start_idx = checklist_indices[-1] + 1 if checklist_indices else 0

    if start_idx == 0:
        seen_meta = False
        for i, line in enumerate(lines):
            if _is_meta_line(line):
                seen_meta = True
                continue
            if seen_meta and line.strip() and not _is_meta_line(line):
                start_idx = i
                break

    filtered_lines = []
    for line in lines[start_idx:]:
        if _is_meta_line(line):
            continue
        cleaned_line = _clean_user_facing_line(line)
        if _is_meta_line(cleaned_line):
            continue
        filtered_lines.append(cleaned_line)

    # Remove leaked rubric bullets at the top when the actual answer follows.
    while filtered_lines:
        first = filtered_lines[0].strip()
        if not first.startswith("*"):
            break
        has_non_bullet_after = any(
            ln.strip() and not ln.strip().startswith("*")
            for ln in filtered_lines[1:]
        )
        if not has_non_bullet_after:
            break
        filtered_lines.pop(0)

    # Drop leaked numbered planning/checklist lines.
    while filtered_lines:
        first = filtered_lines[0].strip()
        if not re.match(
            r"^\d+\.\s*(acknowledge|steps?:|3-5\s+clear|most likely cause|likely cause|warning sign|professional care|explain how to|get the best help from me|identify the\s+\"?cause\"?\s+as\s+a\s+test)",
            first,
            flags=re.IGNORECASE,
        ):
            break
        filtered_lines.pop(0)

    # Drop leaked numbered rubric lines that can survive prefix stripping.
    while filtered_lines:
        first = filtered_lines[0].strip()
        if not re.match(
            r"^\d+\.\s*(acknowledge|3-5\s+clear|most likely cause|likely cause|warning sign|professional care|no \"?i cannot\"?)",
            first,
            flags=re.IGNORECASE,
        ):
            break
        filtered_lines.pop(0)

    result = "\n".join(filtered_lines).strip()
    if not result:
        return cleaned

    # Normalize excessive blank spacing from removed scaffolding.
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


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
    detected_language = _normalize_language_name(language)
    query_text = text_query or ""
    drug_context = ""
    drug_metadata: Dict[str, object] = {}
    raw_response = ""
    model_latency_ms = 0

    try:
        if audio_path and os.path.exists(audio_path):
            transcription = voice_utils.transcribe(audio_path)
            transcript = transcription.text
            query_text = transcript
            detected_language = _normalize_language_name(
                LANGUAGE_MAP.get(transcription.language, transcription.language)
            )
            detected_language = _adjust_detected_language_from_transcript(detected_language, transcript)

        drug_context = drug_lookup.get_drug_context(query_text)
        drug_metadata = drug_lookup.get_drug_context_metadata(query_text)

        if _looks_like_non_medical_test_query(query_text):
            raw_response = _build_ready_response(detected_language)
            model_latency_ms = 0
        else:
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
            raw_response = _sanitize_model_response(infer_result.response)
            model_latency_ms = infer_result.latency_ms

        image_context = f"Image: {image_path}" if image_path else "No image provided"
        routing_mode = (gemma_core.compute_router.get_status().get("mode") or "offline").strip().lower()
        mirror_token_budget = (
            MIRROR_AUDIT_MAX_TOKENS_OFFLINE
            if routing_mode == "offline"
            else MIRROR_AUDIT_MAX_TOKENS
        )
        mirror_report = mirror.audit(
            query=query_text,
            response=raw_response,
            image_context=image_context,
            max_tokens=mirror_token_budget,
        )
        response_text = _sanitize_model_response(
            mirror.apply_verdict(raw_response, mirror_report, detected_language)
        )
        should_rewrite_language = (
            ENABLE_RESPONSE_LANGUAGE_REWRITE
            and (audio_path is None or ENABLE_VOICE_LANGUAGE_REWRITE)
        )
        if should_rewrite_language:
            response_text = _enforce_response_language(response_text, detected_language)

        response_text = _sanitize_model_response(response_text)
        if not response_text:
            response_text = _safe_fallback_message(detected_language)

        detected_language = _reconcile_detected_language_with_response(detected_language, response_text)

    except Exception as exc:
        response_text = _safe_fallback_message(detected_language)
        raw_response = response_text
        if not drug_metadata:
            drug_metadata = {
                "query": query_text,
                "has_context": False,
                "match_count": 0,
                "matches": [],
                "sources": [],
                "live_lookup_enabled": False,
                "live_fallback_used": False,
            }
        mirror_report = mirror.MirrorReport(
            confidence_score=10,
            flags=["system_unavailable", "manual_escalation_required"],
            reasoning_trace=f"Pipeline unavailable: {exc}",
            verdict="block",
        )

    audio_response_path = ""
    if _should_generate_tts(audio_path):
        try:
            # Keep spoken output aligned with the exact finalized user-facing answer.
            audio_response_path = voice_utils.speak_to_file(response_text)
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
        drug_metadata=drug_metadata,
        audio_response_path=audio_response_path,
        model_latency_ms=model_latency_ms,
        processing_time_ms=processing_time_ms,
        image_path=image_path or "",
    )


def process_text_only(text_query: str, language: str = "English") -> VedaResult:
    return process_medical_query(
        image_path=None, audio_path=None,
        text_query=text_query, language=language, session_id=None,
    )
