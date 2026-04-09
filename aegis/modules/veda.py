from dataclasses import dataclass
from typing import Optional
from aegis.core import gemma_core, mirror
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


VEDA_SYSTEM_PROMPT = """You are a medical first-response assistant. You are helping a
person who may not have access to a doctor or hospital. You are
speaking to them directly in {language}.

Rules you must follow:
- Never state a definitive diagnosis. Always use phrases like
  "this appears to be" or "this may be"
- Always provide immediate first-aid steps the person can take
  right now with materials they likely have available
- Always recommend seeking professional medical care when possible
- Use simple everyday language. If you must use a medical term,
  immediately explain it in plain words
- Limit your response to 3 to 5 sentences maximum
- End every response with one clear action the person should
  take right now
- Never recommend specific prescription medications by name
  unless the user specifically asks about a medication they
  already have
- If you are not confident about what you see in the image,
  say so clearly
- Be warm, calm, and reassuring in tone

{drug_context}"""


LANGUAGE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "es": "Spanish",
    "fr": "French",
    "sw": "Swahili",
    "ar": "Arabic",
    "pt": "Portuguese",
    "zh": "Chinese",
    "bn": "Bengali",
    "ur": "Urdu",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
}


SAFE_FALLBACK_MESSAGES = {
    "English": "I cannot safely process this request right now. Please seek help from a nearby health worker or emergency service immediately.",
    "Hindi": "Main is samay surakshit roop se is prashn ko process nahi kar pa raha hoon. Kripya turant kisi health worker ya emergency seva se sampark karein.",
    "Kannada": "Iga ee vinantiyannu surakshitavagi prakriye maadalu sadyavilla. Dayavittu hatra iruva aroghya karmikarannu athava aapathkaaleena seveyannu takshan samparkisi.",
}


def _safe_fallback_message(language: str) -> str:
    return SAFE_FALLBACK_MESSAGES.get(language, SAFE_FALLBACK_MESSAGES["English"])


def build_system_prompt(language: str = "English", drug_context: str = "") -> str:
    if drug_context:
        drug_context_text = "\nRelevant drug information:\n" + drug_context
    else:
        drug_context_text = ""

    return (
        VEDA_SYSTEM_PROMPT.replace("{language}", language).replace("{drug_context}", drug_context_text)
    )


def process_medical_query(
    image_path: str = None,
    audio_path: str = None,
    text_query: str = None,
    language: str = "English",
    session_id: str = None,
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

        infer_result = gemma_core.infer(system_prompt, user_message, image_path)
        raw_response = infer_result.response
        model_latency_ms = infer_result.latency_ms

        image_context = f"Image: {image_path}" if image_path else ""
        mirror_report = mirror.audit(query=query_text, response=raw_response, image_context=image_context)
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

    try:
        audio_response_path = voice_utils.speak_to_file(response_text)
    except Exception:
        audio_response_path = ""

    processing_time_ms = model_latency_ms or int((time.time() - start_time) * 1000)
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
        image_path=None,
        audio_path=None,
        text_query=text_query,
        language=language,
        session_id=None,
    )


if __name__ == "__main__":
    print("=" * 50)
    print("AEGIS VEDA Module — Full Pipeline Test")
    print("=" * 50)

    print("\n--- Test 1: Text query, no image ---")
    result = process_text_only("I burned my hand on a hot pot. What should I do?")
    print(f"Response: {result.response_text}")
    print(
        f"MIRROR: confidence={result.mirror_report.confidence_score}, verdict={result.mirror_report.verdict}"
    )
    print(f"Drug context: {result.drug_context or 'None'}")
    print(f"Processing time: {result.processing_time_ms}ms")

    print("\n--- Test 2: Drug query ---")
    result = process_text_only("Can I give paracetamol to my child who has a fever?")
    print(f"Response: {result.response_text}")
    print(f"Drug context found: {'Yes' if result.drug_context else 'No'}")
    print(
        f"MIRROR: confidence={result.mirror_report.confidence_score}, verdict={result.mirror_report.verdict}"
    )

    print("\n--- Test 3: Edge case query ---")
    result = process_text_only("I have severe chest pain and difficulty breathing")
    print(f"Response: {result.response_text}")
    print(
        f"MIRROR: confidence={result.mirror_report.confidence_score}, verdict={result.mirror_report.verdict}, flags={result.mirror_report.flags}"
    )
