import base64
import json
import os
import re
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
from faster_whisper import WhisperModel

try:
    import pyttsx3
except Exception:  # pragma: no cover - optional local fallback only
    pyttsx3 = None


_whisper_model = None
_tts_engine = None


@dataclass
class TranscriptionResult:
    text: str
    language: str
    confidence: float
    duration_ms: int


def _is_env_enabled(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _safe_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _safe_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _default_wav_output_path(output_path: Optional[str] = None) -> str:
    if output_path:
        return output_path
    temp_name = f"aegis_tts_{int(time.time() * 1000)}.wav"
    return os.path.join(tempfile.gettempdir(), temp_name)


def load_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return

    print("Loading Whisper model for speech recognition...")
    start_time = time.time()
    _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    elapsed = time.time() - start_time
    print(f"Whisper loaded in {elapsed:.1f}s")


def transcribe(audio_path: str) -> TranscriptionResult:
    load_whisper()
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    start_time = time.time()
    segments, info = _whisper_model.transcribe(audio_path, beam_size=5, language=None)
    full_text = " ".join(segment.text.strip() for segment in segments).strip()
    elapsed_ms = int((time.time() - start_time) * 1000)

    return TranscriptionResult(
        text=full_text,
        language=info.language,
        confidence=float(info.language_probability),
        duration_ms=elapsed_ms,
    )


def init_tts():
    global _tts_engine
    if _tts_engine is not None:
        return

    if pyttsx3 is None:
        raise RuntimeError("pyttsx3 is unavailable for local TTS fallback")

    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty("rate", 150)
    try:
        _tts_engine.setProperty("volume", 0.9)
    except Exception:
        pass


def _get_google_tts_api_key() -> str:
    return (
        os.environ.get("AEGIS_GOOGLE_TTS_API_KEY", "").strip()
        or os.environ.get("AEGIS_CLOUD_API_KEY", "").strip()
    )


def _get_google_tts_url(api_key: str) -> str:
    configured_url = os.environ.get("AEGIS_GOOGLE_TTS_API_URL", "").strip()
    model_name = os.environ.get("AEGIS_GOOGLE_TTS_MODEL", "gemini-2.5-flash-preview-tts").strip()

    if configured_url:
        base_url = configured_url
    else:
        base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

    if "key=" in base_url:
        return base_url

    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}key={api_key}"


def _google_tts_payload_variants(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    voice_name = os.environ.get("AEGIS_GOOGLE_TTS_VOICE", "Kore").strip() or "Kore"

    generation_config = {
        "responseModalities": ["AUDIO"],
        "speechConfig": {
            "voiceConfig": {
                "prebuiltVoiceConfig": {
                    "voiceName": voice_name,
                }
            }
        },
    }

    primary = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": generation_config,
    }
    compatibility = {
        "contents": [{"parts": [{"text": text}]}],
        "config": generation_config,
    }
    return primary, compatibility


def _extract_inline_audio(part: Dict[str, Any]) -> Optional[Tuple[bytes, str]]:
    inline = part.get("inlineData") if isinstance(part, dict) else None
    if inline is None and isinstance(part, dict):
        inline = part.get("inline_data")
    if not isinstance(inline, dict):
        return None

    audio_b64 = inline.get("data")
    if not isinstance(audio_b64, str) or not audio_b64.strip():
        return None

    mime_type = inline.get("mimeType") or inline.get("mime_type") or "audio/wav"
    try:
        decoded = base64.b64decode(audio_b64)
    except Exception as exc:
        raise RuntimeError("Google TTS returned invalid base64 audio payload") from exc
    return decoded, str(mime_type).lower()


def _extract_google_tts_audio(response_payload: Dict[str, Any]) -> Tuple[bytes, str]:
    candidates = response_payload.get("candidates", [])
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content", {})
        if not isinstance(content, dict):
            continue
        parts = content.get("parts", [])
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            extracted = _extract_inline_audio(part)
            if extracted is not None:
                return extracted

    raise RuntimeError("Google TTS response did not include inline audio")


def _pcm_sample_rate_from_mime(mime_type: str) -> int:
    matched = re.search(r"rate=(\d+)", mime_type)
    if matched:
        try:
            return max(8000, int(matched.group(1)))
        except Exception:
            pass
    return max(8000, _safe_int_env("AEGIS_GOOGLE_TTS_SAMPLE_RATE_HZ", 24000))


def _write_pcm_wav(audio_bytes: bytes, output_path: str, sample_rate_hz: int) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(audio_bytes)
    return output_path


def _write_google_audio_to_file(audio_bytes: bytes, mime_type: str, output_path: Optional[str]) -> str:
    output = _default_wav_output_path(output_path)
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    if audio_bytes.startswith(b"RIFF") or "audio/wav" in mime_type or "audio/x-wav" in mime_type:
        Path(output).write_bytes(audio_bytes)
        return output

    if "audio/pcm" in mime_type or "audio/l16" in mime_type:
        return _write_pcm_wav(audio_bytes, output, _pcm_sample_rate_from_mime(mime_type))

    raise RuntimeError(f"Unsupported Google TTS audio format: {mime_type}")


def _speak_with_google_ai_studio(text: str, output_path: Optional[str] = None) -> str:
    api_key = _get_google_tts_api_key()
    if not api_key:
        raise RuntimeError("Google AI Studio TTS requires AEGIS_GOOGLE_TTS_API_KEY or AEGIS_CLOUD_API_KEY")

    timeout_sec = max(5.0, _safe_float_env("AEGIS_GOOGLE_TTS_TIMEOUT_SEC", 35.0))
    endpoint = _get_google_tts_url(api_key)
    payload_primary, payload_compat = _google_tts_payload_variants(text)

    last_error: Optional[Exception] = None
    for payload in (payload_primary, payload_compat):
        try:
            response = requests.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=timeout_sec,
            )
            if response.status_code >= 400:
                snippet = (response.text or "")[:300]
                raise RuntimeError(f"Google TTS HTTP {response.status_code}: {snippet}")

            body = response.json()
            audio_bytes, mime_type = _extract_google_tts_audio(body)
            return _write_google_audio_to_file(audio_bytes, mime_type, output_path)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Google AI Studio TTS failed: {last_error}")


def _speak_with_local_pyttsx3(text: str, output_path: Optional[str] = None) -> str:
    init_tts()
    output = _default_wav_output_path(output_path)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    _tts_engine.save_to_file(text, output)
    _tts_engine.runAndWait()
    return output


def _get_tts_provider() -> str:
    return os.environ.get("AEGIS_TTS_PROVIDER", "google_ai_studio").strip().lower()


def speak_to_file(text: str, output_path: str = None) -> str:
    normalized_text = (text or "").strip()
    if not normalized_text:
        raise ValueError("TTS text cannot be empty")

    provider = _get_tts_provider()

    if provider in {"google", "google_ai_studio", "google-studio", "aistudio", "default"}:
        try:
            return _speak_with_google_ai_studio(normalized_text, output_path)
        except Exception:
            if not _is_env_enabled("AEGIS_GOOGLE_TTS_FALLBACK_LOCAL", False):
                raise
            return _speak_with_local_pyttsx3(normalized_text, output_path)

    if provider in {"local", "pyttsx3"}:
        return _speak_with_local_pyttsx3(normalized_text, output_path)

    if provider == "auto":
        try:
            return _speak_with_google_ai_studio(normalized_text, output_path)
        except Exception:
            return _speak_with_local_pyttsx3(normalized_text, output_path)

    raise RuntimeError(f"Unsupported AEGIS_TTS_PROVIDER: {provider}")


def create_test_audio(text: str = "This is a test", output_path: str = None) -> str:
    if output_path is None:
        output_path = os.path.join("aegis", "tests", "test_audio.wav")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return speak_to_file(text, output_path=output_path)


if __name__ == "__main__":
    print("Testing AEGIS Voice Utilities...")

    tts_file = speak_to_file("Hello, this is AEGIS speaking")
    print(f"TTS output: {tts_file}")

    test_audio_file = create_test_audio()
    print(f"Test audio: {test_audio_file}")

    result = transcribe(test_audio_file)
    print(f"Transcription: {result.text}")
    print(f"Language: {result.language}")
    print(f"Confidence: {result.confidence:.2f}")
