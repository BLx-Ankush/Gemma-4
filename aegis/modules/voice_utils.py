from faster_whisper import WhisperModel
import pyttsx3
import os
import time
import wave
import tempfile
from dataclasses import dataclass


_whisper_model = None
_tts_engine = None


@dataclass
class TranscriptionResult:
    text: str
    language: str
    confidence: float
    duration_ms: int


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

    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty("rate", 150)
    try:
        _tts_engine.setProperty("volume", 0.9)
    except Exception:
        pass


def speak_to_file(text: str, output_path: str = None) -> str:
    init_tts()
    if output_path is None:
        temp_name = f"aegis_tts_{int(time.time() * 1000)}.wav"
        output_path = os.path.join(tempfile.gettempdir(), temp_name)

    _tts_engine.save_to_file(text, output_path)
    _tts_engine.runAndWait()
    return output_path


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
