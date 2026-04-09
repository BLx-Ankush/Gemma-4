from pathlib import Path

import pytest

from aegis.modules import voice_utils


class _FakeTtsEngine:
    def __init__(self):
        self.last_text = None
        self.last_path = None

    def setProperty(self, *_args, **_kwargs):
        return None

    def save_to_file(self, text, path):
        self.last_text = text
        self.last_path = path
        Path(path).write_bytes(b"RIFF")

    def runAndWait(self):
        return None


class _FakeWhisperModel:
    def transcribe(self, _audio_path, beam_size=5, language=None):
        class _Segment:
            def __init__(self, text):
                self.text = text

        class _Info:
            language = "en"
            language_probability = 0.93

        return [_Segment("hello"), _Segment("world")], _Info()


def test_speak_to_file_writes_file(monkeypatch, tmp_path):
    fake_engine = _FakeTtsEngine()
    monkeypatch.setattr(voice_utils, "_tts_engine", fake_engine)
    monkeypatch.setattr(voice_utils, "init_tts", lambda: None)

    output_path = tmp_path / "reply.wav"
    result = voice_utils.speak_to_file("test response", str(output_path))

    assert result == str(output_path)
    assert output_path.exists()
    assert fake_engine.last_text == "test response"


def test_transcribe_returns_structured_result(monkeypatch, tmp_path):
    audio_file = tmp_path / "input.wav"
    audio_file.write_bytes(b"RIFF")

    monkeypatch.setattr(voice_utils, "_whisper_model", _FakeWhisperModel())
    monkeypatch.setattr(voice_utils, "load_whisper", lambda: None)

    result = voice_utils.transcribe(str(audio_file))

    assert result.text == "hello world"
    assert result.language == "en"
    assert result.confidence == 0.93
    assert result.duration_ms >= 0


def test_transcribe_raises_for_missing_file(monkeypatch):
    monkeypatch.setattr(voice_utils, "_whisper_model", _FakeWhisperModel())
    monkeypatch.setattr(voice_utils, "load_whisper", lambda: None)

    with pytest.raises(FileNotFoundError):
        voice_utils.transcribe("does-not-exist.wav")
