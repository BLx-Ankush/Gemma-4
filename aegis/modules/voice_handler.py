from dataclasses import dataclass, field
from typing import Dict, List, Optional
import uuid
import time

from . import veda


@dataclass
class VoiceTurn:
    timestamp: float
    user_text: str
    assistant_text: str
    language: str
    audio_response_path: str
    image_path: str = ""


@dataclass
class VoiceSession:
    session_id: str
    created_at: float
    last_updated: float
    turns: List[VoiceTurn] = field(default_factory=list)


_SESSIONS: Dict[str, VoiceSession] = {}


def _normalize_text(value: object) -> str:
    return str(value or "").strip()


def _build_mirror_view_payload(result: veda.VedaResult) -> dict:
    draft = _normalize_text(result.raw_response)
    final = _normalize_text(result.response_text)
    rectified = draft != final

    verdict = (result.mirror_report.verdict or "warn").strip().lower()
    if verdict == "block":
        action = "MIRROR blocked unsafe output and returned a safe fallback response."
    elif rectified:
        action = "MIRROR rectified the draft response before final delivery."
    else:
        action = "Draft response passed MIRROR checks; no rectification was needed."

    return {
        "draft_response": draft,
        "final_response": final,
        "rectified": rectified,
        "rectification_action": action,
        "verdict": verdict,
        "confidence_score": int(max(0, min(100, result.mirror_report.confidence_score))),
        "flags": list(result.mirror_report.flags),
        "reasoning_trace": _normalize_text(result.mirror_report.reasoning_trace),
    }


def create_session() -> str:
    session_id = str(uuid.uuid4())
    now = time.time()
    _SESSIONS[session_id] = VoiceSession(
        session_id=session_id,
        created_at=now,
        last_updated=now,
    )
    return session_id


def get_session(session_id: str) -> Optional[VoiceSession]:
    return _SESSIONS.get(session_id)


def clear_session(session_id: str) -> bool:
    if session_id in _SESSIONS:
        del _SESSIONS[session_id]
        return True
    return False


def get_session_history(session_id: str) -> List[dict]:
    session = get_session(session_id)
    if not session:
        return []

    history = []
    for turn in session.turns:
        history.append(
            {
                "timestamp": turn.timestamp,
                "user_text": turn.user_text,
                "assistant_text": turn.assistant_text,
                "language": turn.language,
                "audio_response_path": turn.audio_response_path,
                "image_path": turn.image_path,
            }
        )
    return history


def process_voice_turn(
    audio_path: str,
    session_id: Optional[str] = None,
    image_path: Optional[str] = None,
    language: str = "English",
) -> dict:
    if not session_id:
        session_id = create_session()

    session = get_session(session_id)
    if not session:
        _SESSIONS[session_id] = VoiceSession(
            session_id=session_id,
            created_at=time.time(),
            last_updated=time.time(),
        )
        session = _SESSIONS[session_id]

    result = veda.process_medical_query(
        image_path=image_path,
        audio_path=audio_path,
        text_query=None,
        language=language,
        session_id=session_id,
    )

    turn = VoiceTurn(
        timestamp=time.time(),
        user_text=result.query_text,
        assistant_text=result.response_text,
        language=result.detected_language,
        audio_response_path=result.audio_response_path,
        image_path=result.image_path,
    )

    session.turns.append(turn)
    session.last_updated = time.time()

    return {
        "session_id": session_id,
        "transcript": result.transcript,
        "detected_language": result.detected_language,
        "query_text": result.query_text,
        "response_text": result.response_text,
        "audio_response_path": result.audio_response_path,
        "mirror": {
            "confidence_score": result.mirror_report.confidence_score,
            "risk_tags": result.mirror_report.flags,
            "flags": result.mirror_report.flags,
            "verdict": result.mirror_report.verdict,
            "notes": result.mirror_report.reasoning_trace,
            "audit_time_ms": result.mirror_report.audit_time_ms,
        },
        "mirror_view": _build_mirror_view_payload(result),
        "drug_context": result.drug_context,
        "drug_lookup": result.drug_metadata,
        "model_latency_ms": result.model_latency_ms,
        "processing_time_ms": result.processing_time_ms,
        "turn_count": len(session.turns),
    }


if __name__ == "__main__":
    sid = create_session()
    print("Created session:", sid)
    print("Session history:", get_session_history(sid))