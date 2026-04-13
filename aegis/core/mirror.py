from dataclasses import dataclass, field
from typing import Dict, List
from . import gemma_core
from .prompts import apply_mirror_verdict, build_mirror_prompt
import re
import time


@dataclass
class MirrorReport:
    confidence_score: int = 0
    flags: List[str] = field(default_factory=list)
    reasoning_trace: str = ""
    verdict: str = "warn"
    audit_time_ms: int = 0


MAX_AUDIT_LOG_ENTRIES = 200
_AUDIT_LOG: List[Dict[str, object]] = []


def _normalize_verdict(value: str) -> str:
    verdict = (value or "warn").strip().lower()
    return verdict if verdict in ["pass", "warn", "block"] else "warn"


def parse_mirror_response(raw_response: str) -> MirrorReport:
    report = MirrorReport(confidence_score=50, verdict="warn")

    try:
        cleaned = (raw_response or "").strip()
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)

        saw_confidence = False
        saw_reasoning = False
        saw_verdict = False

        for line in cleaned.splitlines():
            line = line.strip()
            if not line:
                continue

            upper = line.upper()

            if upper.startswith("CONFIDENCE") or "CONFIDENCE" in upper:
                numbers = re.findall(r"\d+", line)
                if numbers:
                    report.confidence_score = max(0, min(100, int(numbers[0])))
                    saw_confidence = True
                continue

            if upper.startswith("FLAGS"):
                raw_flags = line.split(":", 1)[-1].strip()
                if raw_flags.lower() in ("none", "none.", "", "-", "n/a"):
                    report.flags = []
                else:
                    report.flags = [
                        part.strip().rstrip(".")
                        for part in re.split(r"[,;]", raw_flags)
                        if part.strip() and part.strip().lower() != "none"
                    ]
                continue

            if upper.startswith("HARM") and "YES" in upper:
                if "potential_harm" not in report.flags:
                    report.flags.append("potential_harm")
                continue

            if upper.startswith("HALLUCINATION") and "YES" in upper:
                if "possible_hallucination" not in report.flags:
                    report.flags.append("possible_hallucination")
                continue

            if upper.startswith("OVERDIAGNOSIS") and "YES" in upper:
                if "overdiagnosis" not in report.flags:
                    report.flags.append("overdiagnosis")
                continue

            if upper.startswith("SCOPE") and "OVERSTEPPED" in upper:
                if "scope_overstepped" not in report.flags:
                    report.flags.append("scope_overstepped")
                continue

            if upper.startswith("REASONING"):
                report.reasoning_trace = line.split(":", 1)[-1].strip()
                saw_reasoning = True
                continue

            if upper.startswith("VERDICT"):
                verdict_text = line.split(":", 1)[-1].strip().lower()
                if "block" in verdict_text:
                    report.verdict = "block"
                elif "warn" in verdict_text:
                    report.verdict = "warn"
                elif "pass" in verdict_text:
                    report.verdict = "pass"
                else:
                    report.verdict = "warn"
                saw_verdict = True
                continue

        if not saw_confidence:
            number_after_conf = re.search(r"confidence[^\d]*(\d{1,3})", cleaned, re.IGNORECASE)
            if number_after_conf:
                report.confidence_score = max(0, min(100, int(number_after_conf.group(1))))
                saw_confidence = True

        lower = cleaned.lower()
        if (
            "dangerous" in lower
            or "potential harm" in lower
            or "direct harm" in lower
            or ("harm" in lower and not re.search(r"harm\s*:\s*no", lower))
        ):
            if "potential_harm" not in report.flags:
                report.flags.append("potential_harm")
        if "hallucin" in lower and not re.search(r"hallucination\s*:\s*no", lower):
            if "possible_hallucination" not in report.flags:
                report.flags.append("possible_hallucination")
        if "overdiagnos" in lower and not re.search(r"overdiagnosis\s*:\s*no", lower):
            if "overdiagnosis" not in report.flags:
                report.flags.append("overdiagnosis")

        if not saw_verdict:
            if report.confidence_score < 30 or len(report.flags) >= 2:
                report.verdict = "block"
            elif report.confidence_score < 60 or len(report.flags) >= 1:
                report.verdict = "warn"
            else:
                report.verdict = "pass"

        # If output had no usable structure, keep neutral confidence and warn.
        if not saw_confidence and not saw_reasoning and not saw_verdict and not report.flags:
            report.flags = ["audit_parse_incomplete"]
            report.reasoning_trace = f"MIRROR parse incomplete. Raw output: {cleaned[:200]}"
            report.confidence_score = 50
            report.verdict = "warn"

        report.verdict = _normalize_verdict(report.verdict)
        return report
    except Exception:
        return MirrorReport(
            confidence_score=50,
            flags=["audit_parse_failed"],
            reasoning_trace="Failed to parse audit response",
            verdict="warn",
        )


def audit(query: str, response: str, image_context: str = "") -> MirrorReport:
    start_time = time.time()

    prompt = build_mirror_prompt(query=query, image_context=image_context, response=response)
    try:
        infer_result = gemma_core.infer(
            system_prompt=prompt,
            user_message="Please audit the response above and provide your assessment.",
            image_path=None,
            max_tokens=12,
            force_cloud=True,
        )
        raw_result = infer_result.response if hasattr(infer_result, "response") else str(infer_result)
        report = parse_mirror_response(raw_result)
    except Exception as exc:
        raw_result = str(exc)
        report = MirrorReport(
            confidence_score=60,
            flags=["audit_unavailable"],
            reasoning_trace=f"MIRROR could not complete audit: {exc}. Response passed with caveat.",
            verdict="warn",
        )

    report.audit_time_ms = int((time.time() - start_time) * 1000)

    _AUDIT_LOG.append(
        {
            "timestamp": time.time(),
            "query": query,
            "response": response,
            "image_context": image_context,
            "raw_audit": raw_result,
            "report": {
                "confidence_score": report.confidence_score,
                "flags": list(report.flags),
                "reasoning_trace": report.reasoning_trace,
                "verdict": report.verdict,
                "audit_time_ms": report.audit_time_ms,
            },
            "backend": gemma_core.get_model_info().get("active_backend", "unknown"),
        }
    )
    if len(_AUDIT_LOG) > MAX_AUDIT_LOG_ENTRIES:
        del _AUDIT_LOG[: len(_AUDIT_LOG) - MAX_AUDIT_LOG_ENTRIES]

    return report


def get_audit_log(limit: int = 20) -> List[Dict[str, object]]:
    safe_limit = max(1, min(200, int(limit)))
    return list(reversed(_AUDIT_LOG[-safe_limit:]))


def apply_verdict(original_response: str, report: MirrorReport, language: str = "English") -> str:
    lang = (language or "en").strip().lower()
    if lang.startswith("hindi"):
        language_code = "hi"
    elif lang.startswith("tamil"):
        language_code = "ta"
    elif lang.startswith("kannada"):
        language_code = "kn"
    elif lang.startswith("telugu"):
        language_code = "te"
    else:
        language_code = lang[:2] if len(lang) >= 2 else "en"

    return apply_mirror_verdict(original_response, report.verdict, language_code)


if __name__ == "__main__":
    print("--- Test 1: Safe response ---")
    query = "I have a small cut on my finger"
    response = "Clean the cut gently with clean water. Apply a clean bandage. If bleeding does not stop in 10 minutes, seek medical help."
    safe_report = audit(query=query, response=response)
    print(safe_report)

    print("\n--- Test 2: Dangerous response ---")
    query = "I have chest pain"
    response = "You are having a heart attack. Take 3 aspirin tablets of 500mg each immediately and drive yourself to the hospital."
    danger_report = audit(query=query, response=response)
    print(danger_report)
