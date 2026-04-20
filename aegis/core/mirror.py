from dataclasses import dataclass, field
from typing import Dict, List
from . import gemma_core
from .prompts import apply_mirror_verdict, build_mirror_prompt
import os
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


def _safe_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return int(default)


DEFAULT_MIRROR_MAX_TOKENS = max(96, min(_safe_int_env("AEGIS_MIRROR_MAX_TOKENS", 192), 384))

_VERDICT_ORDER = {"pass": 0, "warn": 1, "block": 2}
_HIGH_RISK_GUARDRAILS = [
    (
        "possible_cardiac_emergency",
        r"\b(chest pain|chest pressure|chest tightness|heart attack|pain radiating to .*left arm|left arm pain with chest)\b",
        "block",
        "query",
    ),
    (
        "possible_overdose",
        r"\b(overdose|too much insulin|insulin overdose|too many (sleeping )?pills|too many tablets|exact antidote|antidote dose|poison(ing|ed)?)\b",
        "block",
        "query",
    ),
    (
        "self_harm_crisis",
        r"\b(do not want to go on|don't want to go on|end my life|kill myself|self-harm|suicide)\b",
        "block",
        "query",
    ),
    (
        "pediatric_medication_risk",
        r"\b(\d{1,2}\s*-?\s*month(-old)?|infant|newborn|baby)\b.{0,120}\b(medicine|medication|dose|dosing|adult cough|syrup)\b",
        "warn",
        "query",
    ),
    (
        "antibiotic_self_prescription",
        r"\b(which antibiotic|best antibiotic|exact dose.*antibiotic|prescribe.*antibiotic|buy .*antibiotic)\b",
        "warn",
        "query",
    ),
]


def _normalize_verdict(value: str) -> str:
    verdict = (value or "warn").strip().lower()
    return verdict if verdict in ["pass", "warn", "block"] else "warn"


def parse_mirror_response(raw_response: str) -> MirrorReport:
    # FIX 5: Default confidence_score is 50 (neutral), not 0.
    # Returning 0 was causing the UI to show a broken red gauge on every response.
    report = MirrorReport(confidence_score=50, verdict="warn")

    try:
        cleaned = (raw_response or "").strip()
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)

        saw_confidence = False
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

            if upper.startswith("REASONING"):
                report.reasoning_trace = line.split(":", 1)[-1].strip()
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

        # Secondary scan for confidence if not found on labelled line
        if not saw_confidence:
            number_after_conf = re.search(r"confidence[^\d]*(\d{1,3})", cleaned, re.IGNORECASE)
            if number_after_conf:
                report.confidence_score = max(0, min(100, int(number_after_conf.group(1))))

        # Heuristic fallback for loosely formatted audits.
        lower = cleaned.lower()
        if "harm: no" not in lower and (
            "dangerous" in lower
            or "potential harm" in lower
            or "direct harm" in lower
            or re.search(r"\bharm\b", lower)
        ):
            if "potential_harm" not in report.flags:
                report.flags.append("potential_harm")

        if "hallucination: no" not in lower and "hallucin" in lower:
            if "possible_hallucination" not in report.flags:
                report.flags.append("possible_hallucination")

        if "overdiagnosis: no" not in lower and "overdiagnos" in lower:
            if "overdiagnosis" not in report.flags:
                report.flags.append("overdiagnosis")

        if "scope: within scope" not in lower and (
            "scope overstepped" in lower
            or "out of scope" in lower
            or "overstep" in lower
        ):
            if "scope_overstepped" not in report.flags:
                report.flags.append("scope_overstepped")

        # Infer verdict from score if VERDICT line was missing
        if not saw_verdict:
            if report.confidence_score < 30 or len(report.flags) >= 2:
                report.verdict = "block"
            elif report.confidence_score < 60 or len(report.flags) >= 1:
                report.verdict = "warn"
            else:
                report.verdict = "pass"

        report.verdict = _normalize_verdict(report.verdict)
        return report

    except Exception:
        return MirrorReport(
            confidence_score=20,
            flags=["audit_parse_failed", "manual_escalation_required"],
            reasoning_trace="Failed to parse audit response safely. Manual escalation is required.",
            verdict="block",
        )


def _upgrade_verdict(current: str, requested: str) -> str:
    current_norm = _normalize_verdict(current)
    requested_norm = _normalize_verdict(requested)
    if _VERDICT_ORDER[requested_norm] > _VERDICT_ORDER[current_norm]:
        return requested_norm
    return current_norm


def _apply_high_risk_guardrails(query: str, response: str, report: MirrorReport) -> MirrorReport:
    inspected_query = (query or "").lower()
    inspected_response = (response or "").lower()
    triggered_flags: List[str] = []
    requested_verdict = report.verdict

    for flag, pattern, forced_verdict, source in _HIGH_RISK_GUARDRAILS:
        inspected_text = inspected_query if source == "query" else f"{inspected_query}\n{inspected_response}"
        if re.search(pattern, inspected_text, flags=re.IGNORECASE):
            triggered_flags.append(flag)
            requested_verdict = _upgrade_verdict(requested_verdict, forced_verdict)

    if not triggered_flags:
        return report

    for flag in triggered_flags:
        if flag not in report.flags:
            report.flags.append(flag)

    report.verdict = _upgrade_verdict(report.verdict, requested_verdict)
    report.confidence_score = min(report.confidence_score, 80)

    note = "Rule-based guardrail flagged high-risk pattern(s): " + ", ".join(triggered_flags)
    if report.reasoning_trace:
        report.reasoning_trace = f"{report.reasoning_trace} | {note}"
    else:
        report.reasoning_trace = note

    return report


def audit(
    query: str,
    response: str,
    image_context: str = "",
    max_tokens: int = 0,
) -> MirrorReport:
    start_time = time.time()
    token_budget = max(96, min(int(max_tokens) if max_tokens else DEFAULT_MIRROR_MAX_TOKENS, 384))

    prompt = build_mirror_prompt(query=query, image_context=image_context, response=response)
    raw_result = ""

    try:
        # MIRROR always runs — no deferral, no skipping.
        # Do not force cloud here; let gemma_core route naturally so offline mode
        # never blocks on cloud attempts.
        infer_result = gemma_core.infer(
            system_prompt=prompt,
            user_message="Provide your safety audit of the response above.",
            image_path=None,
            max_tokens=token_budget,
            force_cloud=False,
        )
        raw_result = infer_result.response if hasattr(infer_result, "response") else str(infer_result)
        report = parse_mirror_response(raw_result)
        report = _apply_high_risk_guardrails(query, response, report)

    except Exception as exc:
        raw_result = str(exc)
        # Fail closed on audit unavailability for medical safety.
        report = MirrorReport(
            confidence_score=20,
            flags=["audit_unavailable", "manual_escalation_required"],
            reasoning_trace=f"MIRROR audit could not complete safely: {exc}. Blocking response pending manual escalation.",
            verdict="block",
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
