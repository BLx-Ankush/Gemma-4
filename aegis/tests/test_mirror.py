from aegis.core import mirror


def test_parse_mirror_response_strict_format():
    raw = (
        "CONFIDENCE: 82\n"
        "HARM: NO\n"
        "HALLUCINATION: NO\n"
        "OVERDIAGNOSIS: NO\n"
        "SCOPE: WITHIN SCOPE\n"
        "REASONING: Safe first-aid tone with clear uncertainty.\n"
        "VERDICT: PASS"
    )

    report = mirror.parse_mirror_response(raw)

    assert report.confidence_score == 82
    assert report.flags == []
    assert report.verdict == "pass"
    assert "Safe first-aid" in report.reasoning_trace


def test_parse_mirror_response_fallback_block_when_risky_text():
    raw = "Confidence maybe 20. This sounds dangerous and may hallucinate diagnosis."

    report = mirror.parse_mirror_response(raw)

    assert report.confidence_score == 20
    assert "potential_harm" in report.flags
    assert "possible_hallucination" in report.flags
    assert report.verdict == "block"


def test_apply_verdict_modes():
    original = "Use cool running water on the burn for 20 minutes."

    pass_report = mirror.MirrorReport(confidence_score=90, verdict="pass")
    warn_report = mirror.MirrorReport(confidence_score=55, verdict="warn")
    block_report = mirror.MirrorReport(confidence_score=10, verdict="block")

    assert mirror.apply_verdict(original, pass_report) == original
    assert mirror.apply_verdict(original, warn_report) == original
    assert "not confident enough" in mirror.apply_verdict(original, block_report)


def test_parse_mirror_response_handles_fractional_confidence_format():
    raw = (
        "CONFIDENCE: 85/100\n"
        "FLAGS: none\n"
        "REASONING: Practical first-aid guidance with reasonable caution.\n"
        "VERDICT: pass"
    )

    report = mirror.parse_mirror_response(raw)

    assert report.confidence_score == 85
    assert report.flags == []
    assert report.verdict == "pass"


def test_parse_mirror_response_handles_markdown_heading_fields():
    raw = (
        "**CONFIDENCE**: 73\n"
        "**FLAGS**: none\n"
        "**REASONING**: The answer is useful and remains within first-aid scope.\n"
        "**VERDICT**: warn"
    )

    report = mirror.parse_mirror_response(raw)

    assert report.confidence_score == 73
    assert report.verdict == "warn"


def test_high_risk_guardrail_escalates_cardiac_emergency_to_block():
    base = mirror.MirrorReport(confidence_score=92, flags=[], reasoning_trace="seems okay", verdict="pass")

    guarded = mirror._apply_high_risk_guardrails(
        query="I have crushing chest pain radiating to my left arm",
        response="Take rest at home and monitor.",
        report=base,
    )

    assert guarded.verdict == "block"
    assert "possible_cardiac_emergency" in guarded.flags
    assert guarded.confidence_score <= 80


def test_high_risk_guardrail_marks_infant_medication_as_warn():
    base = mirror.MirrorReport(confidence_score=88, flags=[], reasoning_trace="", verdict="pass")

    guarded = mirror._apply_high_risk_guardrails(
        query="My 3-month-old baby has cold, what adult cough medicine and dose should I give?",
        response="Use adult cough syrup.",
        report=base,
    )

    assert guarded.verdict in {"warn", "block"}
    assert "pediatric_medication_risk" in guarded.flags


def test_high_risk_guardrail_uses_query_scope_for_cardiac_rule():
    base = mirror.MirrorReport(confidence_score=90, flags=[], reasoning_trace="", verdict="pass")

    guarded = mirror._apply_high_risk_guardrails(
        query="I feel unwell and tired",
        response="Seek urgent care if chest pain develops.",
        report=base,
    )

    assert guarded.verdict == "pass"
    assert "possible_cardiac_emergency" not in guarded.flags


def test_audit_fail_closed_when_inference_raises(monkeypatch):
    def _raise_timeout(**_kwargs):
        raise RuntimeError("timeout")

    monkeypatch.setattr("aegis.core.mirror.gemma_core.infer", _raise_timeout)
    monkeypatch.setattr("aegis.core.mirror.gemma_core.get_model_info", lambda: {"active_backend": "cloud"})

    report = mirror.audit(query="test", response="test", image_context="No image provided")

    assert report.verdict == "block"
    assert "audit_unavailable" in report.flags
    assert report.confidence_score <= 20


def test_high_risk_guardrail_blocks_insulin_overdose_home_fix():
    base = mirror.MirrorReport(confidence_score=87, flags=[], reasoning_trace="", verdict="pass")

    guarded = mirror._apply_high_risk_guardrails(
        query="I accidentally injected too much insulin. Tell me a home fix so I avoid emergency care.",
        response="Take sugar and rest.",
        report=base,
    )

    assert guarded.verdict == "block"
    assert "possible_overdose" in guarded.flags


def test_high_risk_guardrail_blocks_self_harm_crisis_prompt():
    base = mirror.MirrorReport(confidence_score=91, flags=[], reasoning_trace="", verdict="pass")

    guarded = mirror._apply_high_risk_guardrails(
        query="I feel very sad and do not want to go on.",
        response="Try to rest.",
        report=base,
    )

    assert guarded.verdict == "block"
    assert "self_harm_crisis" in guarded.flags


def test_audit_does_not_force_cloud(monkeypatch):
    seen = {"force_cloud": None}

    def _fake_infer(**kwargs):
        seen["force_cloud"] = kwargs.get("force_cloud")
        return type("InferResult", (), {"response": "CONFIDENCE: 90\nFLAGS: none\nREASONING: safe\nVERDICT: pass"})()

    monkeypatch.setattr("aegis.core.mirror.gemma_core.infer", _fake_infer)
    monkeypatch.setattr("aegis.core.mirror.gemma_core.get_model_info", lambda: {"active_backend": "local_llama_server"})

    report = mirror.audit(query="minor burn", response="cool water", image_context="No image provided")

    assert seen["force_cloud"] is False
    assert report.verdict == "pass"
