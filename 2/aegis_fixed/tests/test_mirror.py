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
    assert "first-aid guidance only" in mirror.apply_verdict(original, warn_report)
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
