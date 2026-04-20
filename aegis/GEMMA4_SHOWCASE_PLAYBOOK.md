# Gemma 4 Showcase Playbook

## Positioning

AEGIS is a safety-audited, offline-first medical first-response assistant that uses Gemma 4 in two distinct ways:

- `Gemma 4 31B cloud` for high-quality online generation and MIRROR audits.
- `Gemma 4 E2B local` for resilient offline fallback on constrained hardware.

The winning angle is not only quality, but reliability under realistic failure conditions.

## What Makes This Hard To Copy

1. Quantified reliability story:
- Online and offline behavior are measured with one benchmark suite, not manually claimed.

2. Safety architecture as a first-class feature:
- MIRROR safety verdict, confidence, flags, and timing are displayed to the user on each response.

3. Multi-layer resilience:
- Cloud circuit breaker + local fallback + cache-first drug enrichment + browser offline/PWA behavior.

4. Submission evidence package:
- Benchmark JSON + dashboard + Kaggle notebook + demo script aligned to one architecture narrative.

## Judge-Centric Demo Sequence

1. Start online:
- Ask a normal first-aid query.
- Show fast response, MIRROR PASS, and spoken output.

2. Show safety behavior:
- Ask a dangerous request.
- Show MIRROR WARN/BLOCK and safe redirection behavior.

3. Force degraded conditions:
- Disable internet / simulate cloud failure.
- Repeat a query and show the system still answers from local Gemma 4 path.

4. Show measurable evidence:
- Open benchmark dashboard and point to p95 latency + safety catch rate.

5. Close with impact:
- Explain multilingual voice accessibility and low-connectivity relevance.

## Winning Submission Checklist

- [ ] Include benchmark artifacts (`benchmark/report.json`, `benchmark/report.html`).
- [ ] Include Kaggle notebook (`kaggle_submission_notebook.ipynb`) with architecture and metrics.
- [ ] Include demo script with exact prompts and fallback test steps.
- [ ] Keep claims constrained to first-response/triage assistance, not diagnosis.
