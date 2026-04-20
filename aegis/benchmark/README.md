# AEGIS Benchmark Artifacts

This folder contains reproducible benchmark tooling and generated evidence for submission.

## Generate Reports

From the workspace root:

```powershell
& "d:/Gemma 4/.venv/Scripts/python.exe" aegis/benchmark/run_benchmark.py --base-url http://127.0.0.1:5000 --mode all --out aegis/benchmark/report.json --html-out aegis/benchmark/report.html
```

## Output Files

- `report.json`: machine-readable benchmark payload with metadata, aggregate report, and per-scenario rows.
- `report.html`: judge-friendly dashboard with latency/safety summary and scenario-level outcomes.

## Submission Guidance

- Keep `mode=all` for final numbers.
- Re-run after each major change to keep evidence current.
- Include both files in the hackathon submission package and Kaggle notebook narrative.
