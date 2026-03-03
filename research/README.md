# Research Stack (ttRL + Scripts)

This folder mirrors the core experiment code used for VOLT experiments.

## Layout

- `ttRL/`: training loop, policy wiring, feedback handling, evaluation
- `scripts/`: launchers, ablations, closeout, plots, reporting
- `tests/`: regression/unit tests
- `data/`: minimal specs for smoke runs
- `artifacts/`: selected figure outputs
- `runs/`: selected closeout summaries/tables

## Quick local test

```bash
cd /path/to/repo
PYTHONPATH=. pytest -q research/tests/test_verifier_budget_policy.py
```

## Smoke run (requires Tinker key)

```bash
cd /path/to/repo
export TINKER_API_KEY=...
PYTHONPATH=. python research/scripts/run_ablation.py --spec research/data/ablation_vbc_smoke5.json
```

