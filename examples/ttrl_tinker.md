# Using VOLT in ttRL (with Tinker)

This is the minimal reproducible path used in our local runs.

## 1) Set API key

```bash
export TINKER_API_KEY=...
```

## 2) Run smoke experiment

```bash
cd /Users/ryanmathieu/ttRL
PYTHONPATH=. ./scripts/vbc_end_to_end.sh data/ablation_vbc_smoke5.json
```

## 3) Inspect outputs

- `runs/verify_vbc_smoke5/vbc_smoke_report.md`
- `runs/verify_vbc_smoke5/leaderboard.json`

## 4) Key config knobs

- `verifier_budget_policy`: `legacy` or `vbc_v1`
- `adaptive_budget_scheduler`: `true/false`
- `adaptive_budget_base`, `adaptive_budget_hard`, `adaptive_budget_chain`
- `feedback_budget`

