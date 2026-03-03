# VOLT: Verify Once, Learn Twice

VOLT is a drop-in **Verifier Budget Controller (VBC)** for costly-feedback RL loops.

It decides when to:
- `verify` (spend an external verifier call),
- `refine` (keep learning without spending a new verifier call),
- `stop` (budget exhausted / no further spend).

## Why this exists

In many post-training loops, verifier calls are the expensive bottleneck.  
Fixed schedules (`verify every step`, fixed-`k`) waste calls on easy tasks and under-allocate on hard ones.

VOLT makes verifier spend adaptive and budget-aware.

## Core claim (scoped)

On our KernelBench-style **budgeted feedback** setup:
- strongest gains are in low-budget regimes (`b=1,2`),
- high-budget crossover is real and explicitly reported (`b=4` can favor resampling baselines),
- we report both budgeted cost (`feedback_calls`) and full external cost (`test_calls`).

This is **not** an apples-to-apples official KernelBench-L1 pass@k claim.

## Install

```bash
pip install -e .
```

## Quick use (framework-agnostic)

```python
from volt.verifier_budget_policy import VerifierBudgetPolicy

policy = VerifierBudgetPolicy(
    mode="vbc_v1",
    scheduler_enabled=True,
    base_budget=1,
    hard_budget=2,
    chain_budget=4,
    op_chain_threshold=0.5,
)

budget_limit = policy.initial_limit(max_budget=4)
```

See `examples/min_loop.py` for a minimal loop.
See `examples/paper_loop.md` for the minimal research loop.

## Use with Verifiers

VOLT is a policy layer for your rollout loop.  
It does not replace `prime eval run`; it controls *when* your loop spends verifier calls.

See `examples/verifiers_integration.md`.

## Reproducing our local smoke run (from ttRL)

If you are in the `ttRL` workspace:

```bash
PYTHONPATH=. ./scripts/vbc_end_to_end.sh data/ablation_vbc_smoke5.json
```

Expected report path:
- `runs/verify_vbc_smoke5/vbc_smoke_report.md`

## Suggested citation text

> VOLT (Verify Once, Learn Twice) is a verifier-budget controller that adaptively schedules verifier calls under strict feedback budgets for post-training RL loops.

## Files you likely want in a clean public repo

- `volt/verifier_budget_policy.py` (core controller)
- `examples/min_loop.py` (without Tinker/Verifiers)
- `examples/paper_loop.md` (paper loop pseudocode)
- `examples/verifiers_integration.md` (with Verifiers)
- `examples/ttrl_tinker.md` (with Tinker)
- `THREAD.md` (ready Twitter thread draft)
