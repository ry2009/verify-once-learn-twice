# Using VOLT with Prime Verifiers

VOLT is a policy layer. Keep your existing Verifiers environment and model calls; add VOLT to decide when to spend verifier calls.

## Pattern

1. Keep your normal rollout/refine loop.
2. Track:
   - `feedback_calls` (budgeted channel),
   - `test_calls` (full external cost),
   - `steps_since_feedback`,
   - last failure type (`soft`, `hard_runtime`, `op_chain`).
3. Ask VOLT for the next action.

## Skeleton

```python
from volt import VerifierBudgetPolicy

policy = VerifierBudgetPolicy(
    mode="vbc_v1",
    scheduler_enabled=True,
    base_budget=1,
    hard_budget=2,
    chain_budget=4,
)

max_budget = 4
budget_limit = policy.initial_limit(max_budget=max_budget)
feedback_calls = 0
test_calls = 0
steps_since_feedback = 0
last_feedback = ""

while True:
    action = policy.decide_action(
        budget_used=feedback_calls,
        budget_limit=budget_limit,
        has_feedback=bool(last_feedback),
        steps_since_feedback=steps_since_feedback,
        max_steps_per_feedback=8,
        failure_kind="soft",
    )

    if action.value == "stop":
        break

    if action.value == "verify":
        # call your verifier/environment reward path
        # passed, feedback = env.verify(candidate)
        feedback_calls += 1
        test_calls += 1
        steps_since_feedback = 0

        # optional escalation
        budget_limit, _ = policy.maybe_escalate(
            current_limit=budget_limit,
            max_budget=max_budget,
            failure_kind="hard_runtime",  # your classifier output
        )
    else:
        # refine step (no new verifier spend)
        steps_since_feedback += 1
```

## What to report

- Success vs `feedback_calls`
- Success vs `test_calls`
- Crossover budget (where fixed schedules catch up)

