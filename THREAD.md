# Twitter Thread Draft (VOLT)

1/ We built **VOLT (Verify Once, Learn Twice)**: a drop-in controller for costly-feedback RL loops.  
Instead of fixed verifier schedules, VOLT adaptively decides when to `verify`, `refine`, or `stop`.

2/ Motivation: verifier calls are often the expensive part of post-training (tests, judges, audits, human checks).  
Fixed-k spends too much on easy tasks and too little on hard tasks.

3/ VOLT policy:
- start with low budget (`b=1` behavior),
- escalate only on hard failures/op-chain failures,
- keep strict accounting:
  - budgeted cost = `feedback_calls`
  - full external cost = `test_calls`

4/ What we found in our KernelBench-style budgeted setting:
- strongest gains at low budget (`b=1,2`)
- explicit high-budget crossover exists (`b=4` can favor resampling)
- we report crossover directly (not hidden)

5/ This is important because many real systems are low-budget by design:
- API-rate-limited verifiers
- expensive CI/test suites
- human-in-the-loop review pipelines

6/ We do **not** claim apples-to-apples official KernelBench-L1 pass@k SOTA.  
We claim a strong efficiency/accuracy tradeoff in budgeted-feedback regimes.

7/ VOLT is framework-agnostic:
- use with your own loop (`examples/min_loop.py`)
- integrate into Prime Verifiers control loops (`examples/verifiers_integration.md`)
- run ttRL+Tinker smoke in one command (`examples/ttrl_tinker.md`)

8/ Repo: `verify-once-learn-twice`  
Package path: `volt/verifier_budget_policy.py`  
If you already have a rollout loop, integration is ~20 lines.

9/ Next: broader multi-env benchmark and public leaderboard on:
- success@B=1,2
- AUC over budget
- full-cost curves (`test_calls`)

