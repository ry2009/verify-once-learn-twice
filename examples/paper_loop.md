# Paper Loop (Minimal Algorithm)

This is the core loop we used:

1. Sample candidate from `pi(y|x)`
2. Verify once to get rich feedback `f` (costly call)
3. Distill from teacher `pi(y|x,f)` into student
4. Re-sample from `pi(y|x)`, judge whether feedback is satisfied
5. Use VOLT to decide whether to spend another verifier call or keep refining
6. Stop when satisfied or budget exhausted

## Pseudocode

```python
while not done:
    if no_feedback_yet:
        verify()  # costly
        continue

    train_step_on_teacher_feedback()
    candidate = sample_student()
    judge_pass = judge(candidate, feedback)

    if judge_pass:
        break

    action = volt.decide_action(...)
    if action == "verify":
        verify()  # costly
    elif action == "refine":
        continue
    else:
        break
```

## Accounting

- `feedback_calls`: budgeted channel (primary plot axis)
- `test_calls`: full external cost (guard against hidden spend)

