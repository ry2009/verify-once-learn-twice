from __future__ import annotations

from volt.verifier_budget_policy import VerifierBudgetPolicy


def classify_failure(feedback: str) -> str:
    low = feedback.lower()
    if "shape mismatch" in low or "runtimeerror" in low:
        return "hard_runtime"
    return "soft"


def fake_verify(candidate: str) -> tuple[bool, str]:
    if "x + 1" in candidate:
        return True, ""
    return False, "RuntimeError: shape mismatch"


def main() -> None:
    policy = VerifierBudgetPolicy(
        mode="vbc_v1",
        scheduler_enabled=True,
        base_budget=1,
        hard_budget=2,
        chain_budget=4,
    )
    max_budget = 2
    limit = policy.initial_limit(max_budget=max_budget)
    used = 0
    feedback = ""
    candidate = "return x"

    while True:
        action = policy.decide_action(
            budget_used=used,
            budget_limit=limit,
            has_feedback=bool(feedback),
            steps_since_feedback=0,
            max_steps_per_feedback=8,
            failure_kind=classify_failure(feedback) if feedback else "soft",
        )
        print(f"action={action.value} used={used}/{limit}")
        if action.value == "stop":
            break
        if action.value == "verify":
            passed, feedback = fake_verify(candidate)
            used += 1
            print(f"verify -> passed={passed} feedback={feedback!r}")
            if passed:
                print("done")
                break
            limit, reason = policy.maybe_escalate(
                current_limit=limit,
                max_budget=max_budget,
                failure_kind=classify_failure(feedback),
            )
            print(f"escalate -> limit={limit} reason={reason}")
            candidate = "return x + 1"
        else:
            candidate = "return x + 1"


if __name__ == "__main__":
    main()

