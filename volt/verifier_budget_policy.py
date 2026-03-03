from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BudgetAction(str, Enum):
    VERIFY = "verify"
    REFINE = "refine"
    RESAMPLE = "resample"
    STOP = "stop"


@dataclass(frozen=True)
class VerifierBudgetPolicy:
    mode: str = "vbc_v1"
    scheduler_enabled: bool = True
    base_budget: int = 1
    hard_budget: int = 2
    chain_budget: int = 4
    op_chain_threshold: float = 0.5
    min_refine_steps_before_verify: int = 1

    def initial_limit(self, *, max_budget: int) -> int:
        if not self.scheduler_enabled:
            return max_budget
        return min(max_budget, max(1, int(self.base_budget)))

    def maybe_escalate(
        self,
        *,
        current_limit: int,
        max_budget: int,
        failure_kind: str,
    ) -> tuple[int, str]:
        if not self.scheduler_enabled:
            return current_limit, "scheduler_disabled"
        if current_limit >= max_budget:
            return current_limit, "max_budget_reached"

        target = current_limit
        reason = "no_change"
        if failure_kind == "hard_runtime":
            target = max(target, min(max_budget, max(1, int(self.hard_budget))))
            reason = "hard_runtime"
        elif failure_kind == "op_chain":
            target = max(target, min(max_budget, max(1, int(self.chain_budget))))
            reason = "op_chain"

        return target, reason

    def decide_action(
        self,
        *,
        budget_used: int,
        budget_limit: int,
        has_feedback: bool,
        steps_since_feedback: int,
        max_steps_per_feedback: int,
        failure_kind: str = "soft",
    ) -> BudgetAction:
        if budget_used >= budget_limit:
            return BudgetAction.STOP
        if not has_feedback:
            return BudgetAction.VERIFY
        if failure_kind == "hard_runtime":
            return BudgetAction.VERIFY
        if steps_since_feedback >= max_steps_per_feedback:
            return BudgetAction.VERIFY
        if steps_since_feedback < max(1, int(self.min_refine_steps_before_verify)):
            return BudgetAction.REFINE
        return BudgetAction.REFINE

