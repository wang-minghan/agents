from __future__ import annotations

from typing import Callable, Dict, List, Optional

from agents.dev_team.app.state_machine import MilestoneStateMachine
from agents.dev_team.app.workflow import WorkflowRouter
from agents.dev_team.policies.registry import PolicyRegistry
from agents.dev_team.capabilities import CapabilityRegistry


class PlanTaskCoordinator:
    def __init__(
        self,
        router: WorkflowRouter,
        fsm: MilestoneStateMachine,
        policy_registry: PolicyRegistry,
        capability_registry: CapabilityRegistry,
        event_sink: Optional[Callable[[str, Dict[str, str]], None]] = None,
    ):
        self.router = router
        self.fsm = fsm
        self.policy_registry = policy_registry
        self.capability_registry = capability_registry
        self._event_sink = event_sink

    def build_round_plan(
        self,
        *,
        round_num: int,
        enable_consensus: bool,
        enable_cross_validation: bool,
        has_qa: bool,
    ) -> List[str]:
        plan = self.router.phases(
            enable_consensus=enable_consensus,
            enable_cross_validation=enable_cross_validation,
            has_qa=has_qa,
        )
        self._emit_event("round_plan_built", {"round": str(round_num), "plan": ",".join(plan)})
        return plan

    def transition(self, event: str, *, round_num: int) -> str:
        transition = self.fsm.transition(event)
        self._emit_event(
            "milestone_transition",
            {
                "round": str(round_num),
                "from": transition.from_state,
                "event": transition.event,
                "to": transition.to_state,
            },
        )
        return transition.to_state

    def _emit_event(self, name: str, payload: Dict[str, str]) -> None:
        if self._event_sink:
            self._event_sink(name, payload)
