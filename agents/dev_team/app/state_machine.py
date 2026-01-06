from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StateTransition:
    from_state: str
    event: str
    to_state: str


class MilestoneStateMachine:
    def __init__(self, initial_state: str = "planning"):
        self.state = initial_state
        self._transitions = {
            ("planning", "start_round"): "executing",
            ("executing", "agents_done"): "verifying",
            ("verifying", "tests_done"): "gating",
            ("gating", "gating"): "gating",
            ("gating", "gate_passed"): "completed",
            ("gating", "gate_failed"): "failed",
            ("failed", "retry"): "executing",
            ("completed", "next_round"): "executing",
        }

    def transition(self, event: str) -> StateTransition:
        key = (self.state, event)
        next_state = self._transitions.get(key, self.state)
        transition = StateTransition(self.state, event, next_state)
        self.state = next_state
        return transition
