from __future__ import annotations

from typing import Any


class ReviewPolicyImpl:
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator

    def run(self, **kwargs: Any) -> Any:
        return self.orchestrator._ensure_review_artifacts()


class ConsensusPolicyImpl:
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator

    def run(self, **kwargs: Any) -> Any:
        round_num = kwargs.get("round_num")
        return self.orchestrator._run_consensus_phase(round_num)


class VerificationPolicyImpl:
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator

    def run(self, **kwargs: Any) -> Any:
        return self.orchestrator._run_cross_validation_phase()
