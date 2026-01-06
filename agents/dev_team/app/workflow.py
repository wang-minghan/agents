from __future__ import annotations

from typing import List


class WorkflowRouter:
    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def phases(
        self,
        *,
        enable_consensus: bool,
        enable_cross_validation: bool,
        has_qa: bool,
    ) -> List[str]:
        phases = ["agents"]
        if enable_consensus:
            phases.append("consensus")
        phases.append("tests")
        if enable_cross_validation:
            phases.append("cross_validation")
        if has_qa:
            phases.append("qa")
        phases.append("gate")
        return phases
