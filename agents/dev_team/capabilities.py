from __future__ import annotations

from typing import Dict, List

from agents.dev_team.interfaces import Agent

CAP_TASK_DECOMPOSABLE = "TaskDecomposable"
CAP_ARTIFACT_PRODUCIBLE = "ArtifactProducible"
CAP_RESULT_VERIFIABLE = "ResultVerifiable"
CAP_CONTEXT_CONTRIBUTING = "ContextContributing"


def default_capabilities_for_role(role_name: str, role_type: str | None) -> List[str]:
    upper_name = (role_name or "").upper()
    upper_type = (role_type or "").upper()
    if upper_type == "QA" or "QA" in upper_name or "TEST" in upper_name:
        return [CAP_RESULT_VERIFIABLE, CAP_CONTEXT_CONTRIBUTING]
    return [CAP_TASK_DECOMPOSABLE, CAP_ARTIFACT_PRODUCIBLE, CAP_CONTEXT_CONTRIBUTING]


class CapabilityRegistry:
    def __init__(self):
        self._registry: Dict[str, List[Agent]] = {}

    def register(self, agent: Agent, capabilities: List[str]) -> None:
        for cap in capabilities:
            self._registry.setdefault(cap, []).append(agent)

    def resolve(self, capability: str) -> List[Agent]:
        return list(self._registry.get(capability, []))
