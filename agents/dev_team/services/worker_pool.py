from __future__ import annotations

from typing import List, Optional

from agents.dev_team.capabilities import default_capabilities_for_role
from agents.dev_team.interfaces import Agent
from agents.dev_team.capabilities import CapabilityRegistry


class WorkerPool:
    def __init__(self, registry: CapabilityRegistry):
        self._registry = registry
        self._agents: List[Agent] = []
        self._qa_agents: List[Agent] = []
        self._final_approver: Optional[Agent] = None

    def add_agent(
        self,
        agent: Agent,
        role_name: str,
        *,
        is_qa: bool,
        is_final_approver: bool,
    ) -> None:
        if is_qa:
            self._qa_agents.append(agent)
        elif is_final_approver:
            self._final_approver = agent
        else:
            self._agents.append(agent)
        capabilities = default_capabilities_for_role(role_name, getattr(agent, "role_type", None))
        self._registry.register(agent, capabilities)

    def agents(self) -> List[Agent]:
        return list(self._agents)

    def qa_agents(self) -> List[Agent]:
        return list(self._qa_agents)

    def final_approver(self) -> Optional[Agent]:
        return self._final_approver

    def resolve(self, capability: str) -> List[Agent]:
        return self._registry.resolve(capability)
