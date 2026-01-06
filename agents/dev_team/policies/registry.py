from __future__ import annotations

from typing import Any, Dict, List

from agents.dev_team.policies.interfaces import Policy


class PolicyRegistry:
    def __init__(self):
        self._policies: Dict[str, List[Policy]] = {}

    def register(self, kind: str, policy: Policy) -> None:
        self._policies.setdefault(kind, []).append(policy)

    def run_all(self, kind: str, **kwargs: Any) -> List[Any]:
        policies = self._policies.get(kind, [])
        return [policy.run(**kwargs) for policy in policies]

    def run_first(self, kind: str, **kwargs: Any) -> Any | None:
        policies = self._policies.get(kind, [])
        if not policies:
            return None
        return policies[0].run(**kwargs)
