from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List

from agents.dev_team.interfaces import Agent


class ExecutionService:
    def __init__(self, agent_runner: Callable[[Agent], dict]):
        self._agent_runner = agent_runner

    def run_agents(self, agents: List[Agent]) -> List[dict]:
        if not agents:
            return []
        max_workers = min(4, max(1, len(agents)))
        results: List[dict] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._agent_runner, agent) for agent in agents]
            for future in as_completed(futures):
                results.append(future.result())
        return results
