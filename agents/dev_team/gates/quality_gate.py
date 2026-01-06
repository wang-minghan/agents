from __future__ import annotations

from typing import Dict, List

from agents.dev_team.domain.models import Evidence, GateDecision, GateRule


class QualityGate:
    def __init__(self, rules: List[GateRule]):
        self.rules = rules

    def evaluate(self, evidence: List[Evidence]) -> GateDecision:
        required_map = {rule.key: rule.required for rule in self.rules}
        reasons: List[str] = []
        evidence_payload = [item.to_dict() for item in evidence]
        for item in evidence:
            if not required_map.get(item.kind, item.required):
                continue
            if item.status != "passed":
                reasons.append(item.kind)
        status = "passed" if not reasons else "failed"
        return GateDecision(status=status, reasons=reasons, evidence=evidence_payload)
