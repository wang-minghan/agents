from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class RequirementSpec:
    goal: str
    constraints: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    priority: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionInventory:
    source: str
    functions: List[Dict[str, Any]] = field(default_factory=list)
    generated_at: str = field(default_factory=_utcnow)


@dataclass
class TaskBacklog:
    tasks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Milestone:
    name: str
    round_index: int
    status: str = "pending"
    updated_at: str = field(default_factory=_utcnow)


@dataclass
class Artifact:
    path: str
    kind: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    kind: str
    status: str
    summary: str = ""
    required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "status": self.status,
            "summary": self.summary,
            "required": self.required,
            "metadata": dict(self.metadata),
        }


@dataclass
class GateRule:
    key: str
    required: bool


@dataclass
class GateDecision:
    status: str
    reasons: List[str]
    decided_at: str = field(default_factory=_utcnow)
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "decided_at": self.decided_at,
            "evidence": list(self.evidence),
        }


@dataclass
class GateEvidence:
    rule_key: str
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"rule_key": self.rule_key, "evidence": dict(self.evidence)}


@dataclass
class DomainEvent:
    name: str
    payload: Dict[str, Any]
    occurred_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "payload": dict(self.payload),
            "occurred_at": self.occurred_at,
        }
