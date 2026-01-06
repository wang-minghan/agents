from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from agents.dev_team.architect.agent import suggest_default_assumptions


def _constraints_signature(constraints: dict | None) -> str:
    if not constraints:
        return ""
    try:
        payload = json.dumps(constraints, sort_keys=True, ensure_ascii=False)
    except TypeError:
        payload = str(constraints)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _coerce_json(payload: Any) -> Any:
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return payload
    return payload


def _force_planner_completion(planner_result: dict) -> dict:
    planner_state = planner_result.get("planner_state") or {}
    roles = planner_state.get("roles") or planner_result.get("roles") or []
    current_jds = planner_state.get("current_jds") or {}
    requirements = planner_state.get("requirements") or planner_result.get("requirements") or {}
    tasks = planner_state.get("tasks") or planner_result.get("tasks") or []
    final_jds = []
    for role in roles:
        role_name = role.get("role_name", "Unknown")
        jd_content = current_jds.get(role_name) or role.get("initial_jd", "")
        jd_payload = _coerce_json(jd_content)
        if isinstance(jd_payload, dict):
            jd_payload.setdefault("role_name", role_name)
            final_jds.append(jd_payload)
        else:
            final_jds.append({"role_name": role_name, "content": jd_content})
    return {
        "status": "completed",
        "requirements": requirements,
        "tasks": tasks,
        "final_jds": final_jds,
        "validation_result": planner_result.get("validation_result"),
        "forced_completion": True,
    }


@dataclass
class PlannerStateStore:
    path: Path

    def load(self, user_input: str, constraints: dict | None) -> dict | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if data.get("constraints_sig") != _constraints_signature(constraints):
            return None
        if data.get("user_input") == user_input and isinstance(data.get("planner_state"), dict):
            return data["planner_state"]
        return None

    def save(self, user_input: str, planner_state: dict, constraints: dict | None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "user_input": user_input,
            "constraints_sig": _constraints_signature(constraints),
            "planner_state": planner_state,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


class PlanningUseCase:
    def __init__(self, planner: Any, architect_config: dict, state_store: PlannerStateStore):
        self.planner = planner
        self.architect_config = architect_config
        self.state_store = state_store

    def run(
        self,
        *,
        user_input: str,
        constraints: dict | None,
        max_feedback_rounds: int,
    ) -> dict:
        input_payload: Dict[str, Any] = {"user_input": user_input}
        if constraints:
            input_payload["constraints"] = constraints
        cached_state = self.state_store.load(user_input, constraints)
        if cached_state:
            print("⚠️ 检测到上次规划状态，尝试恢复。")
            input_payload["planner_state"] = cached_state
        planner_result = self.planner.invoke(input_payload)
        feedback_rounds = 0
        while planner_result.get("status") == "needs_feedback" and feedback_rounds < max_feedback_rounds:
            print("⚠️ 未提供补充信息，使用AI默认假设继续规划。")
            if planner_result.get("planner_state"):
                self.state_store.save(user_input, planner_result["planner_state"], constraints)
            ai_feedback = suggest_default_assumptions(
                self.architect_config,
                planner_result,
                user_input,
                constraints or {},
            )
            input_payload = {
                "user_input": user_input,
                "planner_state": planner_result.get("planner_state", {}),
                "user_feedback": ai_feedback,
            }
            if constraints:
                input_payload["constraints"] = constraints
            planner_result = self.planner.invoke(input_payload)
            feedback_rounds += 1
        if planner_result.get("status") != "completed":
            if planner_result.get("status") == "needs_feedback":
                print("⚠️ 规划多轮未通过校验，转为最佳努力结果继续执行。")
                planner_result = _force_planner_completion(planner_result)
            else:
                print("❌ Task Planner 未能完成规划，请检查输入或配置。")
                return {"status": "error", "error": "planner_failed"}
        self.state_store.clear()
        return planner_result


class UseCaseEntry:
    def __init__(self, planning_use_case: PlanningUseCase):
        self.planning_use_case = planning_use_case

    def execute(
        self,
        *,
        user_input: str,
        iteration_target: Optional[str],
        max_feedback_rounds: int,
    ) -> Tuple[dict, dict]:
        constraints: Dict[str, Any] = {}
        if iteration_target:
            constraints["existing_project"] = iteration_target
            evidence_dir = Path(iteration_target) / "evidence" / "ui"
            if evidence_dir.exists():
                baseline = list(evidence_dir.glob("design_baseline.*"))
                implementation = list(evidence_dir.glob("implementation.*"))
                if baseline:
                    constraints["design_baseline"] = str(baseline[0])
                if implementation:
                    constraints["implementation_snapshot"] = str(implementation[0])
        planner_result = self.planning_use_case.run(
            user_input=user_input,
            constraints=constraints or None,
            max_feedback_rounds=max_feedback_rounds,
        )
        return planner_result, constraints
