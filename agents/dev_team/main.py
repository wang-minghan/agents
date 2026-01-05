import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from agents.dev_team.architect.agent import (
    build_agent as build_planner,
    load_config as load_architect_config,
    suggest_default_assumptions,
)
from agents.dev_team.utils import load_config
from agents.dev_team.commander import Commander

STATE_PATH = Path(__file__).parent / "output" / "planner_state.json"


def _constraints_signature(constraints: dict | None) -> str:
    if not constraints:
        return ""
    try:
        payload = json.dumps(constraints, sort_keys=True, ensure_ascii=False)
    except TypeError:
        payload = str(constraints)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_planner_state(user_input: str, constraints: dict | None) -> dict | None:
    if not STATE_PATH.exists():
        return None
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("constraints_sig") != _constraints_signature(constraints):
        return None
    if data.get("user_input") == user_input and isinstance(data.get("planner_state"), dict):
        return data["planner_state"]
    return None


def _save_planner_state(user_input: str, planner_state: dict, constraints: dict | None) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "user_input": user_input,
        "constraints_sig": _constraints_signature(constraints),
        "planner_state": planner_state,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _clear_planner_state() -> None:
    if STATE_PATH.exists():
        STATE_PATH.unlink()

def _coerce_json(payload):
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

def main():
    user_input = "我想做一个支持高并发的秒杀系统，需要考虑到缓存击穿、雪崩以及分布式锁的实现。"
    if len(sys.argv) > 1:
        user_input = sys.argv[1]

    print(f"🚀 启动任务: {user_input}")

    # 1. 运行 Task Planner 获取 JD
    print("\n[Step 1] 运行 Task Planner 规划角色...")
    planner = build_planner()
    architect_config = load_architect_config()
    iteration_target = os.environ.get("DEV_TEAM_ITERATION_TARGET")
    input_payload = {"user_input": user_input}
    constraints = {}
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
    if constraints:
        input_payload["constraints"] = constraints
    cached_state = _load_planner_state(user_input, constraints)
    if cached_state:
        print("⚠️ 检测到上次规划状态，尝试恢复。")
        input_payload["planner_state"] = cached_state
    planner_result = planner.invoke(input_payload)
    feedback_rounds = 0
    max_feedback_rounds = int(os.environ.get("DEV_TEAM_MAX_FEEDBACK_ROUNDS", 2))
    while planner_result.get("status") == "needs_feedback" and feedback_rounds < max_feedback_rounds:
        print("⚠️ 未提供补充信息，使用AI默认假设继续规划。")
        if planner_result.get("planner_state"):
            _save_planner_state(user_input, planner_result["planner_state"], constraints)
        ai_feedback = suggest_default_assumptions(
            architect_config,
            planner_result,
            user_input,
            constraints,
        )
        input_payload = {
            "user_input": user_input,
            "planner_state": planner_result.get("planner_state", {}),
            "user_feedback": ai_feedback,
        }
        if constraints:
            input_payload["constraints"] = constraints
        planner_result = planner.invoke(input_payload)
        feedback_rounds += 1
    if planner_result.get("status") != "completed":
        if planner_result.get("status") == "needs_feedback":
            print("⚠️ 规划多轮未通过校验，转为最佳努力结果继续执行。")
            planner_result = _force_planner_completion(planner_result)
        else:
            print("❌ Task Planner 未能完成规划，请检查输入或配置。")
            return
    _clear_planner_state()

    # 2. 启动动态 Multi-Agent 团队
    print("\n[Step 2] 启动动态 Multi-Agent 团队进行协作开发...")
    
    # 加载配置
    dev_team_config = load_config()
    dev_team_config["session_key"] = user_input
    if iteration_target:
        dev_team_config["iteration_target"] = iteration_target
    
    commander = Commander(dev_team_config)
    commander.initialize_team(planner_result)
    max_rounds = int(os.environ.get("DEV_TEAM_MAX_ROUNDS", dev_team_config.get("max_rounds", 5)))
    final_results = commander.run_collaboration(max_rounds=max_rounds)

    # 3. 输出最终结果
    print("\n" + "="*50)
    print("✅ 团队协作已完成")
    print("="*50 + "\n")

    # 保存结果到文件
    output_file = "agents/dev_team/output_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    print(f"\n📂 详细结果已保存至: {output_file}")
    print(f"📂 生成的代码位于: agents/dev_team/output/codebase/")
    if isinstance(final_results, dict):
        status = final_results.get("status")
        report = final_results.get("report", {})
        report_path = report.get("report_path")
        if status:
            print(f"📌 协作状态: {status}")
        if report_path:
            print(f"📄 协作报告: {report_path}")

if __name__ == "__main__":
    main()
