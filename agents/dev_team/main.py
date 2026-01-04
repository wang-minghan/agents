import os
import sys
import json
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


def _load_planner_state(user_input: str) -> dict | None:
    if not STATE_PATH.exists():
        return None
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("user_input") == user_input and isinstance(data.get("planner_state"), dict):
        return data["planner_state"]
    return None


def _save_planner_state(user_input: str, planner_state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "user_input": user_input,
        "planner_state": planner_state,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _clear_planner_state() -> None:
    if STATE_PATH.exists():
        STATE_PATH.unlink()

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
    cached_state = _load_planner_state(user_input)
    if cached_state:
        print("⚠️ 检测到上次规划状态，尝试恢复。")
        input_payload["planner_state"] = cached_state
    planner_result = planner.invoke(input_payload)
    if planner_result.get("status") == "needs_feedback":
        print("⚠️ 未提供补充信息，使用AI默认假设继续规划。")
        if planner_result.get("planner_state"):
            _save_planner_state(user_input, planner_result["planner_state"])
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
    if planner_result.get("status") == "completed":
        _clear_planner_state()

    if planner_result.get("status") != "completed":
        print("❌ Task Planner 未能完成规划，请检查输入或配置。")
        return

    # 2. 启动动态 Multi-Agent 团队
    print("\n[Step 2] 启动动态 Multi-Agent 团队进行协作开发...")
    
    # 加载配置
    dev_team_config = load_config()
    if iteration_target:
        dev_team_config["iteration_target"] = iteration_target
    
    commander = Commander(dev_team_config)
    commander.initialize_team(planner_result)
    final_results = commander.run_collaboration(max_rounds=2)

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
