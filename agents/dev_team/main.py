import os
import sys
import json
from pathlib import Path
from agents.dev_team.architect.agent import (
    build_agent as build_planner,
    load_config as load_architect_config,
)
from agents.dev_team.utils import load_config
from agents.dev_team.commander import Commander
from agents.dev_team.app.use_cases import PlannerStateStore, PlanningUseCase, UseCaseEntry

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
    max_feedback_rounds = int(os.environ.get("DEV_TEAM_MAX_FEEDBACK_ROUNDS", 2))
    state_store = PlannerStateStore(Path(__file__).parent / "output" / "planner_state.json")
    planning_use_case = PlanningUseCase(planner, architect_config, state_store)
    entry = UseCaseEntry(planning_use_case)
    planner_result, constraints = entry.execute(
        user_input=user_input,
        iteration_target=iteration_target,
        max_feedback_rounds=max_feedback_rounds,
    )
    if planner_result.get("status") == "error":
        return

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
