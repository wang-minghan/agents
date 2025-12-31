import sys
import json
from agents.task_planner.agent import build_agent as build_planner
from agents.dev_team.utils import load_config
from agents.dev_team.orchestrator import Orchestrator

def main():
    user_input = "我想做一个支持高并发的秒杀系统，需要考虑到缓存击穿、雪崩以及分布式锁的实现。"
    if len(sys.argv) > 1:
        user_input = sys.argv[1]

    print(f"🚀 启动任务: {user_input}")

    # 1. 运行 Task Planner 获取 JD
    print("\n[Step 1] 运行 Task Planner 规划角色...")
    planner = build_planner()
    planner_result = planner.invoke({"user_input": user_input})

    if planner_result.get("status") != "completed":
        print("❌ Task Planner 未能完成规划，请检查输入或配置。")
        return

    # 2. 启动动态 Multi-Agent 团队
    print("\n[Step 2] 启动动态 Multi-Agent 团队进行协作开发...")
    
    # 加载配置
    dev_team_config = load_config()
    
    # 初始化协调器
    orchestrator = Orchestrator(dev_team_config)
    
    # 动态组建团队
    orchestrator.initialize_team(planner_result)
    
    # 运行多轮协作
    final_results = orchestrator.run_collaboration(max_rounds=2)

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

if __name__ == "__main__":
    main()
