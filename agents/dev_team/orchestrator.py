
import json
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

from agents.dev_team.role_agent import RoleAgent
from agents.dev_team.memory import SharedMemoryStore
from agents.dev_team.interfaces import CodeExecutor, Agent
from agents.dev_team.execution import SafeExecutor, LocalUnsafeExecutor
from agents.common import save_files_from_content

# Define a factory type for creating agents
AgentFactory = Callable[[Dict[str, Any], Dict[str, Any], SharedMemoryStore, Path], Agent]


def default_agent_factory(jd: Dict[str, Any], config: Dict[str, Any], memory: SharedMemoryStore, output_dir: Path) -> Agent:
    return RoleAgent(jd, config, memory, output_dir)


class Orchestrator:
    def __init__(
        self, 
        config: Dict[str, Any], 
        output_dir: str = None, 
        code_executor: Optional[CodeExecutor] = None,
        agent_factory: AgentFactory = default_agent_factory
    ):
        self.config = config
        self.shared_memory = SharedMemoryStore(config)
        self.agent_factory = agent_factory
        
        # Determine output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Fallback but respect config or use safe default
            # Use absolute path relative to project root if possible, or CWD
            default_path = config.get("output_dir", "agents/dev_team/output/codebase")
            self.output_dir = Path(default_path)
            
        if not self.output_dir.is_absolute():
             self.output_dir = self.output_dir.resolve()

        print(f"Directory: {self.output_dir}")

        # Code Executor Injection
        if code_executor:
             self.code_executor = code_executor
        else:
            # Decide based on config if allowed
            if self.config.get("allow_unsafe_execution", False):
                 print("⚠️ WARNING: Unsafe execution enabled via config.")
                 self.code_executor = LocalUnsafeExecutor()
            else:
                 self.code_executor = SafeExecutor()

        # 将 requirements 注入全局上下文
        self.shared_memory.global_context["requirements"] = ""

        self.agents: List[Agent] = []
        self.qa_agent: Optional[Agent] = None

    def initialize_team(self, planner_result: Dict[str, Any]):
        """
        根据 Planner 的结果动态初始化团队
        """
        final_jds = planner_result.get("final_jds", [])
        requirements = planner_result.get("requirements", {})
        
        # 存储全局需求
        self.shared_memory.global_context["requirements"] = json.dumps(requirements, ensure_ascii=False)
        
        print(f"\n>>> 正在组建开发团队，检测到 {len(final_jds)} 个角色需求...")
        
        for jd in final_jds:
            role_name = jd.get("role_name", "Unknown")
            # Use factory to create agent
            agent = self.agent_factory(jd, self.config, self.shared_memory, self.output_dir)
            
            # Check if QA based on role type or name
            is_qa = agent.role_type == "QA" or "QA" in role_name or "Test" in role_name
            
            if is_qa:
                self.qa_agent = agent
                print(f"    └── 已指派 QA 专家: {role_name}")
            else:
                self.agents.append(agent)
                print(f"    └── 已指派工程师: {role_name}")

    def run_collaboration(self, max_rounds: int = 3):
        """
        运行多轮协作流程，引入测试驱动的自我修正
        """
        if not self.agents:
            print("❌ 错误: 团队未初始化或没有工程师角色。")
            return []

        print(f"\n🚀 启动 TDD 协作流程 (最大轮次: {max_rounds})...")
        collab_cfg = self.config.get("collaboration", {})
        force_qa_on_success = collab_cfg.get("force_qa_on_success", False)
        post_success_qa_rounds = int(collab_cfg.get("post_success_qa_rounds", 0))

        for round_num in range(1, max_rounds + 1):
            print(f"\n🔄 --- 第 {round_num} 轮迭代 ---")

            # 1. 工程师开发/修复
            for agent in self.agents:
                try:
                    agent.run() 
                    # Save files from agent output
                    output_content = self.shared_memory.get_all_outputs()[agent.role_name][-1]
                    saved_files = save_files_from_content(output_content, self.output_dir)
                    if saved_files:
                        self.shared_memory.add_saved_files(saved_files)
                    print(f"    ✅ [{agent.role_name}] 完成工作")
                except Exception as e:
                    print(f"    ❌ [{agent.role_name}] 执行出错: {str(e)}")

            # 2. 自动化测试阶段 (Execution Feedback)
            print(f"    🧪 [System] 正在执行自动化测试/语法检查...")
            
            # Use injected executor
            test_results = self.code_executor.run_tests(str(self.output_dir))
            
            self.shared_memory.global_context["latest_test_results"] = test_results
            summary_lines = test_results.splitlines()
            summary = summary_lines[0] if summary_lines else "No test output."
            print(f"    📋 测试结果摘要: {summary}")
            
            if "FAIL" not in test_results and "Error" not in test_results and "SKIPPED" not in test_results:
                print("    ✨ 自动化测试全部通过！")
                if self.qa_agent and (force_qa_on_success or post_success_qa_rounds > 0):
                    qa_rounds = post_success_qa_rounds if post_success_qa_rounds > 0 else 1
                    for _ in range(qa_rounds):
                        print(f"\n🔍 [QA: {self.qa_agent.role_name}] 进行通过后的审查...")
                        qa_feedback = self.qa_agent.run()
                        self.shared_memory.add_qa_feedback({
                            "round": round_num,
                            "test_status": "Passed",
                            "feedback": qa_feedback
                        })
                        print("    📝 QA 反馈已记录")
                print("    >>> 提前结束协作循环。")
                break

            # 3. QA 进行全局审查与反馈
            if self.qa_agent:
                print(f"\n🔍 [QA: {self.qa_agent.role_name}] 正在进行代码审查与测试分析...")
                qa_feedback = self.qa_agent.run()
                
                self.shared_memory.add_qa_feedback({
                    "round": round_num,
                    "test_status": "Passed" if "FAIL" not in test_results else "Failed",
                    "feedback": qa_feedback
                })
                print(f"    📝 QA 反馈已记录")

        return self.shared_memory.get_all_outputs()
