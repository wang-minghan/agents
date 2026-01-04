
import json
from datetime import datetime, timezone
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
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
        self.run_reports: List[Dict[str, Any]] = []
        self.report_enabled = bool(self.config.get("report", {}).get("enabled", True))
        self.report_path = self._resolve_report_path()

    def _resolve_report_path(self) -> Path:
        report_path = self.config.get("report", {}).get("output_path")
        if report_path:
            report_path = Path(report_path)
            if not report_path.is_absolute():
                report_path = self.output_dir / report_path
        else:
            report_path = self.output_dir / "collaboration_report.json"
        return report_path

    @staticmethod
    def _utcnow() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _truncate(text: str, limit: int = 1000) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...[Truncated]..."

    @staticmethod
    def _classify_test_result(output: str) -> str:
        if not output:
            return "unknown"
        normalized = output.strip().upper()
        if normalized.startswith("SUCCESS"):
            return "passed"
        if normalized.startswith("FAIL"):
            return "failed"
        if normalized.startswith("ERROR"):
            return "error"
        if normalized.startswith("SKIPPED"):
            return "skipped"
        if "FAILED" in normalized or "FAIL" in normalized:
            return "failed"
        if "ERROR" in normalized:
            return "error"
        if "SKIPPED" in normalized:
            return "skipped"
        return "unknown"

    def _build_report(self, status: str, started_at: str) -> Dict[str, Any]:
        return {
            "status": status,
            "started_at": started_at,
            "ended_at": self._utcnow(),
            "rounds": self.run_reports,
            "saved_files": sorted(self.shared_memory.saved_files.keys()),
            "qa_feedback": self.shared_memory.qa_feedback,
            "report_path": str(self.report_path) if self.report_enabled else None,
        }

    def _write_report(self, report: Dict[str, Any]) -> None:
        if not self.report_enabled:
            return
        try:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"    📄 协作报告已写入: {self.report_path}")
        except Exception as e:
            print(f"    ⚠️ 协作报告写入失败: {e}")

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
        started_at = self._utcnow()
        self.run_reports = []
        if not self.agents:
            print("❌ 错误: 团队未初始化或没有工程师角色。")
            report = self._build_report("no_engineers", started_at)
            self._write_report(report)
            return {
                "status": "error",
                "error": "no_engineers",
                "outputs": self.shared_memory.get_all_outputs(),
                "report": report,
            }

        print(f"\n🚀 启动 TDD 协作流程 (最大轮次: {max_rounds})...")
        collab_cfg = self.config.get("collaboration", {})
        force_qa_on_success = collab_cfg.get("force_qa_on_success", False)
        post_success_qa_rounds = int(collab_cfg.get("post_success_qa_rounds", 0))

        run_status = "max_rounds_reached"
        for round_num in range(1, max_rounds + 1):
            print(f"\n🔄 --- 第 {round_num} 轮迭代 ---")
            round_report = {
                "round": round_num,
                "agents": [],
                "tests": {},
                "qa_feedback_recorded": False,
            }

            # 1. 工程师开发/修复
            for agent in self.agents:
                agent_report: Dict[str, Any] = {"role_name": agent.role_name}
                try:
                    agent.run()
                    # Save files from agent output
                    output_content = self.shared_memory.get_all_outputs()[agent.role_name][-1]
                    saved_files = save_files_from_content(output_content, self.output_dir)
                    if saved_files:
                        self.shared_memory.add_saved_files(saved_files)
                    agent_report.update(
                        {
                            "status": "completed",
                            "output_chars": len(output_content) if isinstance(output_content, str) else 0,
                            "saved_files": saved_files,
                        }
                    )
                    print(f"    ✅ [{agent.role_name}] 完成工作")
                except Exception as e:
                    agent_report.update({"status": "error", "error": str(e)})
                    print(f"    ❌ [{agent.role_name}] 执行出错: {str(e)}")
                round_report["agents"].append(agent_report)

            # 2. 自动化测试阶段 (Execution Feedback)
            print(f"    🧪 [System] 正在执行自动化测试/语法检查...")
            
            # Use injected executor
            test_results = self.code_executor.run_tests(str(self.output_dir))
            
            self.shared_memory.global_context["latest_test_results"] = test_results
            summary_lines = test_results.splitlines()
            summary = summary_lines[0] if summary_lines else "No test output."
            print(f"    📋 测试结果摘要: {summary}")
            test_status = self._classify_test_result(test_results)
            self.shared_memory.global_context["latest_test_status"] = test_status
            round_report["tests"] = {"status": test_status, "summary": summary}

            if test_status == "passed":
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
                        round_report["qa_feedback_recorded"] = True
                        round_report["qa_feedback"] = self._truncate(str(qa_feedback))
                        print("    📝 QA 反馈已记录")
                print("    >>> 提前结束协作循环。")
                self.run_reports.append(round_report)
                run_status = "passed"
                break

            # 3. QA 进行全局审查与反馈
            if self.qa_agent:
                print(f"\n🔍 [QA: {self.qa_agent.role_name}] 正在进行代码审查与测试分析...")
                qa_feedback = self.qa_agent.run()
                
                self.shared_memory.add_qa_feedback({
                    "round": round_num,
                    "test_status": "Passed" if test_status == "passed" else "Failed",
                    "feedback": qa_feedback
                })
                round_report["qa_feedback_recorded"] = True
                round_report["qa_feedback"] = self._truncate(str(qa_feedback))
                print(f"    📝 QA 反馈已记录")
            self.run_reports.append(round_report)

        report = self._build_report(run_status, started_at)
        self._write_report(report)
        return {
            "status": run_status,
            "outputs": self.shared_memory.get_all_outputs(),
            "report": report,
        }
