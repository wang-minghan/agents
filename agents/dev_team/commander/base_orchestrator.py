import json
import hashlib
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

from agents.dev_team.role_agent import RoleAgent
from agents.dev_team.memory import SharedMemoryStore
from agents.dev_team.interfaces import CodeExecutor, Agent
from agents.dev_team.execution import LocalUnsafeExecutor
from agents.common import save_files_from_content
from agents.dev_team.code_summarizer import CodeSummarizer

AgentFactory = Callable[[Dict[str, Any], Dict[str, Any], SharedMemoryStore, Path], Agent]


def default_agent_factory(jd: Dict[str, Any], config: Dict[str, Any], memory: SharedMemoryStore, output_dir: Path) -> Agent:
    return RoleAgent(jd, config, memory, output_dir)


class BaseOrchestrator:
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str = None,
        code_executor: Optional[CodeExecutor] = None,
        agent_factory: AgentFactory = default_agent_factory,
    ):
        self.config = config
        self.shared_memory = SharedMemoryStore(config)
        self.agent_factory = agent_factory

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            default_path = config.get("output_dir", "agents/dev_team/output/codebase")
            self.output_dir = Path(default_path)

        if not self.output_dir.is_absolute():
            self.output_dir = self.output_dir.resolve()

        print(f"Directory: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_dir / "evidence" / "collaboration_state.json"

        if code_executor:
            self.code_executor = code_executor
        else:
            self.code_executor = LocalUnsafeExecutor()

        self.iteration_target: Optional[Path] = None
        target = self.config.get("iteration_target")
        if target:
            self.iteration_target = Path(str(target)).expanduser().resolve()
        self._resume_state: Optional[Dict[str, Any]] = self._load_resume_state()
        if self.iteration_target:
            if self._resume_state:
                print("⚠️ 检测到协作状态，跳过迭代目标拷贝，直接恢复。")
            else:
                self._stage_iteration_target()

        self.shared_memory.global_context["requirements"] = ""

        self.agents: List[Agent] = []
        self.qa_agent: Optional[Agent] = None
        self.qa_agents: List[Agent] = []
        self.final_approver: Optional[Agent] = None
        self.run_reports: List[Dict[str, Any]] = []
        self.report_enabled = bool(self.config.get("report", {}).get("enabled", True))
        self.report_path = self._resolve_report_path()
        self._last_sync_at: Optional[float] = None
        self._last_synced_saved_files: set[str] = set()

    def _resolve_report_path(self) -> Path:
        report_path = self.config.get("report", {}).get("output_path")
        if report_path:
            report_path = Path(report_path)
            if not report_path.is_absolute():
                report_path = self.output_dir / report_path
        else:
            report_path = self.output_dir / "collaboration_report.json"
        return report_path

    def _resume_state_matches(self, state: Dict[str, Any]) -> bool:
        if not isinstance(state, dict):
            return False
        if state.get("status") != "in_progress":
            return False
        session_key = self.config.get("session_key")
        if session_key and state.get("session_key") != session_key:
            return False
        if self.iteration_target:
            stored_target = state.get("iteration_target")
            if stored_target and Path(stored_target).resolve() != self.iteration_target:
                return False
        output_dir = state.get("output_dir")
        if output_dir and Path(output_dir).resolve() != self.output_dir.resolve():
            return False
        return True

    def _load_resume_state(self) -> Optional[Dict[str, Any]]:
        if not self.state_path.exists():
            return None
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not self._resume_state_matches(payload):
            return None
        return payload

    def _snapshot_memory(self) -> Dict[str, Any]:
        return {
            "global_context": dict(self.shared_memory.global_context),
            "role_outputs": {k: list(v) for k, v in self.shared_memory.role_outputs.items()},
            "qa_feedback": list(self.shared_memory.qa_feedback),
            "saved_files": list(self.shared_memory.saved_files.keys()),
        }

    def _apply_resume_state(self, state: Dict[str, Any]) -> int:
        memory = state.get("shared_memory", {})
        if isinstance(memory, dict):
            self.shared_memory.global_context = dict(memory.get("global_context", {}))
            self.shared_memory.role_outputs = {
                k: list(v) for k, v in memory.get("role_outputs", {}).items()
            }
            self.shared_memory.qa_feedback = list(memory.get("qa_feedback", []))
            saved_files = memory.get("saved_files", [])
            self.shared_memory.saved_files = {path: "saved" for path in saved_files}
        completed_rounds = state.get("completed_rounds")
        if not isinstance(completed_rounds, int):
            completed_rounds = len(self.run_reports)
        return completed_rounds

    def _save_resume_state(self, completed_rounds: int, status: str) -> None:
        payload = {
            "version": 1,
            "status": status,
            "completed_rounds": completed_rounds,
            "run_reports": self.run_reports,
            "shared_memory": self._snapshot_memory(),
            "session_key": self.config.get("session_key"),
            "iteration_target": str(self.iteration_target) if self.iteration_target else None,
            "output_dir": str(self.output_dir),
            "saved_at": self._utcnow(),
        }
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"    ⚠️ 状态持久化失败: {exc}")

    def _resume_if_available(self, max_rounds: int) -> int:
        if not self._resume_state:
            return 1
        self.run_reports = list(self._resume_state.get("run_reports", []))
        completed_rounds = self._apply_resume_state(self._resume_state)
        if completed_rounds < len(self.run_reports):
            completed_rounds = len(self.run_reports)
        if completed_rounds >= max_rounds:
            print("⚠️ 断点轮次已达到最大轮次，需提高 max_rounds 才能继续。")
        else:
            print(f"⚠️ 已恢复到第 {completed_rounds} 轮，继续执行第 {completed_rounds + 1} 轮。")
        return completed_rounds + 1

    def _sync_iteration_artifacts(self) -> None:
        if not self.iteration_target:
            return
        target_root = self.iteration_target
        evidence_src = self.output_dir / "evidence"
        current_saved_files = set(self.shared_memory.saved_files.keys())
        if not evidence_src.exists() and not current_saved_files:
            return

        latest_mtime = 0.0
        if evidence_src.exists():
            for item in evidence_src.rglob("*"):
                if not item.is_file():
                    continue
                try:
                    latest_mtime = max(latest_mtime, item.stat().st_mtime)
                except OSError:
                    continue

        if (
            self._last_sync_at is not None
            and latest_mtime <= self._last_sync_at
            and current_saved_files == self._last_synced_saved_files
        ):
            return

        if evidence_src.exists():
            for item in evidence_src.rglob("*"):
                if not item.is_file():
                    continue
                rel = item.relative_to(evidence_src)
                dest = target_root / "evidence" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest)
        docs_target = target_root / "evidence" / "docs"
        for saved_path in self.shared_memory.saved_files.keys():
            path_obj = Path(saved_path)
            if path_obj.suffix.lower() != ".md":
                continue
            try:
                rel_doc = path_obj.resolve().relative_to(self.output_dir.resolve())
            except ValueError:
                continue
            dest_doc = docs_target / rel_doc
            dest_doc.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path_obj, dest_doc)
        self._last_sync_at = max(latest_mtime, time.time())
        self._last_synced_saved_files = current_saved_files

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

    def _run_agent_once(self, agent: Agent) -> Dict[str, Any]:
        agent_report: Dict[str, Any] = {"role_name": agent.role_name}
        try:
            agent.run()
            output_content = self.shared_memory.get_latest_output(agent.role_name) or ""
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
        return agent_report

    def _run_qa_agent(self, agent: Agent, round_num: int, test_status: str) -> Dict[str, Any]:
        try:
            feedback = agent.run()
            record = {
                "round": round_num,
                "test_status": test_status,
                "feedback": feedback,
                "role_name": agent.role_name,
            }
            self.shared_memory.add_qa_feedback(record)
            return {
                "role_name": agent.role_name,
                "feedback": self._truncate(str(feedback)),
            }
        except Exception as e:
            return {"role_name": agent.role_name, "error": str(e)}

    def _stage_iteration_target(self) -> None:
        if not self.iteration_target:
            return
        if not self.iteration_target.exists():
            raise FileNotFoundError(f"Iteration target not found: {self.iteration_target}")
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        shutil.copytree(self.iteration_target, self.output_dir)
        context = self._build_project_context(self.output_dir)
        self.shared_memory.global_context["project_context"] = context
        self._write_ast_baseline(context)

    def _write_ast_baseline(self, context: str) -> None:
        evidence_dir = self.output_dir / "evidence" / "ast"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = evidence_dir / "ast_baseline.md"
        baseline_path.write_text(context, encoding="utf-8")

    def _build_project_context(self, root: Path, max_files: int = 80, max_chars: int = 4000) -> str:
        lines: list[str] = [f"root={root}"]
        skip_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", "output", "data"}
        allowed_suffixes = {".py", ".md", ".yaml", ".yml", ".txt"}
        total_len = len(lines[0])
        count = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in sorted(dirnames) if name not in skip_dirs]
            for filename in sorted(filenames):
                path = Path(dirpath) / filename
                suffix = path.suffix.lower()
                if suffix not in allowed_suffixes:
                    continue
                rel = path.relative_to(root)
                try:
                    if suffix == ".py":
                        content = path.read_text(encoding="utf-8", errors="ignore")
                        summary = CodeSummarizer.summarize_python(content)
                    else:
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            summary = f.read(400)
                except Exception:
                    continue
                entry = f"\n[{rel}]\n{summary}"
                lines.append(entry)
                total_len += len(entry)
                count += 1
                if count >= max_files or total_len > max_chars:
                    break
            if count >= max_files or total_len > max_chars:
                break
        combined = "\n".join(lines)
        if len(combined) > max_chars:
            return combined[:max_chars] + "\n...[Truncated]..."
        return combined
    def _requires_final_approval(self) -> bool:
        requirements = self.shared_memory.global_context.get("requirements", "")
        text = str(requirements).lower()
        keywords = ("规划", "项目", "发布", "上线", "交付", "release", "launch", "plan", "project")
        return any(keyword in text for keyword in keywords)

    def _requires_ui_baseline(self) -> bool:
        requirements = self.shared_memory.global_context.get("requirements", "")
        text = str(requirements).lower()
        keywords = ("前端", "ui", "界面", "页面", "网页", "frontend", "web", "design")
        return any(keyword in text for keyword in keywords)

    def _check_ui_evidence(self) -> Dict[str, Any]:
        evidence_dir = self.output_dir / "evidence" / "ui"
        baseline = list(evidence_dir.glob("design_baseline.*")) + list(evidence_dir.glob("design_baseline_v*.*"))
        implementation = list(evidence_dir.glob("implementation.*")) + list(evidence_dir.glob("implementation_v*.*"))
        missing = []
        if not baseline:
            missing.append("design_baseline.*")
        if not implementation:
            missing.append("implementation.*")
        report = {
            "status": "passed" if not missing else "failed",
            "missing": missing,
            "path": str(evidence_dir),
        }
        if not missing:
            comparison = self._write_ui_comparison_report(evidence_dir, baseline[0], implementation[0])
            report["comparison_report"] = comparison
            if comparison.get("status") == "failed":
                report["status"] = "failed"
        return report

    def _write_ui_comparison_report(self, evidence_dir: Path, baseline: Path, implementation: Path) -> Dict[str, Any]:
        def _digest(path: Path) -> str:
            hash_obj = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()

        report: Dict[str, Any] = {
            "status": "passed",
            "baseline": {
                "path": str(baseline),
                "size_bytes": baseline.stat().st_size,
                "sha256": _digest(baseline),
            },
            "implementation": {
                "path": str(implementation),
                "size_bytes": implementation.stat().st_size,
                "sha256": _digest(implementation),
            },
        }
        try:
            from PIL import Image, ImageChops

            base_img = Image.open(baseline).convert("RGB")
            impl_img = Image.open(implementation).convert("RGB")
            if base_img.size != impl_img.size:
                report["status"] = "failed"
                report["pixel_diff"] = {"status": "failed", "reason": "size_mismatch"}
            else:
                diff = ImageChops.difference(base_img, impl_img)
                total_pixels = base_img.size[0] * base_img.size[1]
                diff_l = diff.convert("L")
                histogram = diff_l.histogram()
                mismatched = total_pixels - histogram[0]
                ratio = mismatched / total_pixels if total_pixels else 0.0
                diff_path = evidence_dir / "diff.png"
                diff.save(diff_path)
                threshold = 0.01
                report["pixel_diff"] = {
                    "status": "failed" if ratio > threshold else "passed",
                    "mismatch_ratio": ratio,
                    "threshold": threshold,
                    "diff_path": str(diff_path),
                }
                if ratio > threshold:
                    report["status"] = "failed"
        except Exception as exc:
            report["pixel_diff"] = {"status": "skipped", "reason": str(exc)}

        report_path = evidence_dir / "comparison.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        report["report_path"] = str(report_path)
        return report

    def _write_evidence_manifest(self, round_report: Dict[str, Any]) -> Path:
        evidence_dir = self.output_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "generated_at": self._utcnow(),
            "ui_evidence": round_report.get("ui_evidence"),
            "user_simulation": round_report.get("user_simulation"),
            "tests": round_report.get("tests"),
            "files": [str(p.relative_to(evidence_dir)) for p in evidence_dir.rglob("*") if p.is_file()],
        }
        manifest_path = evidence_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return manifest_path

    def _archive_evidence(self) -> Optional[Path]:
        evidence_dir = self.output_dir / "evidence"
        if not evidence_dir.exists():
            return None
        archive_base = evidence_dir / f"evidence_pack_{self._utcnow().replace(':', '-')}"
        archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=str(evidence_dir))
        return Path(archive_path)

    def _run_approval_gate(self, round_num: int, test_status: str, round_report: Dict[str, Any]) -> bool:
        approvals_enabled = self._requires_final_approval()
        approval_rounds = 2
        require_keyword = "APPROVED"
        if not approvals_enabled:
            return True
        if not self.final_approver:
            round_report["approvals"] = {"status": "rejected", "reason": "no_final_approver"}
            return False

        approvals = []
        approved = True

        for _ in range(approval_rounds):
            print(f"\n✅ [Final Approval: {self.final_approver.role_name}] 进行交付审批...")
            feedback = self.final_approver.run()
            record = {
                "round": round_num,
                "test_status": test_status,
                "feedback": feedback,
            }
            approvals.append(record)
            if require_keyword and require_keyword not in str(feedback).upper():
                approved = False

        round_report["approvals"] = {
            "status": "approved" if approved else "rejected",
            "records": approvals,
            "require_keyword": require_keyword,
        }
        return approved

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
        final_jds = planner_result.get("final_jds", [])
        requirements = planner_result.get("requirements", {})

        self.shared_memory.global_context["requirements"] = json.dumps(requirements, ensure_ascii=False)

        print(f"\n>>> 正在组建开发团队，检测到 {len(final_jds)} 个角色需求...")

        for jd in final_jds:
            role_name = jd.get("role_name", "Unknown")
            agent = self.agent_factory(jd, self.config, self.shared_memory, self.output_dir)

            is_qa = agent.role_type == "QA" or "QA" in role_name or "Test" in role_name
            is_final_approver = any(
                key in role_name.lower()
                for key in ("planner", "product", "pm", "project", "owner", "负责人", "产品", "规划")
            )

            if is_qa:
                self.qa_agent = agent
                self.qa_agents.append(agent)
                print(f"    └── 已指派 QA 专家: {role_name}")
            elif is_final_approver:
                self.final_approver = agent
                print(f"    └── 已指派 最终审批人: {role_name}")
            else:
                self.agents.append(agent)
                print(f"    └── 已指派工程师: {role_name}")

    def run_collaboration(self, max_rounds: int = 3):
        started_at = self._utcnow()
        self.run_reports = []
        start_round = self._resume_if_available(max_rounds)
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

        if start_round > max_rounds:
            report = self._build_report("max_rounds_reached", started_at)
            self._write_report(report)
            self._save_resume_state(max_rounds, "max_rounds_reached")
            return {
                "status": "max_rounds_reached",
                "outputs": self.shared_memory.get_all_outputs(),
                "report": report,
            }

        print(f"\n🚀 启动 TDD 协作流程 (最大轮次: {max_rounds})...")
        run_status = "max_rounds_reached"
        for round_num in range(start_round, max_rounds + 1):
            print(f"\n🔄 --- 第 {round_num} 轮迭代 ---")
            round_report = {
                "round": round_num,
                "agents": [],
                "tests": {},
                "qa_feedback_recorded": False,
            }

            max_workers = min(4, max(1, len(self.agents)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self._run_agent_once, agent) for agent in self.agents]
                for future in as_completed(futures):
                    round_report["agents"].append(future.result())

            print(f"    🧪 [System] 正在执行自动化测试/语法检查...")

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
                if self._requires_ui_baseline():
                    ui_check = self._check_ui_evidence()
                    round_report["ui_evidence"] = ui_check
                    if ui_check["status"] != "passed":
                        print("    ❌ UI 证据不完整，协作流程结束。")
                        self.run_reports.append(round_report)
                        run_status = "ui_evidence_missing"
                        break
                sim_result = "SKIPPED: No user simulation."
                if hasattr(self.code_executor, "run_user_simulation"):
                    print("    🧭 [System] 正在执行用户模拟测试...")
                    sim_result = self.code_executor.run_user_simulation(str(self.output_dir))
                sim_status = self._classify_test_result(sim_result)
                round_report["user_simulation"] = {"status": sim_status, "summary": sim_result.splitlines()[0] if sim_result else "No output"}
                if sim_status not in ("passed", "skipped", "unknown"):
                    print("    ❌ 用户模拟测试未通过，协作流程结束。")
                    self.run_reports.append(round_report)
                    run_status = "user_simulation_failed"
                    break
                self._write_evidence_manifest(round_report)
                self._archive_evidence()
                approved = self._run_approval_gate(round_num, test_status, round_report)
                self.run_reports.append(round_report)
                if approved:
                    print("    >>> 提前结束协作循环。")
                    run_status = "passed"
                else:
                    print("    ⚠️ 审批未通过，协作流程结束。")
                    run_status = "approval_failed"
                break

            if self.qa_agents:
                print(f"\n🔍 [QA] 正在进行代码审查与测试分析...")
                qa_reports = []
                max_workers = min(4, max(1, len(self.qa_agents)))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(
                            self._run_qa_agent,
                            agent,
                            round_num,
                            "Passed" if test_status == "passed" else "Failed",
                        )
                        for agent in self.qa_agents
                    ]
                    for future in as_completed(futures):
                        qa_reports.append(future.result())
                round_report["qa_feedback_recorded"] = True
                round_report["qa_feedback"] = qa_reports
                print(f"    📝 QA 反馈已记录")
            self.run_reports.append(round_report)
            self._save_resume_state(round_num, "in_progress")
            self._sync_iteration_artifacts()

        report = self._build_report(run_status, started_at)
        self._write_report(report)
        self._save_resume_state(len(self.run_reports), run_status)
        self._sync_iteration_artifacts()
        return {
            "status": run_status,
            "outputs": self.shared_memory.get_all_outputs(),
            "report": report,
        }
