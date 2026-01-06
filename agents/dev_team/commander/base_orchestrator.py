import ast
import json
import os
import shutil
import time
import fnmatch
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agents.dev_team.role_agent import RoleAgent
from agents.dev_team.memory import SharedMemoryStore
from agents.dev_team.interfaces import CodeExecutor, Agent
from agents.dev_team.execution import LocalUnsafeExecutor, DisabledExecutor
from agents.common import find_project_root, parse_code_blocks, save_files_from_content
from agents.dev_team.code_summarizer import CodeSummarizer
from agents.dev_team.domain.models import Evidence, GateEvidence, GateRule
from agents.dev_team.gates import QualityGate
from agents.dev_team.capabilities import CapabilityRegistry
from agents.dev_team.app import PlanTaskCoordinator, MilestoneStateMachine, WorkflowRouter
from agents.dev_team.policies import (
    PolicyRegistry,
    ReviewPolicyImpl,
    ConsensusPolicyImpl,
    VerificationPolicyImpl,
    CompliancePolicyImpl,
    UIBaselinePolicy,
)
from agents.dev_team.services import ExecutionService, VerificationService, WorkerPool

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
            exec_cfg = config.get("execution", {})
            allow_unsafe = exec_cfg.get("allow_unsafe", True)
            if not allow_unsafe:
                self.code_executor = DisabledExecutor(reason="Unsafe execution disabled.")
            else:
                testing_cfg = config.get("testing", {})
                self.code_executor = LocalUnsafeExecutor(
                    test_cmd=testing_cfg.get("command"),
                    timeout=testing_cfg.get("timeout", 60),
                    per_file=testing_cfg.get("per_file", True),
                    ui_test_patterns=testing_cfg.get("ui_test_patterns"),
                    coverage_cmd=testing_cfg.get("coverage_command"),
                    coverage_timeout=testing_cfg.get("coverage_timeout", 120),
                )

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
        else:
            self._clear_output_dir_on_start()

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
        self._file_write_lock = threading.Lock()
        self._round_saved_files: set[str] = set()
        self._quality_gate = QualityGate(self._build_gate_rules())
        self._capability_registry = CapabilityRegistry()
        self._policy_registry = PolicyRegistry()
        self._ui_policy = UIBaselinePolicy(self.config, self.output_dir, self.shared_memory)
        self._policy_registry.register("review", ReviewPolicyImpl(self))
        self._policy_registry.register("consensus", ConsensusPolicyImpl(self))
        self._policy_registry.register("verification", VerificationPolicyImpl(self))
        self._policy_registry.register(
            "compliance",
            CompliancePolicyImpl(self.config, self.output_dir, self.shared_memory),
        )
        self._policy_registry.register("ui_baseline", self._ui_policy)
        self._workflow_router = WorkflowRouter(self.config)
        self._milestone_fsm = MilestoneStateMachine()
        self._coordinator = PlanTaskCoordinator(
            self._workflow_router,
            self._milestone_fsm,
            self._policy_registry,
            self._capability_registry,
            self._emit_event,
        )
        self._worker_pool = WorkerPool(self._capability_registry)
        self._execution_service = ExecutionService(self._run_agent_once)
        self._verification_service = VerificationService(self.code_executor)

    def _clear_output_dir_on_start(self) -> None:
        if self._resume_state:
            return
        if not self.config.get("clean_output_on_start", True):
            return
        if not self.output_dir.exists():
            return
        if not self._is_safe_output_dir(self.output_dir):
            print(f"⚠️ 输出目录不安全，已跳过清理: {self.output_dir}")
            return
        shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
            "domain_events": list(self.shared_memory.domain_events),
            "evidence": list(self.shared_memory.evidence),
            "gate_decisions": list(self.shared_memory.gate_decisions),
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
            self.shared_memory.domain_events = list(memory.get("domain_events", []))
            self.shared_memory.evidence = list(memory.get("evidence", []))
            self.shared_memory.gate_decisions = list(memory.get("gate_decisions", []))
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

    def _review_required_paths(self) -> Dict[str, Path]:
        docs_dir = self.output_dir / "evidence" / "docs"
        return {
            "brainstorm_record": docs_dir / "brainstorm_record.md",
            "design_review_checklist": docs_dir / "design_review_checklist.md",
            "acceptance_checklist": docs_dir / "acceptance_checklist.md",
            "adr": docs_dir / "adr.md",
            "current_state_summary": docs_dir / "current_state_summary.md",
        }

    def _ensure_ast_baseline(self) -> Path:
        evidence_dir = self.output_dir / "evidence" / "ast"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = evidence_dir / "ast_baseline.md"
        if baseline_path.exists():
            return baseline_path
        context = self._build_project_context(self.output_dir)
        baseline_path.write_text(context, encoding="utf-8")
        return baseline_path

    def _get_requirements_payload(self) -> Dict[str, Any]:
        raw = self.shared_memory.global_context.get("requirements", "")
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw:
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                return {"raw": raw}
        return {}

    def _get_acceptance_criteria(self, requirements: Dict[str, Any]) -> List[str]:
        criteria: List[str] = []
        if isinstance(requirements, dict):
            items = requirements.get("acceptance_criteria") or requirements.get("acceptance") or []
            if isinstance(items, list):
                criteria.extend([str(item).strip() for item in items if str(item).strip()])
            elif isinstance(items, str) and items.strip():
                criteria.append(items.strip())
        stored = self.shared_memory.global_context.get("acceptance_criteria")
        if isinstance(stored, list):
            criteria.extend([str(item).strip() for item in stored if str(item).strip()])
        seen: set[str] = set()
        result: List[str] = []
        for item in criteria:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    def _render_acceptance_checklist(self, goal: str, criteria: List[str]) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if not criteria:
            items = "- [ ] 未提供验收标准\n"
        else:
            items = "\n".join([f"- [ ] {item}" for item in criteria]) + "\n"
        return (
            "# 验收清单\n"
            f"- 任务: {goal}\n"
            f"- 日期: {today}\n\n"
            "## 验收项\n"
            f"{items}"
        )

    def _verify_acceptance_checklist(self, criteria: List[str]) -> Dict[str, Any]:
        if not criteria:
            return {"status": "skipped", "missing": []}
        checklist_path = self.output_dir / "evidence" / "docs" / "acceptance_checklist.md"
        if not checklist_path.exists():
            return {"status": "failed", "missing": criteria, "path": str(checklist_path)}
        text = checklist_path.read_text(encoding="utf-8", errors="ignore")
        require_checked = self.config.get("quality_gates", {}).get("acceptance", {}).get("require_checked", False)
        if require_checked:
            lines = [line.strip().lower() for line in text.splitlines() if line.strip().lower().startswith("- [x]")]
        else:
            lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
        baseline_missing_allowed = False
        if self._allow_missing_ui_baseline():
            evidence_dir = self.output_dir / "evidence" / "ui"
            has_baseline = any(
                evidence_dir.glob("design_baseline.*")
            ) or any(evidence_dir.glob("design_baseline_v*.*"))
            baseline_missing_allowed = not has_baseline
        missing = []
        for item in criteria:
            if item.lower() in " ".join(lines):
                continue
            if baseline_missing_allowed:
                lowered = item.lower()
                if "基线" in lowered or "design baseline" in lowered:
                    continue
            missing.append(item)
        status = "passed" if not missing else "failed"
        return {"status": status, "missing": missing, "path": str(checklist_path)}

    def _collect_refactor_suggestions(
        self,
        root: Path,
        min_lines: int = 50,
        max_items: int = 20,
    ) -> List[Dict[str, Any]]:
        suggestions: List[Dict[str, Any]] = []
        skip_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", "output", "data", "evidence"}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in skip_dirs]
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                path = Path(dirpath) / filename
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                try:
                    tree = ast.parse(content)
                except Exception:
                    continue

                class_stack: List[str] = []

                class FunctionVisitor(ast.NodeVisitor):
                    def visit_ClassDef(self, node: ast.ClassDef) -> None:
                        class_stack.append(node.name)
                        self.generic_visit(node)
                        class_stack.pop()

                    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                        self._handle_function(node)

                    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                        self._handle_function(node)

                    def _handle_function(self, node: ast.AST) -> None:
                        start = getattr(node, "lineno", None)
                        end = getattr(node, "end_lineno", None)
                        if start is None:
                            return
                        if end is None:
                            body = getattr(node, "body", [])
                            if body:
                                end = getattr(body[-1], "end_lineno", getattr(body[-1], "lineno", start))
                            else:
                                end = start
                        length = end - start + 1
                        if length < min_lines:
                            return
                        func_name = getattr(node, "name", "unknown")
                        if class_stack:
                            func_name = f"{class_stack[-1]}.{func_name}"
                        suggestions.append(
                            {
                                "file": str(path.relative_to(root)),
                                "function": func_name,
                                "lines": length,
                            }
                        )

                FunctionVisitor().visit(tree)

        suggestions.sort(key=lambda item: item["lines"], reverse=True)
        return suggestions[:max_items]

    @staticmethod
    def _format_refactor_suggestions(suggestions: List[Dict[str, Any]]) -> str:
        if not suggestions:
            return "无"
        lines = []
        for item in suggestions:
            lines.append(f"- {item['file']}: {item['function']} ({item['lines']} 行)")
        return "\n".join(lines)

    @staticmethod
    def _truncate_text(text: str, limit: int = 3000) -> str:
        if not text:
            return ""
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...[Truncated]..."

    def _emit_event(self, name: str, payload: Dict[str, Any]) -> None:
        try:
            self.shared_memory.add_domain_event(name, payload)
        except Exception as exc:
            print(f"    ⚠️ 事件记录失败: {exc}")

    def _build_gate_rules(self) -> List[GateRule]:
        gates = self.config.get("quality_gates", {})
        return [
            GateRule("tests", bool(gates.get("require_tests", True))),
            GateRule("ui_tests", bool(gates.get("require_ui_tests", True))),
            GateRule("coverage", bool(gates.get("require_coverage", True))),
            GateRule("user_simulation", bool(gates.get("require_ui_simulation", True))),
            GateRule("ui_evidence", bool(gates.get("require_ui_tests", True))),
            GateRule("acceptance", True),
            GateRule("input_contract", True),
            GateRule("compliance", bool(gates.get("require_compliance", False))),
        ]

    def _build_round_evidence(self, round_report: Dict[str, Any]) -> List[Evidence]:
        evidence: List[Evidence] = []
        tests = round_report.get("tests", {})
        evidence.append(
            Evidence(
                kind="tests",
                status=tests.get("status", "unknown"),
                summary=tests.get("summary", ""),
                required=self._should_require_tests(),
            )
        )
        ui_tests = round_report.get("ui_tests", {})
        evidence.append(
            Evidence(
                kind="ui_tests",
                status=ui_tests.get("status", "unknown"),
                summary=ui_tests.get("summary", ""),
                required=self._requires_ui_functional_audit() and self._should_require_ui_tests(),
            )
        )
        coverage = round_report.get("coverage", {})
        evidence.append(
            Evidence(
                kind="coverage",
                status=coverage.get("status", "unknown"),
                summary=coverage.get("summary", ""),
                required=self._should_require_coverage(),
            )
        )
        input_contract = round_report.get("input_contract", {})
        evidence.append(
            Evidence(
                kind="input_contract",
                status=input_contract.get("status", "unknown"),
                summary=input_contract.get("summary", ""),
                required=True,
            )
        )
        ui_evidence = round_report.get("ui_evidence")
        if ui_evidence:
            evidence.append(
                Evidence(
                    kind="ui_evidence",
                    status=ui_evidence.get("status", "unknown"),
                    summary=str(ui_evidence.get("summary", "")),
                    required=self._requires_ui_functional_audit(),
                    metadata=ui_evidence,
                )
            )
        user_simulation = round_report.get("user_simulation", {})
        if user_simulation:
            evidence.append(
                Evidence(
                    kind="user_simulation",
                    status=user_simulation.get("status", "unknown"),
                    summary=user_simulation.get("summary", ""),
                    required=self._requires_ui_functional_audit() and self._should_require_ui_simulation(),
                )
            )
        acceptance = round_report.get("acceptance", {})
        if acceptance:
            evidence.append(
                Evidence(
                    kind="acceptance",
                    status=acceptance.get("status", "unknown"),
                    summary=acceptance.get("summary", ""),
                    required=True,
                    metadata=acceptance,
                )
            )
        compliance_report = self.shared_memory.global_context.get("compliance_report")
        if isinstance(compliance_report, dict):
            evidence.append(
                Evidence(
                    kind="compliance",
                    status=compliance_report.get("status", "unknown"),
                    summary=str(compliance_report.get("summary") or compliance_report.get("reason") or ""),
                    required=self.config.get("quality_gates", {}).get("require_compliance", False),
                    metadata=compliance_report,
                )
            )
        consensus = round_report.get("consensus")
        if consensus:
            evidence.append(
                Evidence(
                    kind="consensus",
                    status=consensus.get("status", "unknown"),
                    summary=consensus.get("summary", ""),
                    required=False,
                    metadata=consensus,
                )
            )
        validation = round_report.get("validation")
        if validation:
            evidence.append(
                Evidence(
                    kind="cross_validation",
                    status=validation.get("status", "unknown"),
                    summary=validation.get("summary", ""),
                    required=False,
                    metadata=validation,
                )
            )
        return evidence

    def _finalize_round_report(self, round_report: Dict[str, Any]) -> None:
        evidence = self._build_round_evidence(round_report)
        evidence_payload = [item.to_dict() for item in evidence]
        gate_evidence = [GateEvidence(rule_key=item.kind, evidence=item.to_dict()) for item in evidence]
        gate_evidence_payload = [item.to_dict() for item in gate_evidence]
        gate_decision = self._quality_gate.evaluate(evidence)
        round_report["evidence"] = evidence_payload
        round_report["gate_evidence"] = gate_evidence_payload
        round_report["gate_decision"] = gate_decision.to_dict()
        self.shared_memory.add_evidence(evidence_payload)
        self.shared_memory.add_gate_decision(gate_decision.to_dict())
        self._emit_event(
            "gate_decision",
            {"round": round_report.get("round"), **gate_decision.to_dict()},
        )
        self._emit_event(
            "round_completed",
            {"round": round_report.get("round"), "status": gate_decision.status},
        )
        if gate_decision.status == "passed":
            state_event = "gate_passed"
        else:
            state_event = "gate_failed"
        round_report["milestone_state"] = self._coordinator.transition(
            state_event, round_num=round_report.get("round", 0)
        )

    def _rounds_policy(self) -> Dict[str, Any]:
        return self.config.get("rounds_policy", {})

    def _detect_high_risk(self, requirements: Any) -> bool:
        if not requirements:
            return False
        policy = self._rounds_policy()
        keywords = [kw.lower() for kw in policy.get("high_risk_keywords", [])]
        text = json.dumps(requirements, ensure_ascii=False).lower()
        if any(keyword in text for keyword in keywords):
            return True
        if isinstance(requirements, dict):
            risks = requirements.get("risks") or requirements.get("risks_and_mitigations")
            if isinstance(risks, list):
                return bool(risks)
            if isinstance(risks, str) and risks.strip() and risks.strip() != "未提供":
                return True
        return False

    def _detect_high_demand(self, requirements: Any) -> bool:
        if not requirements:
            return False
        policy = self._rounds_policy()
        keywords = [kw.lower() for kw in policy.get("high_demand_keywords", [])]
        text = json.dumps(requirements, ensure_ascii=False).lower()
        if any(keyword in text for keyword in keywords):
            return True
        if isinstance(requirements, dict):
            priority = str(requirements.get("priority", "")).lower()
            if priority in ("high", "p0", "critical", "urgent"):
                return True
        return False

    def _apply_rounds_policy(self, requested_rounds: int) -> int:
        policy = self._rounds_policy()
        min_rounds = int(policy.get("min_rounds", 1))
        max_high_risk = int(policy.get("max_rounds_high_risk", requested_rounds))
        requirements = self._get_requirements_payload()
        rounds = max(requested_rounds, min_rounds)
        if self._detect_high_risk(requirements) or self._detect_high_demand(requirements):
            rounds = max(rounds, max_high_risk)
        return rounds

    def _render_current_state_summary(
        self,
        requirements: Dict[str, Any],
        ast_path: Path,
        suggestions: List[Dict[str, Any]],
    ) -> str:
        goal = requirements.get("goal") or requirements.get("summary") or "未提供"
        functional = requirements.get("functional_requirements") or requirements.get("functions") or []
        if isinstance(functional, list) and functional:
            functional_text = "\n".join([f"- {item}" for item in functional])
        elif functional:
            functional_text = str(functional)
        else:
            functional_text = "未提供"

        key_flow = requirements.get("core_flow") or requirements.get("workflow") or "未提供"
        risks = requirements.get("risks") or requirements.get("risks_and_mitigations") or "未提供"
        suggestion_text = self._format_refactor_suggestions(suggestions)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        return (
            "# 现状摘要\n"
            f"- 任务: {goal}\n"
            f"- 日期: {today}\n"
            f"- AST 基线路径: {ast_path}\n\n"
            "## 功能概览\n"
            f"{functional_text}\n\n"
            "## 关键路径\n"
            f"{key_flow}\n\n"
            "## 已知风险\n"
            f"{risks}\n\n"
            "## 长函数热点\n"
            f"{suggestion_text}\n"
        )

    def _get_review_template_dir(self) -> Path:
        review_cfg = self.config.get("review", {})
        template_dir = review_cfg.get("template_dir")
        if template_dir:
            path = Path(template_dir)
            if not path.is_absolute():
                base_dir = Path(self.config.get("agent_root", Path(__file__).parent))
                path = base_dir / template_dir
            return path
        project_root = find_project_root(Path(__file__).parent)
        return project_root / "docs" / "templates"

    def _read_review_template(self, filename: str) -> str:
        template_dir = self._get_review_template_dir()
        path = template_dir / filename
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _write_review_files_from_content(
        self,
        content: str,
        required_paths: Dict[str, Path],
    ) -> Dict[str, str]:
        saved: Dict[str, str] = {}
        required_rel = {path.relative_to(self.output_dir): key for key, path in required_paths.items()}
        for rel_path, body in parse_code_blocks(content):
            clean_path = Path(rel_path.strip().replace("\\", "/"))
            if clean_path.is_absolute() or ".." in clean_path.parts:
                continue
            full_path = (self.output_dir / clean_path).resolve()
            try:
                rel = full_path.relative_to(self.output_dir)
            except ValueError:
                continue
            if rel not in required_rel:
                continue
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(body, encoding="utf-8")
            saved[required_rel[rel]] = str(full_path)
        return saved

    def _fallback_review_docs(
        self,
        requirements: Dict[str, Any],
        ast_path: Path,
        current_summary_path: Path,
        suggestions: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        goal = requirements.get("goal") or "未提供"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        is_update = bool(requirements.get("update_mode") or requirements.get("iteration_target"))
        suggestion_text = self._format_refactor_suggestions(suggestions)
        acceptance_criteria = self._get_acceptance_criteria(requirements)

        if is_update:
            plan_a = "方案A: 保守增量修改，优先函数级替换，最小风险迭代。"
            plan_b = "方案B: 模块化重构，拆分热点模块并保持向后兼容。"
            plan_c = "方案C: 仅修复关键缺陷，冻结结构变更。"
        else:
            plan_a = "方案A: MVP 分层实现，先打通关键路径。"
            plan_b = "方案B: 模块优先，先定义边界与接口，再补实现。"
            plan_c = "方案C: 仅输出原型与证据，延后完整实现。"

        brainstorm = (
            "# 头脑风暴记录\n"
            f"- 任务: {goal}\n"
            f"- 日期: {today}\n"
            f"- 输入: AST 基线 {ast_path}；现状摘要 {current_summary_path}\n\n"
            "## 约束\n"
            "- 目标:\n"
            "- 边界:\n"
            "- 非功能需求:\n"
            "- 输入契约:\n\n"
            "## 方案A\n"
            f"- 描述: {plan_a}\n"
            "- 质量/稳定性/可读性/工程性/模块化:\n"
            "- 风险与缓解:\n\n"
            "## 方案B\n"
            f"- 描述: {plan_b}\n"
            "- 质量/稳定性/可读性/工程性/模块化:\n"
            "- 风险与缓解:\n\n"
            "## 备选方案C\n"
            f"- 描述: {plan_c}\n"
            "- 质量/稳定性/可读性/工程性/模块化:\n"
            "- 风险与缓解:\n\n"
            "## 选择理由\n"
            "- 结论:\n\n"
            "## 拆解建议\n"
            "- 触发条件: 函数超过 50 行\n"
            f"{suggestion_text}\n"
        )

        design_review = (
            "# 设计审查清单\n"
            f"- 任务: {goal}\n"
            f"- 日期: {today}\n\n"
            "## 质量\n"
            "- [ ] 验收清单齐全\n"
            "- [ ] 证据要求明确\n"
            "- [ ] 关键路径覆盖\n\n"
            "## 稳定性\n"
            "- [ ] 不可变约束明确（目标/边界/输入契约）\n"
            "- [ ] 关键阈值定义清楚（性能/可靠性）\n"
            "- [ ] 失败策略明确（不允许 silent fail）\n\n"
            "## 可读性\n"
            "- [ ] 关键路径可解释\n"
            "- [ ] 命名与层次一致\n"
            "- [ ] 核心模块边界清晰\n\n"
            "## 工程性\n"
            "- [ ] 迁移/回滚方案\n"
            "- [ ] 监控/日志方案\n"
            "- [ ] 依赖与风险评估\n\n"
            "## 模块化\n"
            "- [ ] 依赖层级可控\n"
            "- [ ] 循环依赖检测通过\n"
            "- [ ] 长函数拆解建议已给出\n"
        )

        adr = (
            "# ADR\n"
            f"- title: {goal}\n"
            "- status: proposed\n"
            f"- date: {today}\n\n"
            "## context\n"
            "- 背景:\n"
            "- 约束:\n\n"
            "## decision\n"
            "- 选择:\n\n"
            "## alternatives\n"
            "- 备选:\n"
            "- 取舍:\n\n"
            "## consequences\n"
            "- 短期影响:\n"
            "- 长期维护成本:\n"
        )

        acceptance_checklist = self._render_acceptance_checklist(goal, acceptance_criteria)

        return {
            "brainstorm_record": brainstorm,
            "design_review_checklist": design_review,
            "acceptance_checklist": acceptance_checklist,
            "adr": adr,
        }

    def _generate_review_docs_with_llm(
        self,
        requirements: Dict[str, Any],
        ast_path: Path,
        current_summary: str,
        required_paths: Dict[str, Path],
        suggestions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        llm_cfg = self.config.get("llm", {})
        if not llm_cfg.get("api_key"):
            return {"status": "skipped", "reason": "llm_not_configured"}
        templates = {
            "brainstorm": self._read_review_template("brainstorm_record.md"),
            "design_review": self._read_review_template("design_review_checklist.md"),
            "acceptance": self._read_review_template("acceptance_checklist.md"),
            "adr": self._read_review_template("adr.md"),
        }
        ast_text = ""
        if ast_path.exists():
            ast_text = self._truncate_text(ast_path.read_text(encoding="utf-8", errors="ignore"))
        acceptance_criteria = self._get_acceptance_criteria(requirements)
        prompt = (
            "你是资深架构师，需要基于输入生成审查产出。\n"
            "输出要求：仅输出 4 个 <file> 块，路径分别为：\n"
            "- evidence/docs/brainstorm_record.md\n"
            "- evidence/docs/design_review_checklist.md\n"
            "- evidence/docs/acceptance_checklist.md\n"
            "- evidence/docs/adr.md\n"
            "不得输出其他文本。\n\n"
            "[需求]\n"
            f"{json.dumps(requirements, ensure_ascii=False, indent=2)}\n\n"
            "[AST 基线(节选)]\n"
            f"{ast_text}\n\n"
            "[现状摘要]\n"
            f"{current_summary}\n\n"
            "[验收标准]\n"
            f"{json.dumps(acceptance_criteria, ensure_ascii=False, indent=2)}\n\n"
            "[长函数热点]\n"
            f"{self._format_refactor_suggestions(suggestions)}\n\n"
            "[模板: Brainstorm]\n"
            f"{templates['brainstorm']}\n\n"
            "[模板: Design Review]\n"
            f"{templates['design_review']}\n\n"
            "[模板: Acceptance]\n"
            f"{templates['acceptance']}\n\n"
            "[模板: ADR]\n"
            f"{templates['adr']}\n"
        )
        try:
            llm = ChatOpenAI(
                model=llm_cfg.get("model", "gpt-4o"),
                api_key=llm_cfg.get("api_key"),
                base_url=llm_cfg.get("api_base"),
                temperature=0.2,
                timeout=60,
                max_retries=1,
            )
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            saved = self._write_review_files_from_content(content, required_paths)
            return {"status": "completed", "saved": saved}
        except Exception as exc:
            return {"status": "failed", "reason": str(exc)}

    def _ensure_review_artifacts(self) -> Dict[str, Any]:
        review_cfg = self.config.get("review", {})
        if review_cfg.get("enabled", True) is False:
            return {"status": "skipped", "reason": "disabled"}
        use_llm = review_cfg.get("use_llm", True)
        required_paths = self._review_required_paths()
        missing = [
            key
            for key, path in required_paths.items()
            if not path.exists() or path.stat().st_size == 0
        ]
        if missing:
            ast_path = self._ensure_ast_baseline()
            requirements = self._get_requirements_payload()
            suggestions = self._collect_refactor_suggestions(self.output_dir)
            current_summary = self._render_current_state_summary(requirements, ast_path, suggestions)
            required_paths["current_state_summary"].parent.mkdir(parents=True, exist_ok=True)
            required_paths["current_state_summary"].write_text(current_summary, encoding="utf-8")
            if use_llm:
                self._generate_review_docs_with_llm(
                    requirements=requirements,
                    ast_path=ast_path,
                    current_summary=current_summary,
                    required_paths=required_paths,
                    suggestions=suggestions,
                )
            fallback_docs = self._fallback_review_docs(
                requirements=requirements,
                ast_path=ast_path,
                current_summary_path=required_paths["current_state_summary"],
                suggestions=suggestions,
            )
            for key, content in fallback_docs.items():
                path = required_paths[key]
                if path.exists() and path.stat().st_size > 0:
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

        review_docs = {
            key: str(path)
            for key, path in required_paths.items()
            if path.exists() and path.stat().st_size > 0
        }
        still_missing = [key for key in required_paths.keys() if key not in review_docs]
        self.shared_memory.global_context["review_docs"] = review_docs
        status = "passed" if not still_missing else "failed"
        return {
            "status": status,
            "missing": still_missing,
            "docs": review_docs,
        }

    @staticmethod
    def _utcnow() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _truncate(text: str, limit: int = 1000) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...[Truncated]..."

    def _build_report(self, status: str, started_at: str) -> Dict[str, Any]:
        return {
            "status": status,
            "started_at": started_at,
            "ended_at": self._utcnow(),
            "rounds": self.run_reports,
            "saved_files": sorted(self.shared_memory.saved_files.keys()),
            "qa_feedback": self.shared_memory.qa_feedback,
            "domain_events": self.shared_memory.domain_events,
            "evidence": self.shared_memory.evidence,
            "gate_decisions": self.shared_memory.gate_decisions,
            "acceptance_criteria": self.shared_memory.global_context.get("acceptance_criteria", []),
            "ui_design": self.shared_memory.global_context.get("ui_design"),
            "ui_design_summary": self.shared_memory.global_context.get("ui_design_summary"),
            "compliance_report": self.shared_memory.global_context.get("compliance_report"),
            "bug_cards": self.shared_memory.global_context.get("bug_cards", []),
            "review_docs": self.shared_memory.global_context.get("review_docs", {}),
            "report_path": str(self.report_path) if self.report_enabled else None,
        }

    def _max_files_per_agent(self) -> int:
        delivery_cfg = self.config.get("delivery", {})
        try:
            return int(delivery_cfg.get("max_files_per_agent", 1))
        except (TypeError, ValueError):
            return 1

    def _max_modules_per_agent(self) -> int:
        delivery_cfg = self.config.get("delivery", {})
        try:
            return int(delivery_cfg.get("max_modules_per_agent", 1))
        except (TypeError, ValueError):
            return 1

    def _limit_agent_outputs(self, content: str, role_name: str) -> tuple[str, int]:
        blocks = parse_code_blocks(content)
        if not blocks:
            return content, 0
        module_limit = self._max_modules_per_agent()
        if module_limit > 0:
            modules: List[str] = []
            for path, _ in blocks:
                parts = Path(path.strip().replace("\\", "/")).parts
                modules.append(parts[0] if parts else "")
            if modules:
                kept_module = modules[0]
                if module_limit == 1:
                    filtered = [
                        block
                        for block, module in zip(blocks, modules)
                        if module == kept_module
                    ]
                    if len(filtered) != len(blocks):
                        dropped = len(blocks) - len(filtered)
                        trimmed = "\n".join(
                            [f"<file path=\"{path}\">\n{body}\n</file>" for path, body in filtered]
                        )
                        print(f"    ⚠️ [{role_name}] 输出跨模块，已限制到模块 {kept_module}，丢弃 {dropped} 个文件")
                        blocks = filtered

        limit = self._max_files_per_agent()
        if limit <= 0:
            trimmed = "\n".join(
                [f"<file path=\"{path}\">\n{body}\n</file>" for path, body in blocks]
            )
            return trimmed, 0
        if len(blocks) <= limit:
            trimmed = "\n".join(
                [f"<file path=\"{path}\">\n{body}\n</file>" for path, body in blocks]
            )
            return trimmed, 0
        kept = blocks[:limit]
        trimmed = "\n".join(
            [f"<file path=\"{path}\">\n{body}\n</file>" for path, body in kept]
        )
        dropped = len(blocks) - limit
        print(f"    ⚠️ [{role_name}] 输出文件数超限，已截断 {dropped} 个文件")
        return trimmed, dropped

    def _run_agent_once(self, agent: Agent) -> Dict[str, Any]:
        agent_report: Dict[str, Any] = {"role_name": agent.role_name}
        try:
            agent.run()
            output_content = self.shared_memory.get_latest_output(agent.role_name) or ""
            output_content, dropped = self._limit_agent_outputs(output_content, agent.role_name)
            with self._file_write_lock:
                saved_files = save_files_from_content(
                    output_content,
                    self.output_dir,
                    update_mode=self._is_update_mode(),
                    reserved_paths=self._round_saved_files,
                )
            if saved_files:
                self.shared_memory.add_saved_files(saved_files)
            agent_report.update(
                {
                    "status": "completed",
                    "output_chars": len(output_content) if isinstance(output_content, str) else 0,
                    "saved_files": saved_files,
                    "dropped_files": dropped,
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
            if not self._is_safe_output_dir(self.output_dir):
                raise RuntimeError(f"Unsafe output_dir for deletion: {self.output_dir}")
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
        context_cfg = self.config.get("context", {})
        max_files = int(context_cfg.get("max_files", max_files))
        max_chars = int(context_cfg.get("max_chars", max_chars))
        skip_dirs = set(
            context_cfg.get(
                "skip_dirs",
                [
                    ".git",
                    ".venv",
                    "__pycache__",
                    ".pytest_cache",
                    "output",
                    "data",
                    "node_modules",
                    ".mypy_cache",
                ],
            )
        )
        allowed_suffixes = set(
            context_cfg.get("allowed_suffixes", [".py", ".md", ".yaml", ".yml", ".txt"])
        )
        deny_globs = context_cfg.get(
            "deny_globs",
            [
                "**/.env",
                "**/*.env",
                "**/*secret*",
                "**/*token*",
                "**/*password*",
                "**/*credential*",
                "**/*key*",
                "**/secrets/**",
            ],
        )
        sensitive_keywords = context_cfg.get(
            "sensitive_keywords",
            ["api_key", "apikey", "secret", "token", "password", "credential"],
        )
        sensitive_re = re.compile("|".join(sensitive_keywords), re.IGNORECASE) if sensitive_keywords else None

        lines: list[str] = [f"root={root}"]
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
                rel_posix = rel.as_posix()
                if any(fnmatch.fnmatch(rel_posix, pattern) for pattern in deny_globs):
                    continue
                try:
                    if suffix == ".py":
                        content = path.read_text(encoding="utf-8", errors="ignore")
                        if sensitive_re and sensitive_re.search(content):
                            continue
                        summary = CodeSummarizer.summarize_python(content)
                    else:
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read(600)
                        if sensitive_re and sensitive_re.search(content):
                            continue
                        summary = content[:400]
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

    def _is_safe_output_dir(self, path: Path) -> bool:
        resolved = path.resolve()
        if resolved in (Path("/"), Path.home()):
            return False
        if self.iteration_target and resolved == self.iteration_target.resolve():
            return False
        project_root = find_project_root(Path(__file__))
        if resolved == project_root:
            return False
        try:
            resolved.relative_to(project_root)
        except ValueError:
            return False
        return True

    def _is_update_mode(self) -> bool:
        if self.iteration_target:
            return True
        raw = self.shared_memory.global_context.get("requirements")
        if not raw:
            return False
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            return False
        if not isinstance(data, dict):
            return False
        if data.get("update_mode"):
            return True
        return any(key in data for key in ("existing_project", "iteration_target"))
    def _requires_final_approval(self) -> bool:
        requirements = self.shared_memory.global_context.get("requirements", "")
        text = str(requirements).lower()
        keywords = ("规划", "项目", "发布", "上线", "交付", "release", "launch", "plan", "project")
        return any(keyword in text for keyword in keywords)

    def _requires_ui_baseline_from_requirements(self, requirements: Any) -> bool:
        return self._ui_policy.requires_ui_baseline_from_requirements(requirements)

    def _requires_ui_baseline(self) -> bool:
        return self._ui_policy.requires_ui_baseline()

    def _requires_ui_functional_audit(self) -> bool:
        return self._ui_policy.requires_ui_functional_audit(self._get_requirements_payload())

    def _allow_missing_ui_baseline(self) -> bool:
        return self._ui_policy.allow_missing_baseline()

    def _ui_baseline_required(self) -> bool:
        return self._ui_policy.baseline_required()

    def _user_baseline_configured(self) -> bool:
        return self._ui_policy.user_baseline_configured()

    def _should_force_ui_design(self) -> bool:
        return self._ui_policy.should_force_ui_design()

    def _detect_frontend_presence(self, root: Path, max_files: int = 300) -> bool:
        return self._ui_policy.detect_frontend_presence(root, max_files=max_files)

    def _should_require_ui_tests(self) -> bool:
        return self.config.get("quality_gates", {}).get("require_ui_tests", True)

    def _should_require_coverage(self) -> bool:
        return self.config.get("quality_gates", {}).get("require_coverage", False)

    def _should_require_ui_simulation(self) -> bool:
        return self.config.get("quality_gates", {}).get("require_ui_simulation", True)

    def _apply_default_requirements(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(requirements, dict):
            return requirements
        derived = requirements.get("derived_requirements")
        if not isinstance(derived, list):
            derived = []

        def _add(item: str) -> None:
            if item not in derived:
                derived.append(item)

        _add("必须生成自动化测试文件，覆盖核心路径与边界场景")
        if self._should_require_coverage():
            _add("必须提供测试覆盖率结果（包含前端模块）")
        if self._requires_ui_baseline_from_requirements(requirements):
            _add("前端需先用 nanobanna 生成设计基线图，由 AI 读图形成实现摘要并据此开发")
            _add("前端必须具备 UI/交互测试，并纳入覆盖率门禁")
            _add("交付时必须提供前端实现截图作为验收依据")
            requirements.setdefault("ui_required", True)
        requirements["derived_requirements"] = derived

        acceptance = requirements.get("acceptance_criteria")
        if not isinstance(acceptance, list):
            acceptance = []
        if "测试通过" not in acceptance:
            acceptance.append("测试通过")
        if self._should_require_coverage() and "覆盖率达标" not in acceptance:
            acceptance.append("覆盖率达标")
        if self._requires_ui_baseline_from_requirements(requirements):
            if "UI 实现截图齐全" not in acceptance:
                acceptance.append("UI 实现截图齐全")
            if self._user_baseline_configured():
                if "UI 设计基线与实现截图齐全" not in acceptance:
                    acceptance.append("UI 设计基线与实现截图齐全")
            elif self._allow_missing_ui_baseline():
                optional_text = "UI 设计基线（如提供）与实现截图齐全"
                if optional_text not in acceptance:
                    acceptance.append(optional_text)
        requirements["acceptance_criteria"] = acceptance
        return requirements

    def _check_ui_evidence(self) -> Dict[str, Any]:
        return self._ui_policy.check_ui_evidence()

    def _load_json_template(self, filename: str) -> Dict[str, Any]:
        template_dir = self._get_review_template_dir()
        path = template_dir / filename
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _register_bug_card(self, card_path: Path, reason: str, summary: str) -> None:
        entry = {
            "path": str(card_path),
            "reason": reason,
            "summary": summary,
            "created_at": self._utcnow(),
        }
        existing = self.shared_memory.global_context.get("bug_cards")
        if not isinstance(existing, list):
            existing = []
        existing.append(entry)
        self.shared_memory.global_context["bug_cards"] = existing

    def _write_bug_card(self, reason: str, round_report: Dict[str, Any]) -> Optional[Path]:
        bug_dir = self.output_dir / "evidence" / "bugs"
        bug_dir.mkdir(parents=True, exist_ok=True)
        template = self._load_json_template("bug_card.json")
        if not template:
            template = {
                "id": "",
                "type": "逻辑",
                "severity": "S2",
                "scope": "模块",
                "repro_steps": [],
                "expected": "",
                "actual": "",
                "evidence": [],
                "owner": "dev_team",
                "status": "open",
            }
        timestamp = self._utcnow().replace(":", "-")
        bug_id = f"BUG-{timestamp}"
        type_map = {
            "ui_design_failed": "UI",
            "ui_evidence_missing": "UI",
            "ui_tests_failed": "UI",
            "coverage_failed": "质量",
            "user_simulation_failed": "逻辑",
            "input_contract_failed": "输入",
            "acceptance_failed": "质量",
            "tests_failed": "逻辑",
            "qa_failed": "质量",
            "consensus_failed": "一致性",
            "validation_failed": "一致性",
            "approval_failed": "发布",
        }
        severity_map = {
            "ui_design_failed": "S2",
            "ui_evidence_missing": "S3",
            "ui_tests_failed": "S2",
            "coverage_failed": "S2",
            "user_simulation_failed": "S2",
            "input_contract_failed": "S2",
            "acceptance_failed": "S2",
            "tests_failed": "S2",
            "qa_failed": "S3",
            "consensus_failed": "S3",
            "validation_failed": "S3",
            "approval_failed": "S3",
        }
        evidence_paths: List[str] = []
        evidence_dir = self.output_dir / "evidence"
        manifest_path = evidence_dir / "manifest.json"
        evidence_paths.append(str(manifest_path))
        ui_dir = evidence_dir / "ui"
        if ui_dir.exists():
            for item in ui_dir.glob("*"):
                if item.is_file():
                    evidence_paths.append(str(item))
        card = dict(template)
        card.update(
            {
                "id": bug_id,
                "type": type_map.get(reason, card.get("type", "逻辑")),
                "severity": severity_map.get(reason, card.get("severity", "S2")),
                "scope": card.get("scope", "模块"),
                "repro_steps": ["查看证据与日志", f"触发原因: {reason}"],
                "expected": "所有门禁通过并生成完整证据链",
                "actual": f"门禁失败: {reason}",
                "evidence": evidence_paths or card.get("evidence", []),
                "owner": card.get("owner", "dev_team"),
                "status": "open",
                "metadata": {
                    "round": round_report.get("round"),
                    "tests": round_report.get("tests"),
                    "ui_tests": round_report.get("ui_tests"),
                    "coverage": round_report.get("coverage"),
                    "input_contract": round_report.get("input_contract"),
                    "qa_feedback": round_report.get("qa_feedback"),
                },
            }
        )
        bug_path = bug_dir / f"{bug_id}.json"
        bug_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
        self._register_bug_card(bug_path, reason, card.get("actual", ""))
        return bug_path

    def _should_require_tests(self) -> bool:
        return self.config.get("quality_gates", {}).get("require_tests", True)

    def _qa_feedback_failed(self, qa_reports: List[Dict[str, Any]]) -> bool:
        if not qa_reports:
            return False
        keywords = ("不通过", "fail", "failed", "reject", "rejected")
        for report in qa_reports:
            feedback = str(report.get("feedback", "")).lower()
            if any(keyword in feedback for keyword in keywords):
                return True
        return False

    def _should_block_on_consensus(self, consensus_result: Any) -> bool:
        cfg = self.config.get("quality_gates", {}).get("consensus", {})
        if cfg.get("enabled", True) is False:
            return False
        min_confidence = cfg.get("min_confidence", 0.8)
        require_convergence = cfg.get("require_convergence", True)
        if require_convergence and not getattr(consensus_result, "convergence_achieved", False):
            return True
        return getattr(consensus_result, "confidence", 0.0) < min_confidence

    def _should_block_on_validation(self, validation_report: Any) -> bool:
        cfg = self.config.get("quality_gates", {}).get("validation", {})
        if cfg.get("enabled", True) is False:
            return False
        min_consistency = cfg.get("min_consistency", 0.8)
        allow_conflicts = cfg.get("allow_conflicts", False)
        conflicts = getattr(validation_report, "conflicts", [])
        if conflicts and not allow_conflicts:
            return True
        return getattr(validation_report, "consistency_score", 1.0) < min_consistency

    def _write_evidence_manifest(self, round_report: Dict[str, Any]) -> Path:
        evidence_dir = self.output_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "generated_at": self._utcnow(),
            "ui_design": self.shared_memory.global_context.get("ui_design"),
            "ui_design_summary": self.shared_memory.global_context.get("ui_design_summary"),
            "evidence": round_report.get("evidence", []),
            "gate_evidence": round_report.get("gate_evidence", []),
            "gate_decision": round_report.get("gate_decision"),
            "ui_evidence": round_report.get("ui_evidence"),
            "user_simulation": round_report.get("user_simulation"),
            "input_contract": round_report.get("input_contract"),
            "acceptance": round_report.get("acceptance"),
            "tests": round_report.get("tests"),
            "ui_tests": round_report.get("ui_tests"),
            "coverage": round_report.get("coverage"),
            "bug_cards": self.shared_memory.global_context.get("bug_cards", []),
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

    @staticmethod
    def _serialize_phase_result(result: Any) -> Optional[Dict[str, Any]]:
        if result is None:
            return None
        if isinstance(result, dict):
            return result
        if hasattr(result, "to_dict"):
            try:
                return result.to_dict()
            except Exception:
                return {"raw": str(result)}
        return {"raw": str(result)}

    def _enable_consensus(self) -> bool:
        return False

    def _enable_cross_validation(self) -> bool:
        return False

    def _run_consensus_phase(self, round_num: int) -> Any:
        return None

    def _run_cross_validation_phase(self) -> Any:
        return None

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

        acceptance_criteria: List[str] = []
        for jd in final_jds:
            items = jd.get("acceptance_criteria")
            if isinstance(items, list):
                acceptance_criteria.extend([str(item).strip() for item in items if str(item).strip()])
            elif isinstance(items, str) and items.strip():
                acceptance_criteria.append(items.strip())
        if acceptance_criteria:
            seen: set[str] = set()
            acceptance_criteria = [item for item in acceptance_criteria if not (item in seen or seen.add(item))]
            if isinstance(requirements, dict) and "acceptance_criteria" not in requirements:
                requirements["acceptance_criteria"] = acceptance_criteria
            self.shared_memory.global_context["acceptance_criteria"] = acceptance_criteria

        if self.iteration_target:
            if isinstance(requirements, dict):
                requirements.setdefault("update_mode", True)
                requirements.setdefault("iteration_target", str(self.iteration_target))
                requirements.setdefault(
                    "update_rules",
                    [
                        "基于现有代码修改对应函数，不要新建全新项目结构。",
                        "只输出修改过的文件，且必须包含完整实现。",
                        "禁止占位代码（.../pass）与伪代码。",
                    ],
                )

        if isinstance(requirements, dict):
            requirements = self._apply_default_requirements(requirements)
        self.shared_memory.global_context["requirements"] = json.dumps(requirements, ensure_ascii=False)

        print(f"\n>>> 正在组建开发团队，检测到 {len(final_jds)} 个角色需求...")

        for jd in final_jds:
            role_name = jd.get("role_name", "Unknown")
            agent = self.agent_factory(jd, self.config, self.shared_memory, self.output_dir)

            is_qa = agent.role_type == "QA" or "QA" in role_name or "Test" in role_name
            is_final_approver = self._is_final_approver_role(role_name)

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
            self._worker_pool.add_agent(
                agent,
                role_name,
                is_qa=is_qa,
                is_final_approver=is_final_approver,
            )

    def _is_final_approver_role(self, role_name: str) -> bool:
        if any(token in role_name for token in ("负责人", "产品", "规划")):
            return True
        lowered = role_name.lower()
        if re.search(r"\b(pm|planner|product|owner)\b", lowered):
            return True
        if re.search(r"\b(project manager|product manager|project owner|product owner)\b", lowered):
            return True
        return False

    def run_collaboration(self, max_rounds: int = 5):
        started_at = self._utcnow()
        self.run_reports = []
        max_rounds = self._apply_rounds_policy(max_rounds)
        start_round = self._resume_if_available(max_rounds)
        agents = self._worker_pool.agents()
        qa_agents = self._worker_pool.qa_agents()
        if not agents:
            print("❌ 错误: 团队未初始化或没有工程师角色。")
            report = self._build_report("no_engineers", started_at)
            self._write_report(report)
            return {
                "status": "error",
                "error": "no_engineers",
                "outputs": self.shared_memory.get_all_outputs(),
                "report": report,
            }

        review_report = self._policy_registry.run_first("review") or self._ensure_review_artifacts()
        self.shared_memory.global_context["review_artifacts"] = review_report
        if review_report.get("status") not in ("passed", "skipped"):
            report = self._build_report("review_missing", started_at)
            report["review"] = review_report
            self._write_report(report)
            return {
                "status": "review_missing",
                "outputs": self.shared_memory.get_all_outputs(),
                "report": report,
            }

        compliance_report = self._policy_registry.run_first("compliance") or {"status": "skipped"}
        self.shared_memory.global_context["compliance_report"] = compliance_report
        self._emit_event("compliance_checked", {"status": compliance_report.get("status", "unknown")})
        if (
            compliance_report.get("status") == "failed"
            and self.config.get("compliance", {}).get("block_on_failure", False)
        ):
            report = self._build_report("compliance_failed", started_at)
            report["compliance"] = compliance_report
            self._write_report(report)
            return {
                "status": "compliance_failed",
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

        requirements = self._get_requirements_payload()
        ui_design_report = self._policy_registry.run_first("ui_baseline", requirements=requirements)
        self.shared_memory.global_context["ui_design_report"] = ui_design_report
        allow_missing_ui = self._allow_missing_ui_baseline()
        if (
            ui_design_report.get("status") == "failed"
            and self.config.get("ui_design", {}).get("required", True)
            and not allow_missing_ui
        ):
            report = self._build_report("ui_design_failed", started_at)
            report["ui_design"] = ui_design_report
            self._write_report(report)
            self._save_resume_state(0, "ui_design_failed")
            return {
                "status": "ui_design_failed",
                "outputs": self.shared_memory.get_all_outputs(),
                "report": report,
            }

        self._emit_event(
            "collaboration_started",
            {"max_rounds": max_rounds, "start_round": start_round},
        )
        print(f"\n🚀 启动 TDD 协作流程 (最大轮次: {max_rounds})...")
        run_status = "max_rounds_reached"
        testing_cfg = self.config.get("testing", {})
        testing_enabled = testing_cfg.get("enabled", True)
        for round_num in range(start_round, max_rounds + 1):
            self._emit_event("round_started", {"round": round_num})
            print(f"\n🔄 --- 第 {round_num} 轮迭代 ---")
            round_plan = self._coordinator.build_round_plan(
                round_num=round_num,
                enable_consensus=self._enable_consensus() and len(agents) > 1,
                enable_cross_validation=self._enable_cross_validation() and len(agents) > 1,
                has_qa=bool(qa_agents),
            )
            round_report = {
                "round": round_num,
                "plan": round_plan,
                "agents": [],
                "tests": {},
                "qa_feedback_recorded": False,
                "consensus": None,
                "validation": None,
            }
            self._round_saved_files = set()
            failure_reasons: List[str] = []
            round_report["milestone_state"] = self._coordinator.transition("start_round", round_num=round_num)

            round_report["agents"] = self._execution_service.run_agents(agents)
            round_report["milestone_state"] = self._coordinator.transition("agents_done", round_num=round_num)

            refactor_suggestions = self._collect_refactor_suggestions(self.output_dir)
            round_report["refactor_suggestions"] = refactor_suggestions
            self.shared_memory.global_context["refactor_suggestions"] = refactor_suggestions
            if refactor_suggestions:
                print("    ⚠️ 发现需要拆解的长函数，已生成建议。")

            if self._enable_consensus() and len(agents) > 1:
                print("\n🤝 阶段2: 共识达成")
                consensus_result = self._policy_registry.run_first("consensus", round_num=round_num)
                round_report["consensus"] = self._serialize_phase_result(consensus_result)
                if consensus_result is not None and self._should_block_on_consensus(consensus_result):
                    failure_reasons.append("consensus_failed")

            print("\n🧪 阶段3: 自动化测试")
            print("    🧪 [System] 正在执行自动化测试/语法检查...")
            verification_results = self._verification_service.run_round(
                str(self.output_dir),
                ui_required=self._requires_ui_functional_audit(),
                testing_enabled=testing_enabled,
            )
            round_report.update(verification_results)
            test_output = verification_results["tests"]["output"]
            summary = verification_results["tests"]["summary"]
            print(f"    📋 测试结果摘要: {summary}")
            test_status = verification_results["tests"]["status"]
            self.shared_memory.global_context["latest_test_results"] = test_output
            self.shared_memory.global_context["latest_test_status"] = test_status

            if self._enable_cross_validation() and len(agents) > 1:
                print("\n🔍 阶段4: 交叉核查")
                validation_report = self._policy_registry.run_first("verification")
                round_report["validation"] = self._serialize_phase_result(validation_report)
                if validation_report is not None and self._should_block_on_validation(validation_report):
                    failure_reasons.append("validation_failed")

            ui_test_status = round_report["ui_tests"]["status"]
            coverage_status = round_report["coverage"]["status"]
            input_status = round_report["input_contract"]["status"]
            round_report["milestone_state"] = self._coordinator.transition("tests_done", round_num=round_num)
            round_report["milestone_state"] = self._coordinator.transition("gating", round_num=round_num)

            if test_status == "passed":
                print("    ✨ 自动化测试全部通过！")
                if input_status in ("failed", "error"):
                    print("    ❌ 输入契约测试未通过，进入修复回合。")
                    failure_reasons.append("input_contract_failed")
                if self._requires_ui_functional_audit() and self._should_require_ui_tests():
                    if ui_test_status in ("failed", "error", "skipped", "unknown"):
                        print("    ❌ UI 测试未通过或缺失，进入修复回合。")
                        failure_reasons.append("ui_tests_failed")
                if self._should_require_coverage():
                    if coverage_status in ("failed", "error", "skipped", "unknown"):
                        print("    ❌ 覆盖率统计未通过或缺失，进入修复回合。")
                        failure_reasons.append("coverage_failed")
                if self._requires_ui_functional_audit():
                    ui_check = self._check_ui_evidence()
                    round_report["ui_evidence"] = ui_check
                    if ui_check["status"] != "passed":
                        print("    ❌ UI 证据不完整，进入修复回合。")
                        failure_reasons.append("ui_evidence_missing")
                sim_status = round_report["user_simulation"]["status"]
                if self._requires_ui_functional_audit() and self._should_require_ui_simulation():
                    if sim_status not in ("passed",):
                        print("    ❌ 用户模拟测试缺失或失败，进入修复回合。")
                        failure_reasons.append("user_simulation_failed")
                elif sim_status not in ("passed", "skipped", "unknown"):
                    print("    ❌ 用户模拟测试未通过，进入修复回合。")
                    failure_reasons.append("user_simulation_failed")
                acceptance_criteria = self._get_acceptance_criteria(self._get_requirements_payload())
                acceptance_report = self._verify_acceptance_checklist(acceptance_criteria)
                round_report["acceptance"] = acceptance_report
                if acceptance_report.get("status") == "failed":
                    print("    ❌ 验收清单未完成，进入修复回合。")
                    failure_reasons.append("acceptance_failed")

                if failure_reasons:
                    round_report["failure_reasons"] = failure_reasons
                    self._finalize_round_report(round_report)
                    self._write_bug_card(failure_reasons[0], round_report)
                    self._write_evidence_manifest(round_report)
                    self._archive_evidence()
                    if "gate_decision" not in round_report:
                        self._finalize_round_report(round_report)
                    self.run_reports.append(round_report)
                    self._save_resume_state(round_num, "in_progress")
                    self._sync_iteration_artifacts()
                    run_status = failure_reasons[0]
                    if round_num < max_rounds:
                        continue
                    break

                self._finalize_round_report(round_report)
                self._write_evidence_manifest(round_report)
                self._archive_evidence()
                approved = self._run_approval_gate(round_num, test_status, round_report)
                if not approved:
                    print("    ⚠️ 审批未通过，进入修复回合。")
                    round_report["failure_reasons"] = ["approval_failed"]
                    self._finalize_round_report(round_report)
                    self._write_bug_card("approval_failed", round_report)
                    self._write_evidence_manifest(round_report)
                    self._archive_evidence()
                    if "gate_decision" not in round_report:
                        self._finalize_round_report(round_report)
                    self.run_reports.append(round_report)
                    self._save_resume_state(round_num, "in_progress")
                    self._sync_iteration_artifacts()
                    run_status = "approval_failed"
                    if round_num < max_rounds:
                        continue
                    break
                if "gate_decision" not in round_report:
                    self._finalize_round_report(round_report)
                self.run_reports.append(round_report)
                print("    >>> 提前结束协作循环。")
                run_status = "passed"
                break

            if qa_agents:
                print(f"\n🔍 [QA] 正在进行代码审查与测试分析...")
                qa_reports = []
                max_workers = min(4, max(1, len(qa_agents)))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(
                            self._run_qa_agent,
                            agent,
                            round_num,
                            "Passed" if test_status == "passed" else "Failed",
                        )
                        for agent in qa_agents
                    ]
                    for future in as_completed(futures):
                        qa_reports.append(future.result())
                round_report["qa_feedback_recorded"] = True
                round_report["qa_feedback"] = qa_reports
                print(f"    📝 QA 反馈已记录")

            if test_status in ("failed", "error"):
                failure_reasons.append("tests_failed")
            elif test_status in ("skipped", "unknown") and self._should_require_tests():
                failure_reasons.append("tests_failed")
            if self._requires_ui_functional_audit() and self._should_require_ui_tests():
                if ui_test_status in ("failed", "error", "skipped", "unknown"):
                    failure_reasons.append("ui_tests_failed")
            if self._should_require_coverage():
                if coverage_status in ("failed", "error", "skipped", "unknown"):
                    failure_reasons.append("coverage_failed")

            if failure_reasons:
                round_report["failure_reasons"] = failure_reasons
                self._finalize_round_report(round_report)
                self._write_bug_card(failure_reasons[0], round_report)
                self._write_evidence_manifest(round_report)
                self._archive_evidence()
                run_status = failure_reasons[0]
            if "gate_decision" not in round_report:
                self._finalize_round_report(round_report)
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
