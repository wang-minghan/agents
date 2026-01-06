from agents.dev_team.code_summarizer import CodeSummarizer
from agents.dev_team.utils import parse_code_blocks
import re
import json
import threading
from datetime import datetime, timezone
from typing import List, Dict, Any

class SharedMemoryStore:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.global_context: Dict[str, Any] = {}
        self.role_outputs: Dict[str, List[str]] = {}
        self.qa_feedback: List[Dict[str, Any]] = []
        self.saved_files: Dict[str, str] = {} # Path -> Content or Metadata
        self._latest_output_summary: Dict[str, str] = {}
        self.domain_events: List[Dict[str, Any]] = []
        self.evidence: List[Dict[str, Any]] = []
        self.gate_decisions: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    @staticmethod
    def _is_qa_name(role_name: str) -> bool:
        if not role_name:
            return False
        upper = role_name.upper()
        return "QA" in upper or "TEST" in upper

    def _is_qa_role(self, role_name: str, role_type: str | None) -> bool:
        if role_type and role_type.upper() == "QA":
            return True
        return self._is_qa_name(role_name)

    def _summary_limits(
        self,
        summary_max_chars: int | None = None,
        context_limit: int | None = None,
    ) -> tuple[int, int]:
        memory_cfg = self.config.get("memory", {})
        summary_limit = summary_max_chars if summary_max_chars is not None else memory_cfg.get("summary_max_chars", 400)
        context_limit_value = context_limit if context_limit is not None else memory_cfg.get("context_limit", 3000)
        return summary_limit, context_limit_value

    def add_qa_feedback(self, feedback: Dict[str, Any]):
        with self._lock:
            self.qa_feedback.append(feedback)

    def add_output(self, role: str, content: str):
        with self._lock:
            if role not in self.role_outputs:
                self.role_outputs[role] = []
            self.role_outputs[role].append(content)
        summary = self._summarize_output(content)
        with self._lock:
            self._latest_output_summary[role] = summary

    def add_saved_files(self, file_paths: List[str]):
        with self._lock:
            for path in file_paths:
                self.saved_files[path] = "saved"

    @staticmethod
    def _utcnow() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def add_domain_event(self, name: str, payload: Dict[str, Any]):
        event = {"name": name, "payload": payload, "occurred_at": self._utcnow()}
        with self._lock:
            self.domain_events.append(event)

    def add_evidence(self, evidence_items: List[Dict[str, Any]]):
        if not evidence_items:
            return
        with self._lock:
            self.evidence.extend(evidence_items)

    def add_gate_decision(self, decision: Dict[str, Any]):
        with self._lock:
            self.gate_decisions.append(decision)

    def get_all_outputs(self) -> Dict[str, List[str]]:
        with self._lock:
            return {k: list(v) for k, v in self.role_outputs.items()}

    def get_latest_output(self, role: str) -> str | None:
        with self._lock:
            outputs = self.role_outputs.get(role)
            if not outputs:
                return None
            return outputs[-1]

    def get_peer_output_summaries(
        self,
        requesting_role: str,
        include_qa: bool = False,
        summary_max_chars: int | None = None,
    ) -> Dict[str, str]:
        summary_limit, context_limit = self._summary_limits(summary_max_chars=summary_max_chars)
        with self._lock:
            outputs_snapshot = {
                role: outputs[-1]
                for role, outputs in self.role_outputs.items()
                if outputs and role != requesting_role
            }
            summary_snapshot = dict(self._latest_output_summary)

        summaries: Dict[str, str] = {}
        for role, latest_output in outputs_snapshot.items():
            if not include_qa and self._is_qa_name(role):
                continue
            summarized_output = None
            if summary_max_chars is None:
                summarized_output = summary_snapshot.get(role)
            if summarized_output is None:
                summarized_output = self._summarize_output(
                    latest_output,
                    summary_max_chars=summary_limit,
                    context_limit=context_limit,
                )
                with self._lock:
                    self._latest_output_summary[role] = summarized_output
            if summarized_output:
                summaries[role] = summarized_output
        return summaries

    def get_context_for_role(self, requesting_role: str, role_type: str | None = None) -> str:
        context_parts = []
        memory_cfg = self.config.get("memory", {})
        is_qa = self._is_qa_role(requesting_role, role_type)
        include_role_outputs_for_qa = memory_cfg.get("include_role_outputs_for_qa", False)
        include_project_context_for_qa = memory_cfg.get("include_project_context_for_qa")

        project_context = self.global_context.get("project_context")
        acceptance_criteria = self.global_context.get("acceptance_criteria")
        bug_cards = self.global_context.get("bug_cards")
        ui_design = self.global_context.get("ui_design")
        ui_design_summary = self.global_context.get("ui_design_summary")
        
        # 1. 包含其他角色的最新输出 (使用 AST 智能摘要)
        with self._lock:
            outputs_snapshot = {
                role: outputs[-1]
                for role, outputs in self.role_outputs.items()
                if outputs and role != requesting_role
            }
            summary_snapshot = dict(self._latest_output_summary)
            last_feedback = self.qa_feedback[-1] if self.qa_feedback else None
            saved_files = list(self.saved_files.keys())

        has_outputs = bool(outputs_snapshot)
        include_project_context = True
        if is_qa:
            if isinstance(include_project_context_for_qa, bool):
                include_project_context = include_project_context_for_qa
            else:
                include_project_context = not has_outputs

        if project_context and include_project_context:
            context_parts.append(f"### 项目现有代码基线:\n{project_context}")

        if acceptance_criteria:
            criteria_text = "\n".join([f"- {item}" for item in acceptance_criteria])
            context_parts.append(f"### 验收清单:\n{criteria_text}")

        if bug_cards:
            bug_text = json.dumps(bug_cards, ensure_ascii=False, indent=2)
            if len(bug_text) > 3000:
                bug_text = bug_text[:3000] + "\n...[Bug Cards Truncated]..."
            context_parts.append(f"### Bug Card:\n{bug_text}")

        if ui_design:
            ui_text = json.dumps(ui_design, ensure_ascii=False, indent=2)
            context_parts.append(f"### UI 设计基线:\n{ui_text}")

        if ui_design_summary:
            summary_text = str(ui_design_summary)
            if len(summary_text) > 2000:
                summary_text = summary_text[:2000] + "\n...[Summary Truncated]..."
            context_parts.append(f"### UI 设计摘要:\n{summary_text}")

        if not is_qa or include_role_outputs_for_qa:
            for role, latest_output in outputs_snapshot.items():
                if not is_qa and self._is_qa_name(role):
                    continue
                summarized_output = summary_snapshot.get(role)
                if summarized_output is None:
                    summarized_output = self._summarize_output(latest_output)
                    with self._lock:
                        self._latest_output_summary[role] = summarized_output
                context_parts.append(f"### 来自角色 [{role}] 的最新进展:\n{summarized_output}")

        # 2. 包含最近的 QA 反馈（如果有）
        if last_feedback and not is_qa:
            feedback_str = json.dumps(last_feedback, ensure_ascii=False, indent=2)
            if len(feedback_str) > 3000:
                feedback_str = feedback_str[:3000] + "\n...[Feedback Truncated]..."
            context_parts.append(f"### 最近的 QA 审查反馈:\n{feedback_str}")

        # 3. 包含文件系统概览 (Simulated for now as we don't track saved files deeply yet)
        # In a real implementation, RoleAgent should update memory.saved_files
        if saved_files:
            files_list = "\n".join([f"- {path}" for path in saved_files])
            context_parts.append(f"### 当前已生成的文件列表:\n{files_list}")

        review_docs = self.global_context.get("review_docs")
        if review_docs:
            review_text = json.dumps(review_docs, ensure_ascii=False, indent=2)
            context_parts.append(f"### 审查产出路径:\n{review_text}")

        return "\n\n".join(context_parts) if context_parts else "暂无其他角色的上下文信息。"

    @staticmethod
    def _summarize_text(text: str, max_chars: int) -> str:
        if not text:
            return text
        stripped = text.strip()
        if len(stripped) <= max_chars:
            return stripped
        return stripped[:max_chars].rstrip() + "\n...[Summary Truncated]..."

    def _summarize_output(
        self,
        output: str,
        summary_max_chars: int | None = None,
        context_limit: int | None = None,
    ) -> str:
        summary_max_chars, context_limit = self._summary_limits(
            summary_max_chars=summary_max_chars,
            context_limit=context_limit,
        )
        summarized_output = output
        code_blocks = []
        if "<file" in output:
            code_blocks = parse_code_blocks(output)
        if code_blocks:
            summarized_parts = []
            text_without_code = re.sub(
                r'<file path=[\"\'].*?[\"\']>.*?</file>',
                '[Code File Omitted]',
                output,
                flags=re.DOTALL,
            )
            summarized_parts.append(text_without_code[:500])

            for path, content in code_blocks:
                if path.endswith(".py"):
                    summary = CodeSummarizer.summarize_python(content)
                    summary_block = f"\n<file path=\"{path}\">\n{summary}\n</file>\n"
                    summarized_parts.append(summary_block)
                else:
                    if len(content) > 500:
                        content = content[:200] + "\n...[Truncated]...\n" + content[-200:]
                    summarized_parts.append(f"\n<file path=\"{path}\">\n{content}\n</file>\n")

            summarized_output = "\n".join(summarized_parts)

        summarized_output = self._summarize_text(summarized_output, summary_max_chars)
        if len(summarized_output) > context_limit:
            summarized_output = (
                summarized_output[:context_limit // 2]
                + "\n...[Context Limit Reached]...\n"
                + summarized_output[-(context_limit // 3):]
            )
        return summarized_output

    def clear(self):
        self.role_outputs.clear()
        self.saved_files.clear()
        self.qa_feedback.clear()
        self.global_context.clear()
        self._latest_output_summary.clear()
