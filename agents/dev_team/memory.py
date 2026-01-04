from agents.dev_team.code_summarizer import CodeSummarizer
from agents.dev_team.utils import parse_code_blocks
import re
import json
import threading
from typing import List, Dict, Any

class SharedMemoryStore:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.global_context: Dict[str, Any] = {}
        self.role_outputs: Dict[str, List[str]] = {}
        self.qa_feedback: List[Dict[str, Any]] = []
        self.saved_files: Dict[str, str] = {} # Path -> Content or Metadata
        self._lock = threading.Lock()

    def add_qa_feedback(self, feedback: Dict[str, Any]):
        with self._lock:
            self.qa_feedback.append(feedback)

    def add_output(self, role: str, content: str):
        with self._lock:
            if role not in self.role_outputs:
                self.role_outputs[role] = []
            self.role_outputs[role].append(content)

    def add_saved_files(self, file_paths: List[str]):
        with self._lock:
            for path in file_paths:
                self.saved_files[path] = "saved"

    def get_all_outputs(self) -> Dict[str, List[str]]:
        with self._lock:
            return {k: list(v) for k, v in self.role_outputs.items()}

    def get_latest_output(self, role: str) -> str | None:
        with self._lock:
            outputs = self.role_outputs.get(role)
            if not outputs:
                return None
            return outputs[-1]

    def get_context_for_role(self, requesting_role: str) -> str:
        context_parts = []
        view_mode = "summary"
        summary_max_chars = 400

        project_context = self.global_context.get("project_context")
        if project_context:
            context_parts.append(f"### 项目现有代码基线:\n{project_context}")
        
        # 1. 包含其他角色的最新输出 (使用 AST 智能摘要)
        for role, outputs in self.role_outputs.items():
            if role != requesting_role and outputs:
                latest_output = outputs[-1]
                
                # 尝试提取代码块并进行 AST 摘要
                code_blocks = parse_code_blocks(latest_output)
                
                summarized_output = latest_output
                if code_blocks:
                    # 如果包含代码文件，则替换为摘要
                    summarized_parts = []
                    
                    # To keep it simple and consistent with new parser:
                    # Remove code blocks from text to avoid duplication
                    # This regex matches the XML style. For markdown loop we might need adjustment later.
                    text_without_code = re.sub(r'<file path=[\"\'].*?[\"\']>.*?</file>', '[Code File Omitted]', latest_output, flags=re.DOTALL)
                    summarized_parts.append(text_without_code[:500]) # Keep some context text

                    for path, content in code_blocks:
                        # 处理代码摘要
                        if path.endswith(".py"):
                            summary = CodeSummarizer.summarize_python(content)
                            summary_block = f"\n<file path=\"{path}\">\n{summary}\n</file>\n"
                            summarized_parts.append(summary_block)
                        else:
                            # 非 Python 文件还是做简单截断
                            if len(content) > 500:
                                content = content[:200] + "\n...[Truncated]...\n" + content[-200:]
                            summarized_parts.append(f"\n<file path=\"{path}\">\n{content}\n</file>\n")
                            
                    summarized_output = "\n".join(summarized_parts)
                
                if view_mode == "summary":
                    summarized_output = self._summarize_text(summarized_output, summary_max_chars)

                limit = self.config.get("memory", {}).get("context_limit", 3000)
                if len(summarized_output) > limit:
                    summarized_output = summarized_output[:limit // 2] + "\n...[Context Limit Reached]...\n" + summarized_output[-(limit // 3):]
                    
                context_parts.append(f"### 来自角色 [{role}] 的最新进展:\n{summarized_output}")

        # 2. 包含最近的 QA 反馈（如果有）
        if self.qa_feedback:
            last_feedback = self.qa_feedback[-1]
            feedback_str = json.dumps(last_feedback, ensure_ascii=False, indent=2)
            if len(feedback_str) > 3000:
                feedback_str = feedback_str[:3000] + "\n...[Feedback Truncated]..."
            context_parts.append(f"### 最近的 QA 审查反馈:\n{feedback_str}")

        # 3. 包含文件系统概览 (Simulated for now as we don't track saved files deeply yet)
        # In a real implementation, RoleAgent should update memory.saved_files
        if self.saved_files:
            files_list = "\n".join([f"- {path}" for path in self.saved_files.keys()])
            context_parts.append(f"### 当前已生成的文件列表:\n{files_list}")

        return "\n\n".join(context_parts) if context_parts else "暂无其他角色的上下文信息。"

    @staticmethod
    def _summarize_text(text: str, max_chars: int) -> str:
        if not text:
            return text
        stripped = text.strip()
        if len(stripped) <= max_chars:
            return stripped
        return stripped[:max_chars].rstrip() + "\n...[Summary Truncated]..."

    def clear(self):
        self.role_outputs.clear()
        self.saved_files.clear()
        self.qa_feedback.clear()
        self.global_context.clear()
