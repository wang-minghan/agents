import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.base import BaseCallbackHandler


class _StreamingPrintHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token:
            print(token, end="", flush=True)

def _encode_image(path: Path, max_bytes: int = 2_000_000) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    if path.stat().st_size > max_bytes:
        return None
    mime, _ = mimetypes.guess_type(path.name)
    if not mime:
        mime = "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


class _AnalyzerChain:
    def __init__(self, config: Dict[str, Any], template: str) -> None:
        role_config = config.get("roles", {}).get("requirement_analyzer", {})
        model_name = role_config.get("model", "gpt-4")
        temperature = role_config.get("temperature", 0.3)
        api_key = role_config.get("api_key")
        api_base = role_config.get("api_base")
        self._template = template
        self._parser = JsonOutputParser()
        self._llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=api_base,
            streaming=True,
            callbacks=[_StreamingPrintHandler()],
        )

    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        user_input = str(input_data.get("user_input") or "")
        constraints = input_data.get("constraints") or {}
        baseline_raw = str(constraints.get("design_baseline", "")).strip()
        impl_raw = str(constraints.get("implementation_snapshot", "")).strip()
        baseline_path = Path(baseline_raw) if baseline_raw else None
        impl_path = Path(impl_raw) if impl_raw else None

        prompt = (
            self._template
            + "\n\n请根据用户输入与任何附加的设计/实现证据进行事实分析。"
            + "\n如果提供了设计基线与实现截图，请指出新增的需求约束与可能的差异风险。"
            + "\n\n用户输入: "
            + user_input
            + "\n输出:"
        )

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if baseline_path and baseline_path.exists():
            encoded = _encode_image(baseline_path)
            if encoded:
                content.append({"type": "image_url", "image_url": {"url": encoded, "detail": "high"}})
            else:
                content.append({"type": "text", "text": f"[设计基线图无法加载或过大]: {baseline_path}"})
        if impl_path and impl_path.exists():
            encoded = _encode_image(impl_path)
            if encoded:
                content.append({"type": "image_url", "image_url": {"url": encoded, "detail": "high"}})
            else:
                content.append({"type": "text", "text": f"[实现截图无法加载或过大]: {impl_path}"})

        messages = [SystemMessage(content=self._template), HumanMessage(content=content)]
        response = self._llm.invoke(messages)
        raw = response.content if hasattr(response, "content") else response
        try:
            return self._parser.parse(raw)
        except Exception:
            return {"raw": raw}


def build_analyzer_chain(config: Dict[str, Any]):
    agent_root = Path(config.get("agent_root", "."))
    prompt_path = agent_root / "prompts/requirement_analyzer.md"
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    return _AnalyzerChain(config, template)
