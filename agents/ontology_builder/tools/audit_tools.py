from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

JsonDict = dict[str, Any]


def _ensure_list_of_dict(value: object) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def build_design_audit_prompt(prompt: str, design_data: JsonDict, issues: list[JsonDict]) -> str:
    base = json.dumps(design_data, ensure_ascii=False, indent=2)
    issue_text = json.dumps(issues, ensure_ascii=False, indent=2)
    instruction = (
        "请基于设计数据与规则审计问题输出JSON：\n"
        "{\n"
        '  "status": "pass|fail",\n'
        '  "issues": [{"category":"coverage|relation|mapping|other","message":"...","severity":"high|medium|low","fix":"..."}]\n'
        "}\n"
        "只输出JSON。"
    )
    return "\n\n".join([prompt, instruction, "设计数据：", base, "规则审计问题：", issue_text]).strip()


def ai_audit_design(
    design_data: JsonDict,
    issues: list[JsonDict],
    llm_config_path: Path | None,
    load_llm_config: Callable[[Path], dict[str, Any]],
    build_llm: Callable[[dict[str, Any]], Any],
    invoke_llm: Callable[[Any, str], str],
    extract_json_payload: Callable[[str], Any],
    load_prompt: Callable[[str], str],
) -> tuple[JsonDict, str | None]:
    prompt_text = build_design_audit_prompt(load_prompt("design_audit"), design_data, issues)
    try:
        llm_path = llm_config_path or Path("configs/llm.yaml")
        config = load_llm_config(llm_path)
        llm = build_llm(config)
        response = invoke_llm(llm, prompt_text)
        payload = extract_json_payload(response)
        if not isinstance(payload, dict):
            raise ValueError("审计输出无效")
        merged_issues = _ensure_list_of_dict(payload.get("issues")) + issues
        status = "pass" if not merged_issues else "fail"
        return {"status": status, "issues": merged_issues}, None
    except Exception as exc:
        status = "pass" if not issues else "fail"
        return {"status": status, "issues": issues}, f"AI3 审计失败: {exc}"


def _audit_owl_rules(owl_text: str, design_data: JsonDict) -> list[JsonDict]:
    issues: list[JsonDict] = []
    if "# 关系定义" not in owl_text:
        issues.append({"category": "owl_format", "message": "缺少关系定义区块", "severity": "high"})
    object_ids = {str(obj.get("id")) for obj in _ensure_list_of_dict(design_data.get("objects"))}
    for line in owl_text.splitlines():
        if " a owl:ObjectProperty" not in line:
            continue
        if "rdfs:subPropertyOf ex:custom_relation" not in line:
            continue
        if "rdfs:domain" not in line or "rdfs:range" not in line:
            issues.append({"category": "relation", "message": "关系缺少domain/range", "severity": "high"})
        domain_match = re.search(r"rdfs:domain ex:([A-Za-z0-9_]+)", line)
        range_match = re.search(r"rdfs:range ex:([A-Za-z0-9_]+)", line)
        if domain_match and domain_match.group(1) not in object_ids:
            issues.append(
                {
                    "category": "relation",
                    "message": f"关系domain未指向对象类: {domain_match.group(1)}",
                    "severity": "high",
                }
            )
        if range_match and range_match.group(1) not in object_ids:
            issues.append(
                {
                    "category": "relation",
                    "message": f"关系range未指向对象类: {range_match.group(1)}",
                    "severity": "high",
                }
            )
    return issues


def build_owl_audit_prompt(prompt: str, owl_text: str, issues: list[JsonDict]) -> str:
    instruction = (
        "请基于OWL与规则审计问题输出JSON：\n"
        "{\n"
        '  "status": "pass|fail",\n'
        '  "issues": [{"category":"owl_format|relation|other","message":"...","severity":"high|medium|low","fix":"..."}]\n'
        "}\n"
        "只输出JSON。"
    )
    issue_text = json.dumps(issues, ensure_ascii=False, indent=2)
    return "\n\n".join([prompt, instruction, "OWL内容：", owl_text, "规则审计问题：", issue_text]).strip()


def ai_audit_owl(
    owl_text: str,
    design_data: JsonDict,
    llm_config_path: Path | None,
    load_llm_config: Callable[[Path], dict[str, Any]],
    build_llm: Callable[[dict[str, Any]], Any],
    invoke_llm: Callable[[Any, str], str],
    extract_json_payload: Callable[[str], Any],
    load_prompt: Callable[[str], str],
) -> tuple[JsonDict, str | None]:
    issues = _audit_owl_rules(owl_text, design_data)
    prompt_text = build_owl_audit_prompt(load_prompt("owl_audit"), owl_text, issues)
    try:
        llm_path = llm_config_path or Path("configs/llm.yaml")
        config = load_llm_config(llm_path)
        llm = build_llm(config)
        response = invoke_llm(llm, prompt_text)
        payload = extract_json_payload(response)
        if not isinstance(payload, dict):
            raise ValueError("OWL审计输出无效")
        merged_issues = _ensure_list_of_dict(payload.get("issues")) + issues
        status = "pass" if not merged_issues else "fail"
        return {"status": status, "issues": merged_issues}, None
    except Exception as exc:
        status = "pass" if not issues else "fail"
        return {"status": status, "issues": issues}, f"AI5 审计失败: {exc}"
