"""Ontology builder agent.

Builds ontology design data and OWL from CSV using a fully AI-driven pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml
from pydantic import SecretStr

from agents.ontology_builder.tools.audit_tools import ai_audit_design, ai_audit_owl
from agents.ontology_builder.tools.csv_tools import build_table_summary, read_csv_rows
from agents.ontology_builder.tools.relation_defs import (
    RELATION_CODE_TO_ID,
    RELATION_DEFS,
    RELATION_LABEL_TO_ID,
)
from agents.ontology_builder.tools.render_tools import render_owl

@dataclass(frozen=True)
class OntologyObject:
    label: str
    code: str
    en_label: str | None = None
    source_table: str | None = None


@dataclass(frozen=True)
class OntologyAttribute:
    label: str
    code: str
    object_id: str
    raw_chinese: str
    raw_english: str
    source_table: str | None = None
    en_label: str | None = None
    data_type: str | None = None
    comment: str | None = None


@dataclass(frozen=True)
class RelationEdge:
    subject: str
    predicate: str
    object: str
    predicate_label: str
    domain_attribute: str | None = None
    range_attribute: str | None = None


JsonDict = dict[str, Any]

PROMPT_FILES = {
    "data_extract": "role_data_extract.md",
    "design": "role_design.md",
    "design_audit": "role_design_audit.md",
    "owl_audit": "role_owl_audit.md",
}

DEFAULT_TEMPLATE_PATH = Path(__file__).parent / "templates" / "ontology_template.ttl"

def _coerce_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _ensure_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _ensure_list_of_dict(value: object) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _normalize_key(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _load_llm_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Missing LLM config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}
    profiles = data.get("profiles")
    if isinstance(profiles, dict):
        active = str(data.get("active_profile") or "")
        selected = profiles.get(active) if active else None
        if selected is None and profiles:
            selected = next(iter(profiles.values()))
        merged: dict[str, Any] = {}
        common = data.get("common")
        if isinstance(common, dict):
            merged.update(common)
        if isinstance(selected, dict):
            merged.update(selected)
        return merged
    return data


def _build_llm(config: dict[str, Any]) -> Any:
    from langchain_openai import ChatOpenAI

    api_key_value = config.get("api_key")
    if not api_key_value:
        raise ValueError("LLM api_key is missing in config.")
    if isinstance(api_key_value, SecretStr):
        api_key = api_key_value
    else:
        api_key_text = str(api_key_value).strip()
        if not api_key_text:
            raise ValueError("LLM api_key is missing in config.")
        api_key = SecretStr(api_key_text)
    api_base = str(config.get("api_base", "https://api.deepseek.com"))
    model = str(config.get("model", "deepseek-chat"))
    timeout_seconds = _coerce_int(config.get("timeout_seconds"), 120)
    max_retries = _coerce_int(config.get("max_retries"), 1)
    return ChatOpenAI(
        api_key=api_key,
        base_url=api_base,
        model=model,
        temperature=0,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )


def _invoke_llm(llm: Any, prompt: str) -> str:
    response = llm.invoke(prompt)
    content = getattr(response, "content", None)
    return str(content if content is not None else response)


def _load_prompt(role: str) -> str:
    filename = PROMPT_FILES.get(role)
    if not filename:
        return ""
    prompt_path = Path(__file__).parent / "prompts" / filename
    if not prompt_path.exists():
        return ""
    return prompt_path.read_text(encoding="utf-8").strip()


def _extract_json_payload(text: str) -> JsonDict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _truncate_identifier(value: str, max_len: int = 64) -> str:
    if len(value) <= max_len:
        return value
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:6]
    keep = max_len - len(digest) - 1
    return f"{value[:keep]}_{digest}"


def _normalize_label_repeat(label: str) -> str:
    if not label:
        return label
    half = len(label) // 2
    if half > 0 and len(label) % 2 == 0 and label[:half] == label[half:]:
        return label[:half]
    return label


def _safe_object_id(label: str, code: str, used: set[str]) -> str:
    if code:
        cleaned = re.sub(r"[^A-Za-z0-9_]", "", code).strip("_")
        if cleaned:
            candidate = _truncate_identifier(cleaned)
            if candidate not in used:
                used.add(candidate)
                return candidate
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:8]
    candidate = _truncate_identifier(f"obj_{digest}")
    if candidate in used:
        suffix = 1
        while f"{candidate}_{suffix}" in used:
            suffix += 1
        candidate = _truncate_identifier(f"{candidate}_{suffix}")
    used.add(candidate)
    return candidate


def _safe_generic_id(label: str, code: str, used: set[str], prefix: str) -> str:
    if code:
        cleaned = re.sub(r"[^A-Za-z0-9_]", "", code).strip("_")
        if cleaned:
            candidate = _truncate_identifier(cleaned)
            if candidate not in used:
                used.add(candidate)
                return candidate
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:8]
    candidate = _truncate_identifier(f"{prefix}_{digest}")
    if candidate in used:
        suffix = 1
        while f"{candidate}_{suffix}" in used:
            suffix += 1
        candidate = _truncate_identifier(f"{candidate}_{suffix}")
    used.add(candidate)
    return candidate


def _safe_attribute_id(object_code: str, code: str, used: set[str]) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", code).strip("_")
    if not cleaned:
        cleaned = "attr"
    candidate = _truncate_identifier(cleaned)
    if candidate not in used:
        used.add(candidate)
        return candidate
    if object_code:
        prefixed = candidate if candidate.startswith(f"{object_code}_") else f"{object_code}_{candidate}"
        prefixed = _truncate_identifier(prefixed)
        if prefixed not in used:
            used.add(prefixed)
            return prefixed
    digest = hashlib.sha1(f"{object_code}:{code}".encode("utf-8")).hexdigest()[:6]
    fallback = _truncate_identifier(f"{candidate}_{digest}")
    if fallback in used:
        suffix = 1
        while f"{fallback}_{suffix}" in used:
            suffix += 1
        fallback = _truncate_identifier(f"{fallback}_{suffix}")
    used.add(fallback)
    return fallback


def _ai_extract_raw_meta(
    input_csv: Path,
    llm_config_path: Path | None = None,
    revision_notes: str | None = None,
) -> tuple[JsonDict, str | None]:
    rows = read_csv_rows(input_csv)
    if not rows:
        return {
            "source_csv": str(input_csv),
            "source_type": "unknown",
            "tables": [],
            "relations": [],
            "notes": ["空文件"],
            "ai_status": "fail",
        }, "AI1 数据识别失败: 空文件"

    preview_rows = rows[:200]
    preview = [{"row": idx + 1, "cells": row} for idx, row in enumerate(preview_rows)]
    prompt = _load_prompt("data_extract")
    instruction = (
        "请根据CSV原始内容识别表/对象/关系并输出JSON：\n"
        "{\n"
        '  "source_type": "fields|relations|objects",\n'
        '  "tables": [\n'
        '    {"table_name": "中文表名/对象名", "table_en": "english_name", "fields": [\n'
        '      {"field_name": "...", "field_en": "...", "data_type": "", "comment": "",\n'
        '       "position": {"row": 0, "col": 0}, "source_columns": {"field_name": ""}}\n'
        "    ]}\n"
        "  ],\n"
        '  "relations": [\n'
        '    {"subject": "对象名", "predicate": "includes|relatesTo|affects|constitutes",\n'
        '     "object": "对象名", "domain_attribute": "", "range_attribute": ""}\n'
        "  ],\n"
        '  "notes": ["可选"]\n'
        "}\n"
        "要求：尽量输出tables；fields无法识别可为空；position/source_columns无法确认可为空但需在notes说明；"
        "仅输出JSON。"
    )
    if revision_notes:
        instruction += f"\n修订要求：{revision_notes}"
    payload_text = json.dumps(
        {"preview": preview, "total_rows": len(rows)},
        ensure_ascii=False,
        indent=2,
    )
    prompt_text = "\n\n".join([prompt, instruction, "CSV预览：", payload_text]).strip()

    ai_error: str | None = None
    tables: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    notes: list[str] = []
    source_type = "unknown"
    try:
        llm_path = llm_config_path or Path("configs/llm.yaml")
        config = _load_llm_config(llm_path)
        llm = _build_llm(config)
        response = _invoke_llm(llm, prompt_text)
        payload = _extract_json_payload(response)
        if not isinstance(payload, dict):
            raise ValueError("AI1 输出不是JSON对象")
        source_type = str(payload.get("source_type") or "").strip() or source_type
        notes = [str(item) for item in _ensure_list(payload.get("notes")) if item]
        raw_tables = _ensure_list_of_dict(payload.get("tables"))
        raw_relations = _ensure_list_of_dict(payload.get("relations"))
        raw_fields = payload.get("fields") or payload.get("columns")
        raw_objects = payload.get("objects")
        for table in raw_tables:
            table_name = str(table.get("table_name") or "").strip()
            if not table_name:
                continue
            table_en = str(table.get("table_en") or "").strip()
            fields: list[dict[str, Any]] = []
            for field in _ensure_list_of_dict(table.get("fields")):
                field_name = str(field.get("field_name") or "").strip()
                field_en = str(field.get("field_en") or "").strip()
                if not field_name and not field_en:
                    continue
                fields.append(
                    {
                        "field_name": field_name or field_en,
                        "field_en": field_en,
                        "data_type": str(field.get("data_type") or ""),
                        "comment": str(field.get("comment") or ""),
                        "table_name": table_name,
                        "table_en": table_en,
                        "position": field.get("position") if isinstance(field.get("position"), dict) else {},
                        "source_columns": field.get("source_columns")
                        if isinstance(field.get("source_columns"), dict)
                        else {},
                    }
                )
            tables.append({"table_name": table_name, "table_en": table_en, "fields": fields})
        relations = raw_relations
        if not tables and isinstance(raw_objects, list):
            for obj in _ensure_list_of_dict(raw_objects):
                obj_name = str(obj.get("object_name") or obj.get("table_name") or obj.get("name") or "").strip()
                if not obj_name:
                    continue
                obj_en = str(obj.get("object_en") or obj.get("table_en") or "").strip()
                tables.append({"table_name": obj_name, "table_en": obj_en, "fields": []})
        if not tables and isinstance(raw_fields, list):
            table_name = str(payload.get("table_name") or payload.get("object_name") or "未命名").strip() or "未命名"
            table_en = str(payload.get("table_en") or payload.get("object_en") or "").strip()
            fields: list[dict[str, Any]] = []
            for field in raw_fields:
                if isinstance(field, dict):
                    field_name = str(field.get("field_name") or field.get("name") or "").strip()
                    field_en = str(field.get("field_en") or field.get("en") or "").strip()
                    if not field_name and not field_en:
                        continue
                    fields.append(
                        {
                            "field_name": field_name or field_en,
                            "field_en": field_en,
                            "data_type": str(field.get("data_type") or ""),
                            "comment": str(field.get("comment") or ""),
                            "table_name": table_name,
                            "table_en": table_en,
                            "position": field.get("position") if isinstance(field.get("position"), dict) else {},
                            "source_columns": field.get("source_columns")
                            if isinstance(field.get("source_columns"), dict)
                            else {},
                        }
                    )
                else:
                    field_text = str(field).strip()
                    if field_text:
                        fields.append(
                            {
                                "field_name": field_text,
                                "field_en": "",
                                "data_type": "",
                                "comment": "",
                                "table_name": table_name,
                                "table_en": table_en,
                                "position": {},
                                "source_columns": {},
                            }
                        )
            tables.append({"table_name": table_name, "table_en": table_en, "fields": fields})
        if not source_type:
            source_type = "relations" if relations else "fields"
    except Exception as exc:
        ai_error = f"AI1 数据识别失败: {exc}"

    if not tables and relations:
        object_names = {
            str(rel.get("subject") or "").strip()
            for rel in relations
            if isinstance(rel, dict)
        }
        object_names.update(
            {
                str(rel.get("object") or "").strip()
                for rel in relations
                if isinstance(rel, dict)
            }
        )
        for name in sorted({name for name in object_names if name}):
            tables.append({"table_name": name, "table_en": "", "fields": []})

    if not tables:
        ai_error = ai_error or "AI1 数据识别失败: 未返回表结构"

    raw_meta = {
        "source_csv": str(input_csv),
        "source_type": source_type,
        "tables": tables,
        "relations": relations,
        "notes": notes,
    }
    raw_meta["ai_status"] = "ok" if not ai_error else "fail"
    if ai_error:
        raw_meta["ai_error"] = ai_error
    return raw_meta, ai_error


def _build_objects_from_raw_meta(raw_meta: JsonDict) -> tuple[list[OntologyObject], list[OntologyAttribute]]:
    used_ids: set[str] = set()
    used_attr_ids: set[str] = set()
    objects: list[OntologyObject] = []
    attributes: list[OntologyAttribute] = []
    for table in _ensure_list_of_dict(raw_meta.get("tables")):
        table_name = str(table.get("table_name") or "").strip() or "未命名"
        table_en = str(table.get("table_en") or "").strip()
        obj_id = _safe_object_id(table_name, table_en or table_name, used_ids)
        obj = OntologyObject(
            label=table_name,
            code=obj_id,
            en_label=table_en or None,
            source_table=table_name,
        )
        objects.append(obj)
        for field in _ensure_list_of_dict(table.get("fields")):
            field_name = _normalize_label_repeat(str(field.get("field_name") or "").strip())
            field_en = str(field.get("field_en") or "").strip()
            attr_label = _normalize_label_repeat(field_name or field_en)
            if attr_label and attr_label == obj.label and field_en:
                attr_label = f"{obj.label}标识"
            if not attr_label:
                continue
            attr_code = _safe_attribute_id(obj.code, field_en or field_name, used_attr_ids)
            attributes.append(
                OntologyAttribute(
                    label=attr_label,
                    code=attr_code,
                    object_id=obj.code,
                    raw_chinese=field_name,
                    raw_english=field_en,
                    source_table=table_name,
                    en_label=field_en or None,
                    data_type=str(field.get("data_type") or "") or None,
                    comment=str(field.get("comment") or "") or None,
                )
            )
    return objects, attributes


def _build_object_map(objects: list[OntologyObject]) -> dict[str, OntologyObject]:
    mapping: dict[str, OntologyObject] = {}
    for obj in objects:
        mapping[_normalize_key(obj.label)] = obj
        if obj.en_label:
            mapping[_normalize_key(obj.en_label)] = obj
    return mapping


def _build_attribute_map(attributes: list[OntologyAttribute]) -> dict[str, OntologyAttribute]:
    mapping: dict[str, OntologyAttribute] = {}
    for attr in attributes:
        for key in {attr.label, attr.raw_chinese, attr.raw_english, attr.en_label}:
            if key:
                mapping[_normalize_key(key)] = attr
    return mapping


def _map_relation_predicate(value: str) -> str | None:
    if not value:
        return None
    key = value.strip()
    lower = key.lower()
    if lower in RELATION_CODE_TO_ID:
        return RELATION_CODE_TO_ID[lower]
    if key in RELATION_LABEL_TO_ID:
        return RELATION_LABEL_TO_ID[key]
    return None


def _apply_ai_object_renames(
    ai_objects: list[JsonDict], objects: list[OntologyObject]
) -> tuple[list[OntologyObject], list[JsonDict]]:
    issues: list[JsonDict] = []
    table_map: dict[str, OntologyObject] = {}
    used_ids = {obj.code for obj in objects}
    for obj in objects:
        if obj.source_table:
            table_map[_normalize_key(obj.source_table)] = obj
        if obj.en_label:
            table_map[_normalize_key(obj.en_label)] = obj
        table_map[_normalize_key(obj.label)] = obj
    updates: dict[str, OntologyObject] = {}
    new_objects: list[OntologyObject] = []
    for item in ai_objects:
        table_name = str(item.get("table_name") or "").strip()
        obj_name = str(item.get("object_name") or "").strip()
        obj_en = str(item.get("object_en") or "").strip()
        if not obj_name:
            issues.append({"category": "object", "message": "对象缺少object_name", "severity": "high"})
            continue
        target = table_map.get(_normalize_key(table_name)) if table_name else None
        if target:
            updates[target.code] = OntologyObject(
                label=obj_name or target.label,
                code=target.code,
                en_label=obj_en or target.en_label,
                source_table=target.source_table,
            )
            continue
        if _normalize_key(obj_name) in table_map or (obj_en and _normalize_key(obj_en) in table_map):
            continue
        new_id = _safe_object_id(obj_name, obj_en or obj_name, used_ids)
        new_objects.append(
            OntologyObject(
                label=obj_name,
                code=new_id,
                en_label=obj_en or None,
                source_table=table_name or None,
            )
        )
        issues.append({"category": "object", "message": f"AI2 新增对象: {obj_name}", "severity": "low"})
    if updates:
        objects = [updates.get(obj.code, obj) for obj in objects]
    if new_objects:
        objects = objects + new_objects
    return objects, issues


def _extend_objects_from_names(
    objects: list[OntologyObject], names: Iterable[str]
) -> tuple[list[OntologyObject], list[JsonDict]]:
    issues: list[JsonDict] = []
    used_ids = {obj.code for obj in objects}
    obj_map = _build_object_map(objects)
    new_objects: list[OntologyObject] = []
    for name in names:
        label = str(name).strip()
        if not label:
            continue
        if _normalize_key(label) in obj_map:
            continue
        new_id = _safe_object_id(label, label, used_ids)
        new_objects.append(
            OntologyObject(
                label=label,
                code=new_id,
                en_label=None,
                source_table=None,
            )
        )
        issues.append({"category": "object", "message": f"关系补齐对象: {label}", "severity": "low"})
    if new_objects:
        objects = objects + new_objects
    return objects, issues


def _extend_attributes_from_relations(
    attributes: list[OntologyAttribute],
    objects: list[OntologyObject],
    ai_relations: list[JsonDict],
) -> tuple[list[OntologyAttribute], list[JsonDict]]:
    issues: list[JsonDict] = []
    used_ids = {attr.code for attr in attributes}
    attr_map = _build_attribute_map(attributes)
    obj_map = _build_object_map(objects)
    new_attrs: list[OntologyAttribute] = []
    for rel in ai_relations:
        subject_label = str(rel.get("subject") or "").strip()
        object_label = str(rel.get("object") or "").strip()
        subject = obj_map.get(_normalize_key(subject_label))
        target = obj_map.get(_normalize_key(object_label))
        domain_attr_name = str(rel.get("domain_attribute") or "").strip()
        range_attr_name = str(rel.get("range_attribute") or "").strip()
        if subject and domain_attr_name and _normalize_key(domain_attr_name) not in attr_map:
            attr_code = _safe_attribute_id(subject.code, domain_attr_name, used_ids)
            new_attrs.append(
                OntologyAttribute(
                    label=domain_attr_name,
                    code=attr_code,
                    object_id=subject.code,
                    raw_chinese=domain_attr_name,
                    raw_english="",
                    source_table=subject.source_table,
                )
            )
            attr_map[_normalize_key(domain_attr_name)] = new_attrs[-1]
            issues.append({"category": "attribute", "message": f"关系补齐属性: {domain_attr_name}", "severity": "low"})
        if target and range_attr_name and _normalize_key(range_attr_name) not in attr_map:
            attr_code = _safe_attribute_id(target.code, range_attr_name, used_ids)
            new_attrs.append(
                OntologyAttribute(
                    label=range_attr_name,
                    code=attr_code,
                    object_id=target.code,
                    raw_chinese=range_attr_name,
                    raw_english="",
                    source_table=target.source_table,
                )
            )
            attr_map[_normalize_key(range_attr_name)] = new_attrs[-1]
            issues.append({"category": "attribute", "message": f"关系补齐属性: {range_attr_name}", "severity": "low"})
    if new_attrs:
        attributes = attributes + new_attrs
    return attributes, issues


def _extend_attributes_from_scenarios(
    attributes: list[OntologyAttribute],
    objects: list[OntologyObject],
    ai_scenarios: list[JsonDict],
) -> tuple[list[OntologyAttribute], list[JsonDict]]:
    issues: list[JsonDict] = []
    used_ids = {attr.code for attr in attributes}
    attr_map = _build_attribute_map(attributes)
    obj_map = _build_object_map(objects)
    new_attrs: list[OntologyAttribute] = []
    for scenario in ai_scenarios:
        object_names = _coerce_str_list(scenario.get("objects"))
        if not object_names:
            continue
        first_obj = obj_map.get(_normalize_key(object_names[0]))
        if not first_obj:
            continue
        for key_attr in _coerce_str_list(scenario.get("key_attributes")):
            if _normalize_key(key_attr) in attr_map:
                continue
            attr_code = _safe_attribute_id(first_obj.code, key_attr, used_ids)
            new_attrs.append(
                OntologyAttribute(
                    label=key_attr,
                    code=attr_code,
                    object_id=first_obj.code,
                    raw_chinese=key_attr,
                    raw_english="",
                    source_table=first_obj.source_table,
                )
            )
            attr_map[_normalize_key(key_attr)] = new_attrs[-1]
            issues.append({"category": "attribute", "message": f"场景补齐属性: {key_attr}", "severity": "low"})
    if new_attrs:
        attributes = attributes + new_attrs
    return attributes, issues


def _map_relations(
    ai_relations: list[JsonDict],
    objects: list[OntologyObject],
    attributes: list[OntologyAttribute],
) -> tuple[list[RelationEdge], list[JsonDict]]:
    issues: list[JsonDict] = []
    obj_map = _build_object_map(objects)
    attr_map = _build_attribute_map(attributes)
    edges: list[RelationEdge] = []
    for rel in ai_relations:
        subject_label = str(rel.get("subject") or "").strip()
        object_label = str(rel.get("object") or "").strip()
        predicate_raw = str(rel.get("predicate") or rel.get("relation") or "").strip()
        predicate = _map_relation_predicate(predicate_raw)
        if not subject_label or not object_label or not predicate:
            issues.append(
                {
                    "category": "relation",
                    "message": f"关系缺少必要字段: {subject_label}/{predicate_raw}/{object_label}",
                    "severity": "high",
                }
            )
            continue
        subject = obj_map.get(_normalize_key(subject_label))
        target = obj_map.get(_normalize_key(object_label))
        if not subject or not target:
            issues.append(
                {
                    "category": "relation",
                    "message": f"关系对象未识别: {subject_label} -> {object_label}",
                    "severity": "high",
                }
            )
            continue
        domain_attr_name = str(rel.get("domain_attribute") or "").strip()
        range_attr_name = str(rel.get("range_attribute") or "").strip()
        domain_attr = (
            attr_map.get(_normalize_key(domain_attr_name)).code
            if domain_attr_name and _normalize_key(domain_attr_name) in attr_map
            else None
        )
        range_attr = (
            attr_map.get(_normalize_key(range_attr_name)).code
            if range_attr_name and _normalize_key(range_attr_name) in attr_map
            else None
        )
        if domain_attr_name and not domain_attr:
            issues.append(
                {
                    "category": "relation",
                    "message": f"domain_attribute未匹配: {domain_attr_name}",
                    "severity": "medium",
                }
            )
        if range_attr_name and not range_attr:
            issues.append(
                {
                    "category": "relation",
                    "message": f"range_attribute未匹配: {range_attr_name}",
                    "severity": "medium",
                }
            )
        edges.append(
            RelationEdge(
                subject=subject.code,
                predicate=predicate,
                object=target.code,
                predicate_label=RELATION_DEFS[predicate]["label"],
                domain_attribute=domain_attr,
                range_attribute=range_attr,
            )
        )
    return edges, issues


def _map_scenarios(
    ai_scenarios: list[JsonDict],
    objects: list[OntologyObject],
    attributes: list[OntologyAttribute],
    edges: list[RelationEdge],
) -> tuple[list[JsonDict], list[JsonDict]]:
    issues: list[JsonDict] = []
    obj_map = _build_object_map(objects)
    attr_map = _build_attribute_map(attributes)
    edge_map = {(edge.subject, edge.object): edge for edge in edges}
    scenarios: list[JsonDict] = []
    for idx, item in enumerate(ai_scenarios, start=1):
        title = str(item.get("title") or "").strip()
        question = str(item.get("question") or "").strip()
        object_names = _coerce_str_list(item.get("objects"))
        key_attrs_raw = _coerce_str_list(item.get("key_attributes"))
        object_ids: list[str] = []
        for name in object_names:
            obj = obj_map.get(_normalize_key(name))
            if obj:
                object_ids.append(obj.code)
        if not title:
            issues.append({"category": "scenario", "message": f"场景{idx}缺少标题", "severity": "high"})
        if not question:
            issues.append({"category": "scenario", "message": f"场景{idx}缺少问题", "severity": "high"})
        if not object_ids:
            issues.append({"category": "scenario", "message": f"场景{idx}对象为空", "severity": "high"})
        attr_ids: list[str] = []
        for key in key_attrs_raw:
            attr = attr_map.get(_normalize_key(key))
            if attr:
                attr_ids.append(attr.code)
        if not attr_ids:
            issues.append({"category": "scenario", "message": f"场景{idx}关键属性为空", "severity": "high"})
        scenario_edges: list[JsonDict] = []
        for pos in range(len(object_ids) - 1):
            edge = edge_map.get((object_ids[pos], object_ids[pos + 1]))
            if edge:
                scenario_edges.append(
                    {
                        "subject": edge.subject,
                        "predicate": edge.predicate,
                        "predicate_label": edge.predicate_label,
                        "object": edge.object,
                    }
                )
        scenarios.append(
            {
                "id": f"scenario_{idx}",
                "title": title,
                "question": question,
                "objects": [obj.label for obj in objects if obj.code in object_ids],
                "object_ids": object_ids,
                "edges": scenario_edges,
                "attributes": attr_ids,
            }
        )
    return scenarios, issues


def _normalize_parameters(raw_params: object) -> list[JsonDict]:
    params: list[JsonDict] = []
    for item in _ensure_list_of_dict(raw_params):
        name = str(item.get("name") or item.get("param_name") or item.get("param") or "").strip()
        if not name:
            continue
        param_type = str(item.get("type") or item.get("param_type") or "xsd:string").strip() or "xsd:string"
        optional_raw = item.get("optional")
        if isinstance(optional_raw, str):
            optional = optional_raw.strip().lower() in {"true", "1", "yes", "y"}
        else:
            optional = bool(optional_raw) if optional_raw is not None else False
        default_value = item.get("default") or item.get("default_value")
        type_node: str | None = None
        param_type_lower = param_type.lower()
        if param_type_lower in {"list", "array"}:
            param_type = "list"
            type_node = "ListType"
        elif param_type_lower in {"dict", "map", "object"}:
            param_type = "dict"
            type_node = "DictType"
        normalized: JsonDict = {
            "name": name,
            "type": param_type,
            "optional": optional,
        }
        if default_value not in (None, ""):
            normalized["default"] = str(default_value)
        if type_node:
            normalized["type_node"] = type_node
        params.append(normalized)
    return params


def _map_logics(ai_logics: list[JsonDict]) -> tuple[list[JsonDict], list[JsonDict]]:
    issues: list[JsonDict] = []
    logics: list[JsonDict] = []
    used: set[str] = set()
    for idx, item in enumerate(ai_logics, start=1):
        logic_id_raw = str(item.get("logic_id") or item.get("id") or "").strip()
        label = str(item.get("label") or item.get("name") or "").strip()
        logic_id = _safe_generic_id(label or logic_id_raw or f"logic_{idx}", logic_id_raw, used, "logic")
        if not label:
            issues.append({"category": "logic", "message": f"逻辑{idx}缺少名称", "severity": "medium"})
        logics.append(
            {
                "id": logic_id,
                "label": label or logic_id,
                "comment": str(item.get("comment") or "").strip(),
                "parameters": _normalize_parameters(item.get("parameters")),
            }
        )
    return logics, issues


def _map_actions(ai_actions: list[JsonDict]) -> tuple[list[JsonDict], list[JsonDict]]:
    issues: list[JsonDict] = []
    actions: list[JsonDict] = []
    used: set[str] = set()
    for idx, item in enumerate(ai_actions, start=1):
        action_id_raw = str(item.get("action_id") or item.get("id") or "").strip()
        label = str(item.get("label") or item.get("name") or "").strip()
        action_id = _safe_generic_id(label or action_id_raw or f"action_{idx}", action_id_raw, used, "action")
        if not label:
            issues.append({"category": "action", "message": f"动作{idx}缺少名称", "severity": "medium"})
        actions.append(
            {
                "id": action_id,
                "label": label or action_id,
                "comment": str(item.get("comment") or "").strip(),
                "parameters": _normalize_parameters(item.get("parameters")),
            }
        )
    return actions, issues


def _dedupe_edges(edges: Iterable[RelationEdge]) -> list[RelationEdge]:
    seen: set[tuple[str, str, str, str | None, str | None]] = set()
    unique: list[RelationEdge] = []
    for edge in edges:
        key = (
            edge.subject,
            edge.predicate,
            edge.object,
            edge.domain_attribute,
            edge.range_attribute,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(edge)
    return unique


def _validate_design(
    objects: list[OntologyObject],
    attributes: list[OntologyAttribute],
    edges: list[RelationEdge],
    scenarios: list[JsonDict],
) -> list[JsonDict]:
    issues: list[JsonDict] = []
    object_ids = {obj.code for obj in objects}
    if len(scenarios) < 3:
        issues.append(
            {
                "category": "scenario",
                "message": "场景数量不足，至少需要3个场景",
                "severity": "high",
            }
        )
    covered: set[str] = set()
    for scenario in scenarios:
        covered.update(_coerce_str_list(scenario.get("object_ids")))
        if not _coerce_str_list(scenario.get("attributes")):
            issues.append(
                {
                    "category": "scenario",
                    "message": f"场景 {scenario.get('title')} 关键属性为空",
                    "severity": "high",
                }
            )
    if object_ids:
        coverage = len(covered) / len(object_ids)
        if coverage < 0.8:
            issues.append(
                {
                    "category": "coverage",
                    "message": f"场景覆盖率不足：{coverage:.0%}",
                    "severity": "high",
                }
            )
    for edge in edges:
        if edge.subject not in object_ids or edge.object not in object_ids:
            issues.append(
                {
                    "category": "relation",
                    "message": "关系存在未知对象",
                    "severity": "high",
                }
            )
        if edge.predicate not in RELATION_DEFS:
            issues.append(
                {
                    "category": "relation",
                    "message": f"未知关系类型: {edge.predicate}",
                    "severity": "high",
                }
            )
    if not attributes:
        issues.append(
            {
                "category": "mapping",
                "message": "未识别任何属性，可能影响关键属性映射",
                "severity": "medium",
            }
        )
    return issues


def build_design_data(
    source_csv: Path,
    objects: list[OntologyObject],
    edges: list[RelationEdge],
    attributes: list[OntologyAttribute],
    scenarios: list[JsonDict],
    logics: list[JsonDict],
    actions: list[JsonDict],
) -> JsonDict:
    return {
        "ontology_name": "客户经营本体",
        "source_csv": str(source_csv),
        "objects": [
            {
                "id": obj.code,
                "label": obj.label,
                "en_label": obj.en_label,
                "source_table": obj.source_table,
            }
            for obj in sorted(objects, key=lambda o: o.label)
        ],
        "relations": [{"id": rid, **RELATION_DEFS[rid]} for rid in RELATION_DEFS],
        "edges": [
            {
                "subject": edge.subject,
                "predicate": edge.predicate,
                "predicate_label": edge.predicate_label,
                "object": edge.object,
                "domain_attribute": edge.domain_attribute,
                "range_attribute": edge.range_attribute,
            }
            for edge in edges
        ],
        "attributes": [
            {
                "id": attr.code,
                "label": attr.label,
                "en_label": attr.en_label,
                "object_id": attr.object_id,
                "raw_chinese": attr.raw_chinese,
                "raw_english": attr.raw_english,
                "source_table": attr.source_table,
                "data_type": attr.data_type,
                "comment": attr.comment,
            }
            for attr in attributes
        ],
        "scenarios": scenarios,
        "logics": logics,
        "actions": actions,
        "notes": [
            "输入可能仅包含字段名称，表与字段映射允许后续补充。",
            "对象默认对应单表，避免跨表绑定属性。",
        ],
    }


def _build_design_prompt(raw_meta: JsonDict, feedback: str | None = None) -> str:
    prompt = _load_prompt("design")
    instruction = (
        "请根据表与字段摘要输出JSON（严格遵循字段名，不得省略scenarios）：\n"
        "{\n"
        '  "objects": [\n'
        '    {"table_name": "源表中文名", "object_name": "对象中文名", "object_en": "object_en"}\n'
        "  ],\n"
        '  "relations": [\n'
        '    {"subject": "对象中文名", "predicate": "includes|relatesTo|affects|constitutes", '
        '"object": "对象中文名", "domain_attribute": "属性名可选", "range_attribute": "属性名可选"}\n'
        "  ],\n"
        '  "logics": [\n'
        '    {"logic_id": "evaluate_x", "label": "逻辑名", "comment": "说明", "parameters": [\n'
        '      {"name": "param", "type": "xsd:string", "optional": false}\n'
        "    ]}\n"
        "  ],\n"
        '  "actions": [\n'
        '    {"action_id": "update_x", "label": "动作名", "comment": "说明", "parameters": [\n'
        '      {"name": "param", "type": "xsd:string", "optional": false}\n'
        "    ]}\n"
        "  ],\n"
        '  "scenarios": [\n'
        '    {"title": "场景标题", "question": "问题", "objects": ["对象1","对象2"], '
        '"key_attributes": ["属性1","属性2"]}\n'
        "  ]\n"
        "}\n"
        "要求：对象一表一对象；场景覆盖>=80%对象；输出至少3个场景；"
        "场景标题由你自行推断；关系尽量避免形成双向includes或循环；"
        "logics/actions 不需要可输出空数组；只输出JSON。"
    )
    if feedback:
        instruction += f"\n修订意见：{feedback}"
    summary = build_table_summary(raw_meta)
    return "\n\n".join([prompt, instruction, "表与字段摘要：", summary]).strip()


def _ai_design(
    raw_meta: JsonDict,
    objects: list[OntologyObject],
    attributes: list[OntologyAttribute],
    llm_config_path: Path | None = None,
    feedback: str | None = None,
) -> tuple[
    list[OntologyObject],
    list[RelationEdge],
    list[JsonDict],
    list[JsonDict],
    list[JsonDict],
    list[JsonDict],
    str | None,
]:
    def _call_design(extra_feedback: str | None) -> tuple[JsonDict, str | None]:
        prompt_text = _build_design_prompt(raw_meta, extra_feedback)
        try:
            llm_path = llm_config_path or Path("configs/llm.yaml")
            config = _load_llm_config(llm_path)
            llm = _build_llm(config)
            response = _invoke_llm(llm, prompt_text)
            payload = _extract_json_payload(response)
            if not isinstance(payload, dict):
                raise ValueError("AI2 输出不是JSON对象")
            return payload, None
        except Exception as exc:
            return {}, str(exc)

    payload, err = _call_design(feedback)
    if err:
        return (
            objects,
            [],
            [],
            [],
            [],
            [{"category": "design", "message": "AI2 设计失败", "severity": "high"}],
            err,
        )

    def _needs_scenario_fix(payload_data: JsonDict) -> bool:
        scenarios_data = payload_data.get("scenarios")
        if not isinstance(scenarios_data, list) or len(scenarios_data) < 3:
            return True
        for item in scenarios_data:
            if not isinstance(item, dict):
                return True
            if not _coerce_str_list(item.get("objects")):
                return True
            if not _coerce_str_list(item.get("key_attributes")):
                return True
        return False

    if _needs_scenario_fix(payload):
        retry_feedback = (feedback + "；" if feedback else "") + (
            "必须输出>=3个scenarios，且每个场景必须包含objects与key_attributes，"
            "key_attributes可以是业务指标或主键字段名。"
        )
        retry_payload, retry_err = _call_design(retry_feedback)
        if not retry_err and not _needs_scenario_fix(retry_payload):
            payload = retry_payload

    ai_objects = payload.get("objects")
    ai_relations = payload.get("relations")
    ai_scenarios = payload.get("scenarios")
    ai_logics = payload.get("logics")
    ai_actions = payload.get("actions")

    issues: list[JsonDict] = []
    if not isinstance(ai_objects, list):
        issues.append({"category": "object", "message": "AI2 未返回对象清单", "severity": "high"})
        ai_objects = []
    if not isinstance(ai_relations, list):
        issues.append({"category": "relation", "message": "AI2 未返回关系清单", "severity": "high"})
        ai_relations = []
    if not isinstance(ai_scenarios, list):
        issues.append({"category": "scenario", "message": "AI2 未返回场景清单", "severity": "high"})
        ai_scenarios = []
    if ai_logics is None:
        ai_logics = []
    if ai_actions is None:
        ai_actions = []

    objects, object_issues = _apply_ai_object_renames(_ensure_list_of_dict(ai_objects), objects)
    issues.extend(object_issues)
    relation_names: list[str] = []
    for rel in _ensure_list_of_dict(ai_relations):
        relation_names.append(str(rel.get("subject") or ""))
        relation_names.append(str(rel.get("object") or ""))
    for scenario in _ensure_list_of_dict(ai_scenarios):
        relation_names.extend(_coerce_str_list(scenario.get("objects")))
    objects, extend_issues = _extend_objects_from_names(objects, relation_names)
    issues.extend(extend_issues)
    attributes, relation_attr_issues = _extend_attributes_from_relations(
        attributes,
        objects,
        _ensure_list_of_dict(ai_relations),
    )
    issues.extend(relation_attr_issues)
    edges, relation_issues = _map_relations(_ensure_list_of_dict(ai_relations), objects, attributes)
    issues.extend(relation_issues)
    attributes, scenario_attr_issues = _extend_attributes_from_scenarios(
        attributes,
        objects,
        _ensure_list_of_dict(ai_scenarios),
    )
    issues.extend(scenario_attr_issues)
    scenarios, scenario_issues = _map_scenarios(_ensure_list_of_dict(ai_scenarios), objects, attributes, edges)
    issues.extend(scenario_issues)
    logics, logic_issues = _map_logics(_ensure_list_of_dict(ai_logics))
    issues.extend(logic_issues)
    actions, action_issues = _map_actions(_ensure_list_of_dict(ai_actions))
    issues.extend(action_issues)
    return objects, edges, scenarios, logics, actions, issues, None


def _format_mermaid_flow(scenario: JsonDict, id_to_label: dict[str, str]) -> str:
    lines = ["flowchart LR"]
    edges = _ensure_list_of_dict(scenario.get("edges"))
    object_ids = _coerce_str_list(scenario.get("object_ids"))
    if not edges:
        for node_id in object_ids[:6]:
            label = id_to_label.get(node_id, node_id)
            lines.append(f"    {node_id}[\"{label}\"]")
        return "\n".join(lines)
    for edge in edges:
        subject = str(edge.get("subject") or "").strip()
        target = str(edge.get("object") or "").strip()
        predicate_label = str(edge.get("predicate_label") or "").strip()
        if not subject or not target:
            continue
        source_label = id_to_label.get(subject, subject)
        target_label = id_to_label.get(target, target)
        lines.append(
            f"    {subject}[\"{source_label}\"] -- {predicate_label} --> {target}[\"{target_label}\"]"
        )
    return "\n".join(lines)


def write_design_doc(path: Path, data: JsonDict) -> None:
    objects = _ensure_list_of_dict(data.get("objects"))
    id_to_label = {
        str(obj.get("id")): str(obj.get("label"))
        for obj in objects
        if obj.get("id") and obj.get("label")
    }
    attributes = _ensure_list_of_dict(data.get("attributes"))
    attr_id_to_label = {
        str(attr.get("id")): str(attr.get("label"))
        for attr in attributes
        if attr.get("id") and attr.get("label")
    }
    scenarios = _ensure_list_of_dict(data.get("scenarios"))
    total_objects = len(objects)
    attribute_count = len(attributes)
    covered: set[str] = set()
    for scenario in scenarios:
        covered.update(_coerce_str_list(scenario.get("object_ids")))
    coverage = 0 if total_objects == 0 else int(len(covered) / total_objects * 100)
    relation_types = "、".join({str(rel.get("label")) for rel in _ensure_list_of_dict(data.get("relations"))}) or "无"

    sections = [
        "# 本体智能体设计文档",
        "",
        "文档类型｜设计说明",
        "目标读者｜本体建模与数据治理团队",
        "核心问题｜如何在仅有字段名时构建可用本体并支持业务提问、如何保障关系流转健康、如何沉淀可扩展映射",
        "读完能做｜运行智能体生成本体与OWL、理解三类业务问题流转、补齐表/列映射",
        "",
        "## 结论",
        "- 输入即便只有字段名，也能先生成对象清单、关系推断与三类业务问题流程，覆盖核心业务链路。",
        "- 本体结构以‘对象=表’为约束，不跨表绑定属性，映射表允许后续增补。",
        "- 三个问题场景覆盖超过 80% 的对象，流程以包含/构成/影响为主线。",
        "",
        "## 输入与约束",
        f"- 数据源：{data.get('source_csv', '')}",
        "- 最坏情况：仅字段名/中英名称，无表名、类型、注释。",
        "- 结构约束：一个对象对应一张表；关系体现业务流程，避免过度交叉。",
        "",
        "## 本体结构概要",
        f"- 对象数量：{total_objects}",
        f"- 关系数量：{len(_ensure_list_of_dict(data.get('edges')))}",
        f"- 属性数量：{attribute_count}",
        f"- 关系类型：{relation_types}",
        "",
        "## 关系命名建议",
        "- 使用指向性命名：如“关联/被关联”“引用/被引用”，避免语义方向不清。",
        "- 同一域类内保持关系名称一致，减少同义关系混用。",
        "- 关系命名尽量反映业务动作或归属关系，避免泛化词。",
        "",
        "## 三类业务问题与流程",
    ]

    for scenario in scenarios:
        key_attrs = _coerce_str_list(scenario.get("attributes"))
        display_attrs = [attr_id_to_label.get(attr, attr) for attr in key_attrs]
        key_attr_text = "、".join(display_attrs) if display_attrs else "待补充"
        flow = _format_mermaid_flow(scenario, id_to_label)
        sections.extend(
            [
                f"### {scenario.get('title', '')}",
                f"- 问题：{scenario.get('question', '')}",
                f"- 覆盖对象数：{len(_coerce_str_list(scenario.get('object_ids')))}",
                f"- 关键属性：{key_attr_text}",
                "- 流程图：",
                "```mermaid",
                flow,
                "```",
            ]
        )

    output_paths = data.get("output_paths") or {}
    sections.extend(
        [
            "## 覆盖率",
            f"- 覆盖对象：{len(covered)}/{total_objects}（{coverage}%）",
            "",
            "## 输出清单",
            f"- 设计文档：{output_paths.get('design_doc', 'output/ontology_builder_design.md')}",
            f"- OWL：{output_paths.get('owl', 'output/ontology.owl')}",
            "",
            "## 后续行动",
            "- What：补充来源表/列；Who：数据治理；When：下一次元数据刷新时。",
            "- What：完善指标口径与单位；Who：业务与数据产品；When：场景上线前。",
        ]
    )

    path.write_text("\n".join(sections) + "\n", encoding="utf-8")


def _issues_to_revision_notes(issues: list[JsonDict]) -> str:
    notes: list[str] = []
    for issue in issues:
        message = str(issue.get("message") or "")
        fix = str(issue.get("fix") or "")
        if fix:
            notes.append(f"{message}｜建议：{fix}")
        else:
            notes.append(message)
    return "；".join(note for note in notes if note)


def build_all(
    input_csv: Path,
    output_dir: Path,
    llm_config_path: Path | None = None,
    max_rounds: int = 3,
    template_path: Path | None = None,
) -> JsonDict:
    output_dir.mkdir(parents=True, exist_ok=True)
    template_path = template_path or DEFAULT_TEMPLATE_PATH
    design_doc_path = output_dir / "ontology_builder_design.md"
    owl_path = output_dir / "ontology.owl"
    revision_notes: str | None = None
    route = "data"

    for round_idx in range(1, max_rounds + 1):
        raw_meta, ai1_error = _ai_extract_raw_meta(
            input_csv,
            llm_config_path=llm_config_path,
            revision_notes=revision_notes if route == "data" else None,
        )
        if ai1_error:
            if round_idx == max_rounds:
                raise RuntimeError(ai1_error)
            revision_notes = ai1_error
            route = "data"
            continue

        objects, attributes = _build_objects_from_raw_meta(raw_meta)
        base_edges, base_issues = _map_relations(
            _ensure_list_of_dict(raw_meta.get("relations")),
            objects,
            attributes,
        )

        objects, ai2_edges, scenarios, logics, actions, ai2_issues, ai2_error = _ai_design(
            raw_meta,
            objects,
            attributes,
            llm_config_path=llm_config_path,
            feedback=revision_notes if route == "design" else None,
        )
        if ai2_error:
            if round_idx == max_rounds:
                raise RuntimeError(ai2_error)
            revision_notes = ai2_error
            route = "design"
            continue

        edges = _dedupe_edges(base_edges + ai2_edges)
        issues = base_issues + ai2_issues
        issues.extend(_validate_design(objects, attributes, edges, scenarios))

        design_data = build_design_data(
            input_csv,
            objects,
            edges,
            attributes,
            scenarios,
            logics,
            actions,
        )
        design_data["ai_status"] = "ok" if not issues else "fail"
        if issues:
            design_data["ai_issues"] = issues
        design_data["output_paths"] = {
            "design_doc": str(design_doc_path),
            "owl": str(owl_path),
        }

        write_design_doc(design_doc_path, design_data)

        design_audit, audit_error = ai_audit_design(
            design_data,
            issues,
            llm_config_path=llm_config_path,
            load_llm_config=_load_llm_config,
            build_llm=_build_llm,
            invoke_llm=_invoke_llm,
            extract_json_payload=_extract_json_payload,
            load_prompt=_load_prompt,
        )
        if audit_error:
            design_audit["ai_error"] = audit_error

        if design_audit.get("status") != "pass":
            issues = _ensure_list_of_dict(design_audit.get("issues"))
            revision_notes = _issues_to_revision_notes(issues)
            route = "design"
            if round_idx == max_rounds:
                raise RuntimeError("AI3 设计审计未通过")
            continue

        render_owl(template_path, design_data, owl_path)
        owl_text = owl_path.read_text(encoding="utf-8")
        owl_audit, owl_error = ai_audit_owl(
            owl_text,
            design_data,
            llm_config_path=llm_config_path,
            load_llm_config=_load_llm_config,
            build_llm=_build_llm,
            invoke_llm=_invoke_llm,
            extract_json_payload=_extract_json_payload,
            load_prompt=_load_prompt,
        )
        if owl_error:
            owl_audit["ai_error"] = owl_error

        if owl_audit.get("status") == "pass":
            break
        issues = _ensure_list_of_dict(owl_audit.get("issues"))
        revision_notes = _issues_to_revision_notes(issues)
        route = "design"
        if round_idx == max_rounds:
            raise RuntimeError("AI5 OWL 抽检未通过")

    return {
        "objects": len(design_data.get("objects", [])) if design_data else 0,
        "edges": len(design_data.get("edges", [])) if design_data else 0,
        "design_doc": str(design_doc_path),
        "owl": str(owl_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ontology builder agent")
    parser.add_argument("--input", default="input/本体对象关系列表.csv", help="输入CSV路径")
    parser.add_argument("--output", default="output", help="输出目录")
    parser.add_argument("--max-rounds", type=int, default=3, help="圆桌最大迭代轮数")
    parser.add_argument("--llm-config", default="configs/llm.yaml", help="LLM配置路径")
    return parser.parse_args()


@dataclass(frozen=True)
class ParsedArgs:
    input: str
    output: str
    max_rounds: int
    llm_config: str


def main() -> None:
    args = ParsedArgs(**vars(parse_args()))
    result = build_all(
        input_csv=Path(args.input),
        output_dir=Path(args.output),
        llm_config_path=Path(args.llm_config),
        max_rounds=args.max_rounds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
