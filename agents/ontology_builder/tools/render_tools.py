from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from agents.ontology_builder.tools.relation_defs import RELATION_DEFS

JsonDict = dict[str, Any]


def _ensure_list_of_dict(value: object) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _truncate_identifier(value: str, max_len: int = 56) -> str:
    if len(value) <= max_len:
        return value
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:6]
    keep = max_len - len(digest) - 1
    return f"{value[:keep]}_{digest}"


def _safe_relation_id(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", name).strip("_")
    if not cleaned:
        cleaned = "relation"
    return _truncate_identifier(cleaned)


def _ensure_unique_relation_id(candidate: str, used: set[str]) -> str:
    if candidate not in used:
        used.add(candidate)
        return candidate
    suffix = 2
    base = candidate
    while f"{base}_{suffix}" in used:
        suffix += 1
    unique = _truncate_identifier(f"{base}_{suffix}")
    used.add(unique)
    return unique


def _infer_xsd_type(attr: JsonDict) -> str:
    data_type = str(attr.get("data_type") or "").lower()
    if any(token in data_type for token in ["bool", "boolean", "bit"]):
        return "xsd:boolean"
    if any(token in data_type for token in ["date", "time", "timestamp"]):
        return "xsd:dateTime"
    if any(token in data_type for token in ["int", "bigint", "smallint", "tinyint"]):
        return "xsd:integer"
    if any(token in data_type for token in ["decimal", "numeric", "double", "float", "real", "money"]):
        return "xsd:decimal"
    return "xsd:string"


def _ttl_literal(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _render_parameters(params: list[JsonDict]) -> list[str]:
    blocks: list[str] = []
    for param in params:
        name = str(param.get("name") or "").strip()
        if not name:
            continue
        param_type = str(param.get("type") or "xsd:string").strip() or "xsd:string"
        optional = bool(param.get("optional"))
        parts = [
            "a ex:Parameter",
            f"ex:paramName {_ttl_literal(name)}",
            f'ex:paramType "{param_type}"',
            f"ex:isOptional {'true' if optional else 'false'}",
        ]
        default_value = param.get("default")
        if default_value not in (None, ""):
            parts.append(f"ex:defaultValue {_ttl_literal(str(default_value))}")
        type_node = str(param.get("type_node") or "").strip()
        if type_node:
            parts.append(f"ex:paramTypeNode [ a ex:{type_node} ]")
        blocks.append("[ " + " ; ".join(parts) + " ]")
    return blocks


def render_owl(template_path: Path, design_data: JsonDict, output_path: Path) -> None:
    template = template_path.read_text(encoding="utf-8")
    lines: list[str] = [template.rstrip()]

    objects = _ensure_list_of_dict(design_data.get("objects"))
    attributes = _ensure_list_of_dict(design_data.get("attributes"))
    edges = _ensure_list_of_dict(design_data.get("edges"))
    logics = _ensure_list_of_dict(design_data.get("logics"))
    actions = _ensure_list_of_dict(design_data.get("actions"))

    lines.extend(
        [
            "",
            "##################################",
            "# 本体名称",
            "##################################",
            f'ex:CustomerOpsOntology a ex:ontologyName ; rdfs:label "{design_data.get("ontology_name", "")}"@zh .',
            "",
            "##################################",
            "# 可用函数",
            "##################################",
        ]
    )
    if logics:
        for logic in logics:
            logic_id = str(logic.get("id") or "").strip()
            label = str(logic.get("label") or logic_id).strip()
            if not logic_id or not label:
                continue
            parts = [
                f"ex:{logic_id} a ex:Logic ;",
                f"rdfs:label {_ttl_literal(label)}@zh ;",
            ]
            comment = str(logic.get("comment") or "").strip()
            if comment:
                parts.append(f"rdfs:comment {_ttl_literal(comment)}@zh ;")
            param_blocks = _render_parameters(_ensure_list_of_dict(logic.get("parameters")))
            if param_blocks:
                parts.append("ex:hasParameter " + ", ".join(param_blocks) + " .")
                lines.append(" ".join(parts))
            else:
                parts[-1] = parts[-1].rstrip(";") + " ."
                lines.append(" ".join(parts))
    else:
        lines.append("# 暂无逻辑函数定义")

    lines.extend(
        [
            "",
            "##################################",
            "# Action动作定义",
            "##################################",
        ]
    )
    if actions:
        for action in actions:
            action_id = str(action.get("id") or "").strip()
            label = str(action.get("label") or action_id).strip()
            if not action_id or not label:
                continue
            parts = [
                f"ex:{action_id} a ex:Action ;",
                f"rdfs:label {_ttl_literal(label)}@zh ;",
            ]
            comment = str(action.get("comment") or "").strip()
            if comment:
                parts.append(f"rdfs:comment {_ttl_literal(comment)}@zh ;")
            param_blocks = _render_parameters(_ensure_list_of_dict(action.get("parameters")))
            if param_blocks:
                parts.append("ex:hasParameter " + ", ".join(param_blocks) + " .")
                lines.append(" ".join(parts))
            else:
                parts[-1] = parts[-1].rstrip(";") + " ."
                lines.append(" ".join(parts))
    else:
        lines.append("# 暂无动作定义")

    lines.extend(
        [
            "",
            "##################################",
            "# 本体对象",
            "##################################",
        ]
    )

    for obj in objects:
        obj_id = str(obj.get("id") or "").strip()
        label = str(obj.get("label") or "").strip()
        if not obj_id or not label:
            continue
        parts = [
            f"ex:{obj_id} a owl:Class ;",
            "rdfs:subClassOf ex:Object ;",
            f'rdfs:label "{label}"@zh ;',
        ]
        en_label = obj.get("en_label")
        if en_label and en_label != label:
            parts.append(f'rdfs:label "{en_label}"@en ;')
        parts[-1] = parts[-1].rstrip(";") + " ."
        lines.append(" ".join(parts))

    lines.extend(
        [
            "",
            "##################################",
            "# 本体属性",
            "##################################",
        ]
    )

    if attributes:
        for attr in attributes:
            attr_id = str(attr.get("id") or "").strip()
            label = str(attr.get("label") or "").strip()
            obj_id = str(attr.get("object_id") or "").strip()
            if not attr_id or not label or not obj_id:
                continue
            attr_range = _infer_xsd_type(attr)
            parts = [
                f"ex:{attr_id} a owl:DatatypeProperty ;",
                f'rdfs:label "{label}"@zh ;',
            ]
            en_label = attr.get("en_label")
            if en_label and en_label != label:
                parts.append(f'rdfs:label "{en_label}"@en ;')
            parts.extend(
                [
                    f"rdfs:domain ex:{obj_id} ;",
                    f"rdfs:range {attr_range} .",
                ]
            )
            lines.append(" ".join(parts))
    else:
        lines.append("# 暂无属性定义")

    lines.extend(
        [
            "",
            "##################################",
            "# 关系定义",
            "# 有正向关系就需定义一个反向关系",
            "##################################",
        ]
    )
    predicate_counts: dict[str, int] = {}
    for edge in edges:
        predicate = str(edge.get("predicate") or "").strip()
        if not predicate:
            continue
        base = _safe_relation_id(predicate)
        predicate_counts[base] = predicate_counts.get(base, 0) + 1
    predicate_index: dict[str, int] = {}
    used_relation_ids: set[str] = set()

    for edge in edges:
        predicate = str(edge.get("predicate") or "").strip()
        if not predicate or predicate not in RELATION_DEFS:
            continue
        subject = str(edge.get("subject") or "").strip()
        target = str(edge.get("object") or "").strip()
        if not subject or not target:
            continue
        base_rel_id = _safe_relation_id(predicate)
        base_inverse_id = _safe_relation_id(RELATION_DEFS[predicate]["inverse_id"])
        predicate_index[base_rel_id] = predicate_index.get(base_rel_id, 0) + 1
        suffix = f"_{predicate_index[base_rel_id]}" if predicate_counts[base_rel_id] > 1 else ""
        rel_id = _ensure_unique_relation_id(f"{base_rel_id}{suffix}", used_relation_ids)
        inverse_id = _ensure_unique_relation_id(f"{base_inverse_id}{suffix}", used_relation_ids)
        rel_label = str(edge.get("predicate_label") or "").strip() or RELATION_DEFS[predicate]["label"]
        inverse_label = RELATION_DEFS[predicate]["inverse_label"]
        domain_attr = str(edge.get("domain_attribute") or "").strip()
        range_attr = str(edge.get("range_attribute") or "").strip()
        lines.append(
            " ".join(
                [
                    f"ex:{rel_id} a owl:ObjectProperty ;",
                    "rdfs:subPropertyOf ex:custom_relation ;",
                    f"rdfs:domain ex:{subject} ;",
                    f"rdfs:range ex:{target} ;",
                    f"rdfs:label \"{rel_label}\"@zh ;",
                    f'rdfs:label "{base_rel_id}"@en ;',
                    *([f"ex:domainAttribute ex:{domain_attr} ;"] if domain_attr else []),
                    *([f"ex:rangeAttribute ex:{range_attr} ;"] if range_attr else []),
                    'rdfs:comment ""@zh .',
                ]
            )
        )
        lines.append(
            " ".join(
                [
                    f"ex:{inverse_id} a owl:ObjectProperty ;",
                    "rdfs:subPropertyOf ex:custom_relation ;",
                    f"rdfs:domain ex:{target} ;",
                    f"rdfs:range ex:{subject} ;",
                    f"rdfs:label \"{inverse_label}\"@zh ;",
                    f"owl:inverseOf ex:{rel_id} ;",
                    f'rdfs:label "{base_inverse_id}"@en ;',
                    *([f"ex:domainAttribute ex:{range_attr} ;"] if range_attr else []),
                    *([f"ex:rangeAttribute ex:{domain_attr} ;"] if domain_attr else []),
                    'rdfs:comment ""@zh .',
                ]
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
