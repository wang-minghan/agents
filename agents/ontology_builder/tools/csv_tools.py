from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]


def _ensure_list_of_dict(value: object) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def read_csv_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        return list(reader)


def build_table_summary(raw_meta: JsonDict, max_tables: int = 30, max_fields: int = 16) -> str:
    tables = _ensure_list_of_dict(raw_meta.get("tables"))
    lines: list[str] = []
    for table in tables[:max_tables]:
        table_name = table.get("table_name") or "未命名"
        table_en = table.get("table_en") or ""
        fields = _ensure_list_of_dict(table.get("fields"))[:max_fields]
        field_items = []
        for field in fields:
            field_name = field.get("field_name") or ""
            field_en = field.get("field_en") or ""
            if field_en:
                field_items.append(f"{field_name}({field_en})")
            else:
                field_items.append(field_name)
        field_text = "、".join(field_items) if field_items else "无"
        en_part = f"({table_en})" if table_en else ""
        lines.append(f"- 表：{table_name}{en_part} | 字段：{field_text}")
    relations = _ensure_list_of_dict(raw_meta.get("relations"))
    if relations:
        lines.append("- 关系摘要：")
        for rel in relations[:20]:
            subject = str(rel.get("subject") or "").strip()
            predicate = str(rel.get("predicate") or rel.get("relation") or "").strip()
            target = str(rel.get("object") or "").strip()
            if not subject or not predicate or not target:
                continue
            lines.append(f"  - {subject} {predicate} {target}")
    return "\n".join(lines)
