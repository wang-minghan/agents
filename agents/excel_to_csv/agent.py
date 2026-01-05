"""Excel to CSV agent entry.

Converts multi-sheet Excel files to readable CSV while preserving directory structure.
LangChain runnable wrapper provides a standard agent-style entrypoint.
"""

from __future__ import annotations

import argparse
import contextvars
import csv
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yaml
from pydantic import SecretStr

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableLambda
    from langchain_openai import ChatOpenAI
    from langsmith import Client
    from langsmith.run_trees import RunTree

_LANGSMITH_CLIENT: "Client | None" = None
_LANGSMITH_PROJECT: str | None = None
_LANGSMITH_ENABLED = False
_CURRENT_TRACE: contextvars.ContextVar["RunTree | None"] = contextvars.ContextVar(
    "langsmith_trace",
    default=None,
)
_LLM_CACHE = threading.local()

def _iter_input_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".xlsx", ".xls", ".xlsm", ".csv"}:
            yield path


def _read_csv_preserve_columns(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame()
        rows = []
        for row in reader:
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            elif len(row) > len(header):
                row = row[: len(header)]
            rows.append(row)
    header = [col.strip() for col in header]
    return pd.DataFrame(rows, columns=header)


def _load_llm_config(config_path: Path) -> dict[str, object]:
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
        merged: dict[str, object] = {}
        common = data.get("common")
        if isinstance(common, dict):
            merged.update(common)
        if isinstance(selected, dict):
            merged.update(selected)
        return merged
    return data


def _load_langsmith_config(config_path: Path) -> dict[str, object]:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _configure_langsmith(config: dict[str, object]) -> None:
    if not config:
        return
    enabled = bool(config.get("enabled"))
    api_key = config.get("api_key")
    endpoint = config.get("endpoint")
    project = config.get("project")
    if not enabled or not api_key:
        return
    global _LANGSMITH_CLIENT, _LANGSMITH_PROJECT, _LANGSMITH_ENABLED
    _LANGSMITH_ENABLED = True
    _LANGSMITH_PROJECT = str(project) if project else None
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = str(api_key)
    if project:
        os.environ["LANGCHAIN_PROJECT"] = str(project)
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = str(api_key)
    if endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = str(endpoint)
        os.environ["LANGCHAIN_ENDPOINT"] = str(endpoint)
    if project:
        os.environ["LANGSMITH_PROJECT"] = str(project)
    try:
        from langsmith import Client

        _LANGSMITH_CLIENT = Client(
            api_url=str(endpoint) if endpoint else None,
            api_key=str(api_key),
        )
    except Exception:
        _LANGSMITH_CLIENT = None


def _build_llm(
    config: dict[str, object], model_override: str | None = None
) -> "ChatOpenAI":
    from langchain_openai import ChatOpenAI

    api_key = config.get("api_key")
    if not api_key:
        raise ValueError("LLM api_key is missing in config.")
    api_key_str = str(api_key)
    api_base = str(config.get("api_base", "https://api.deepseek.com"))
    model = model_override or str(config.get("model", "deepseek-chat"))
    timeout_seconds = _safe_int(config.get("timeout_seconds", 120), 120)
    max_retries = _safe_int(config.get("max_retries", 2), 2)
    return ChatOpenAI(
        api_key=SecretStr(api_key_str),
        base_url=api_base,
        model=model,
        temperature=0,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )


def _get_llm(config: dict[str, object], model_override: str | None = None) -> "ChatOpenAI":
    model = model_override or str(config.get("model", "deepseek-chat"))
    api_key = str(config.get("api_key") or "")
    api_base = str(config.get("api_base", "https://api.deepseek.com"))
    timeout_seconds = _safe_int(config.get("timeout_seconds", 120), 120)
    max_retries = _safe_int(config.get("max_retries", 2), 2)
    key = (model, api_key, api_base, timeout_seconds, max_retries)
    cache = getattr(_LLM_CACHE, "clients", None)
    if cache is None:
        cache = {}
        _LLM_CACHE.clients = cache
    if key not in cache:
        cache[key] = _build_llm(config, model_override=model)
    return cache[key]


def _extract_version(file_stem: str) -> str:
    match = re.search(r"v\d+(?:\.\d+)*", file_stem, re.IGNORECASE)
    return match.group(0) if match else "v1"


def _strip_version_tokens(name: str, version: str) -> str:
    cleaned = name.replace(version, "")
    cleaned = re.sub(r"\(\d+\)", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" -_")


def _infer_topic_table_from_name(file_stem: str, version: str) -> tuple[str, str]:
    parts = [part.strip() for part in file_stem.split(" - ") if part.strip()]
    if len(parts) >= 2:
        topic = _strip_version_tokens(parts[0], version)
        table = _strip_version_tokens(parts[-1], version)
        return topic or parts[0], table or parts[-1]
    return _strip_version_tokens(file_stem, version) or file_stem, ""


def _sample_df(df: pd.DataFrame, max_rows: int = 20, max_cols: int = 20) -> pd.DataFrame:
    return df.iloc[:max_rows, :max_cols]


def _summarize_df(df: pd.DataFrame, max_rows: int = 20, max_cols: int = 20) -> str:
    head = _sample_df(df, max_rows=max_rows, max_cols=max_cols)
    return head.to_csv(index=False, lineterminator="\n")


def _invoke_llm(
    llm: "ChatOpenAI",
    prompt: str,
    run_name: str,
    metadata: dict[str, object],
) -> str:
    parent_run = _CURRENT_TRACE.get()
    if _LANGSMITH_ENABLED and parent_run is not None:
        start_time = datetime.now(timezone.utc)
        child = parent_run.create_child(
            name=run_name,
            run_type="llm",
            inputs={"prompt": prompt},
            start_time=start_time,
            extra={"metadata": metadata},
        )
        parent_run.child_runs.append(child)
        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else response
            child.end(outputs={"response": content}, end_time=datetime.now(timezone.utc))
            return str(content)
        except Exception as exc:
            child.end(error=str(exc), end_time=datetime.now(timezone.utc))
            raise
    response = llm.invoke(prompt)
    return str(response.content if hasattr(response, "content") else response)


def _start_task_trace(
    source_path: Path,
    sheet_label: str,
    output_dir: Path,
) -> contextvars.Token["RunTree | None"] | None:
    if not _LANGSMITH_ENABLED or _LANGSMITH_CLIENT is None:
        return None
    from langsmith.run_trees import RunTree

    run = RunTree(
        name="excel_to_csv.run",
        run_type="chain",
        inputs={
            "file": str(source_path),
            "sheet": sheet_label,
            "output_dir": str(output_dir),
        },
        start_time=datetime.now(timezone.utc),
        extra={"metadata": {"file": str(source_path), "sheet": sheet_label}},
        project_name=_LANGSMITH_PROJECT or "default",
        ls_client=_LANGSMITH_CLIENT,
    )
    return _CURRENT_TRACE.set(run)


def _end_task_trace(
    token: contextvars.Token["RunTree | None"] | None,
    outputs: dict[str, object] | None = None,
    error: str | None = None,
) -> None:
    run = _CURRENT_TRACE.get()
    if run is not None:
        run.end(outputs=outputs or {}, error=error, end_time=datetime.now(timezone.utc))
        try:
            run.post(exclude_child_runs=False)
        except Exception as exc:
            print(f"[langsmith] post failed: {exc}", flush=True)
    if token is not None:
        _CURRENT_TRACE.reset(token)


def _extract_json(text: Any) -> dict[str, object]:
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
    raise json.JSONDecodeError("Invalid JSON", cleaned, 0)


def _transpose_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    transposed = df.transpose().reset_index()
    if not transposed.empty:
        candidate = transposed.iloc[0].astype(str).tolist()
        if len(set(candidate)) == len(candidate):
            transposed = transposed[1:].reset_index(drop=True)
            transposed.columns = candidate
    return transposed


def _merge_header_rows(df: pd.DataFrame, header_rows: list[int]) -> pd.DataFrame:
    if not header_rows:
        return df
    idxs = sorted({row - 1 for row in header_rows if row > 0})
    idxs = [i for i in idxs if 0 <= i < len(df)]
    if not idxs:
        return df
    header_parts = df.iloc[idxs].astype(str)
    merged_cols = []
    for col in header_parts.columns:
        parts = [value.strip() for value in header_parts[col].tolist() if value.strip()]
        merged_cols.append(" | ".join(parts) if parts else str(col))
    df = df.drop(df.index[idxs]).reset_index(drop=True)
    df.columns = merged_cols
    return df


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _english_ratio(series: pd.Series) -> float:
    values = series.astype(str).tolist()
    if not values:
        return 0.0
    ascii_count = 0
    total = 0
    for value in values:
        for ch in value:
            if ch.isascii():
                total += 1
                if ch.isalpha() or ch.isdigit() or ch in {" ", "_", "-"}:
                    ascii_count += 1
            else:
                total += 1
    return ascii_count / total if total else 0.0


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(col).replace("\n", " ").strip() for col in df.columns]
    cleaned = []
    unnamed_count = 0
    for col in cols:
        if not col or col.lower().startswith("unnamed"):
            unnamed_count += 1
            cleaned.append(f"unnamed_{unnamed_count}")
        else:
            cleaned.append(col)
    df.columns = cleaned
    return df


def _forward_fill_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].astype(str)
        df[col] = series.replace("", pd.NA).ffill().fillna("")
    return df


def _post_process_output(df: pd.DataFrame, plan: dict[str, object]) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}
    if "index" in df.columns:
        rename_map["index"] = "序号"
    if "英文编码.1" in df.columns:
        rename_map["英文编码.1"] = "反关系英文编码"
    if "反关系_英文编码" in df.columns:
        rename_map["反关系_英文编码"] = "反关系英文编码"
    if rename_map:
        df = df.rename(columns=rename_map)

    forward_cols = plan.get("forward_fill_cols") or []
    if isinstance(forward_cols, list):
        allowed = {"关系", "英文编码", "反关系", "反关系英文编码"}
        cols = [str(col) for col in forward_cols if str(col) in allowed]
        if cols:
            df = _forward_fill_columns(df, cols)
    return df


def _split_suffix_column(name: str) -> tuple[str, int | None]:
    cleaned = str(name).strip()
    group_match = re.match(r"^(.*?)(?:_组|组)(\d+)$", cleaned)
    if group_match:
        base = group_match.group(1).strip(" _.-")
        return base, int(group_match.group(2))
    for sep in (".", "_"):
        if sep in cleaned:
            base, suffix = cleaned.rsplit(sep, 1)
            if suffix.isdigit():
                return base.rstrip("._"), int(suffix)
    mid_match = re.match(r"^(.*?)(\d+)(\D+)$", cleaned)
    if mid_match:
        prefix, num, tail = mid_match.groups()
        if prefix and not prefix.isdigit():
            tail = tail.lstrip(" _.-")
            base = f"{prefix}{tail}".strip(" _.-")
            if base:
                return base, int(num)
    trailing_match = re.match(r"^(.*?)(\d+)$", cleaned)
    if trailing_match:
        prefix, num = trailing_match.groups()
        if prefix and not prefix.isdigit():
            base = prefix.strip(" _.-")
            if base:
                return base, int(num)
    return cleaned, None


def _should_auto_header(df: pd.DataFrame) -> bool:
    columns = [str(col).strip() for col in df.columns]
    if not columns:
        return False
    invalid = 0
    for name in columns:
        if not name:
            invalid += 1
            continue
        lowered = name.lower()
        if lowered.startswith("unnamed") or name.isdigit():
            invalid += 1
    return (invalid / len(columns)) >= 0.3


def _is_date_like(series: pd.Series) -> bool:
    sample = series.head(50).astype(str)
    hits = 0
    for value in sample:
        if re.search(r"\\d{4}[-/年]\\d{1,2}[-/月]\\d{1,2}", value):
            hits += 1
    return hits >= max(1, len(sample) // 5)


def _structure_diagnostics(df: pd.DataFrame) -> dict[str, object]:
    diag = {
        "shape": df.shape,
        "missing_ratio_by_col": {},
        "unique_ratio_by_col": {},
        "duplicate_columns": [],
        "date_like_columns": [],
        "mixed_type_columns": [],
    }
    sample = df.head(200)
    for col in df.columns:
        col_name = str(col)
        series = sample[col].astype(str)
        missing_ratio = (series == "").mean()
        unique_ratio = series.nunique(dropna=False) / max(1, len(series))
        diag["missing_ratio_by_col"][col_name] = round(missing_ratio, 3)
        diag["unique_ratio_by_col"][col_name] = round(unique_ratio, 3)
        if _is_date_like(series):
            diag["date_like_columns"].append(col_name)
        types = set()
        for value in series.head(50):
            if value == "":
                continue
            if re.fullmatch(r"-?\\d+(\\.\\d+)?", value):
                types.add("number")
            else:
                types.add("text")
        if len(types) > 1:
            diag["mixed_type_columns"].append(col_name)
    seen_cols = set()
    dup_cols = set()
    for col in df.columns:
        name = str(col)
        if name in seen_cols:
            dup_cols.add(name)
        else:
            seen_cols.add(name)
    diag["duplicate_columns"] = sorted(dup_cols)
    return diag


def _format_diagnostics(diag: dict[str, object]) -> str:
    missing = diag.get("missing_ratio_by_col")
    unique = diag.get("unique_ratio_by_col")
    missing_items = (
        list(cast(dict[str, float], missing).items())[:10]
        if isinstance(missing, dict)
        else []
    )
    unique_items = (
        list(cast(dict[str, float], unique).items())[:10]
        if isinstance(unique, dict)
        else []
    )
    return (
        f"shape={diag.get('shape')}\n"
        f"duplicate_columns={diag.get('duplicate_columns')}\n"
        f"date_like_columns={diag.get('date_like_columns')}\n"
        f"mixed_type_columns={diag.get('mixed_type_columns')}\n"
        f"missing_ratio_by_col(sample)={missing_items}\n"
        f"unique_ratio_by_col(sample)={unique_items}"
    )


def _safe_int(value: object, fallback: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def _stable_signature(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _df_signature(df: pd.DataFrame) -> str:
    head_csv = df.head(5).to_csv(index=False, lineterminator="\n")
    payload = {
        "shape": df.shape,
        "columns": [str(col) for col in df.columns],
        "head": head_csv,
    }
    return _stable_signature(payload)


def _detect_repeated_group_pattern(
    columns: list[str], min_pairs: int = 7
) -> tuple[bool, int]:
    # Detect repeated column groups like "X1/X2/.../Xn" across multiple bases.
    groups = _group_columns(columns)
    qualifying = [base for base, suffixes in groups.items() if len(suffixes) >= min_pairs]
    count = len(qualifying)
    return count >= 2, count


def _auto_header_rows(df: pd.DataFrame, max_rows: int = 5) -> list[int]:
    header_rows = []
    for idx in range(min(max_rows, len(df))):
        row = df.iloc[idx].astype(str)
        empty_ratio = (row == "").mean()
        if empty_ratio >= 0.7:
            header_rows.append(idx + 1)
    return header_rows


def _long_format_hint(df: pd.DataFrame) -> dict[str, object]:
    triggered, group_count = _detect_repeated_group_pattern(
        [str(c) for c in df.columns], min_pairs=7
    )
    wide = df.shape[1]
    suggested = wide > 50 and triggered
    reasons = []
    if wide > 50:
        reasons.append("列数>50")
    if triggered:
        reasons.append(f"重复组数={group_count}")
    reason_text = "，".join(reasons) if reasons else "未满足触发条件"
    return {
        "suggested": suggested,
        "reason": reason_text,
        "group_count": group_count,
    }


def _build_long_format(
    df: pd.DataFrame, min_bases: int, min_suffixes: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    groups_list: dict[str, dict[int, int]] = {}
    occurrence: dict[str, int] = {}
    for col_idx, col_name in enumerate(df.columns):
        base, suffix = _split_suffix_column(str(col_name))
        if not base:
            continue
        if suffix is None:
            occurrence[base] = occurrence.get(base, 0) + 1
            index = occurrence[base]
        else:
            index = suffix
        groups_list.setdefault(base, {})
        while index in groups_list[base]:
            index += 1
        groups_list[base][index] = col_idx
    bases = [base for base, items in groups_list.items() if len(items) >= min_suffixes]
    if len(bases) < min_bases:
        return df, {"eligible": False, "reason": "insufficient_bases"}
    indices = sorted(set().union(*(set(groups_list[base].keys()) for base in bases)))

    base_order: list[str] = []
    for col in df.columns:
        base, _ = _split_suffix_column(str(col))
        if base in bases and base not in base_order:
            base_order.append(base)

    static_idx = [
        idx
        for idx, col in enumerate(df.columns)
        if _split_suffix_column(str(col))[0] not in bases
    ]
    frames = []
    for idx in indices:
        chunk = df.iloc[:, static_idx].copy()
        for base in base_order:
            col_idx = groups_list.get(base, {}).get(idx)
            chunk[base] = df.iloc[:, col_idx] if col_idx is not None else ""
        empty_mask = chunk[base_order].astype(str).eq("").all(axis=1)
        chunk = chunk.loc[~empty_mask]
        frames.append(chunk)
    if not frames:
        return df, {"eligible": False, "reason": "empty_frames"}
    long_df = pd.concat(frames, ignore_index=True)
    base_cols = [idx for base in bases for idx in groups_list[base].values()]
    wide_non_empty = (
        df.iloc[:, base_cols].astype(str).ne("").sum().sum() if base_cols else 0
    )
    long_non_empty = (
        long_df[base_order].astype(str).ne("").sum().sum() if base_order else 0
    )
    return long_df, {
        "eligible": True,
        "bases": base_order,
        "wide_non_empty": int(wide_non_empty),
        "long_non_empty": int(long_non_empty),
    }


def _to_long_format(
    df: pd.DataFrame, min_bases: int = 2, min_suffixes: int = 2
) -> pd.DataFrame:
    long_df, _ = _build_long_format(df, min_bases, min_suffixes)
    return long_df


def _group_columns(columns: list[str]) -> dict[str, set[int]]:
    groups: dict[str, set[int]] = {}
    occurrence: dict[str, int] = {}
    for col in columns:
        base, suffix = _split_suffix_column(str(col))
        if not base:
            continue
        if suffix is None:
            occurrence[base] = occurrence.get(base, 0) + 1
            index = occurrence[base]
        else:
            index = suffix
        groups.setdefault(base, set())
        while index in groups[base]:
            index += 1
        groups[base].add(index)
    return groups


def _maybe_long_format(
    df: pd.DataFrame,
    min_bases: int,
    min_suffixes: int,
    min_col_reduction: float = 0.3,
    min_non_empty_ratio: float = 0.98,
) -> tuple[pd.DataFrame, dict[str, object]]:
    long_df, meta = _build_long_format(df, min_bases, min_suffixes)
    if not meta.get("eligible"):
        return df, {"applied": False, **meta}
    col_reduction = 1 - (long_df.shape[1] / max(1, df.shape[1]))
    wide_non_empty = _safe_int(meta.get("wide_non_empty"), 0)
    long_non_empty = _safe_int(meta.get("long_non_empty"), 0)
    non_empty_ratio = long_non_empty / wide_non_empty if wide_non_empty else 1.0
    if col_reduction < min_col_reduction:
        return df, {"applied": False, "reason": "low_reduction", **meta}
    if non_empty_ratio < min_non_empty_ratio:
        return df, {"applied": False, "reason": "non_empty_drop", **meta}
    return long_df, {
        "applied": True,
        "col_reduction": round(col_reduction, 3),
        "non_empty_ratio": round(non_empty_ratio, 3),
        **meta,
    }


def _select_model(df: pd.DataFrame, sheet_name: str, config: dict[str, object]) -> str:
    # Use reasoner for complex, wide, or multi-header sheets.
    reasoner = str(config.get("model_reasoner", "deepseek-reasoner"))
    chat = str(config.get("model", "deepseek-chat"))
    min_cols = _safe_int(config.get("reasoner_min_cols"), 80)
    min_name_len = _safe_int(config.get("reasoner_min_sheet_name_len"), 12)
    force_reasoner = str(config.get("force_reasoner", "")).lower() == "true"
    force_chat = str(config.get("force_chat", "")).lower() == "true"
    if force_chat:
        return chat
    if force_reasoner:
        return reasoner
    if df.shape[1] >= min_cols:
        return reasoner
    if sheet_name and len(sheet_name) >= min_name_len:
        return reasoner
    return chat


def _render_audit(audit: dict[str, object]) -> str:
    rename_map_raw = audit.get("rename_map") or {}
    rename_map: dict[str, str] = {}
    if isinstance(rename_map_raw, dict):
        rename_map = {str(k): str(v) for k, v in rename_map_raw.items()}
    rename_text = "\n".join([f"- {k} -> {v}" for k, v in rename_map.items()])
    if not rename_text:
        rename_text = "- 无"
    lines = [
        "# 表结构审核报告",
        "",
        f"- 主题: {audit.get('topic', '')}",
        f"- 版本: {audit.get('version', '')}",
        f"- 工作表: {audit.get('sheet_name', '')}",
        f"- 快速模式: {bool(audit.get('fast_mode', False))}",
        f"- 表头行: {audit.get('header_rows', [])}",
        f"- 转置: {audit.get('transpose', False)}",
        f"- 删除空行: {audit.get('drop_empty_rows', False)}",
        f"- 删除空列: {audit.get('drop_empty_columns', False)}",
        "",
        "## 列名调整",
        rename_text,
        "",
        "## 总结",
        str(audit.get("summary", "")),
        "",
    ]
    return "\n".join(lines)


def _fast_review_and_transform(
    df: pd.DataFrame,
    sheet_name: str,
    file_stem: str,
    allow_auto_header: bool,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    version = _extract_version(file_stem)
    inferred_topic, inferred_table = _infer_topic_table_from_name(file_stem, version)
    header_rows = _auto_header_rows(df) if allow_auto_header else []
    if header_rows:
        df = _merge_header_rows(df, header_rows)
    df = _normalize_columns(df)
    df = _post_process_output(df, {})
    audit = {
        "topic": inferred_topic,
        "table_name": inferred_table,
        "version": version,
        "sheet_name": sheet_name,
        "header_rows": header_rows,
        "rename_map": {},
        "transpose": False,
        "summary": "\n".join(
            [
                "总目标: 快速生成结构可读的 CSV。",
                "结构要点: 仅执行基础表头合并与列名清洗。",
                "结论: 快速模式已完成，未调用 LLM。",
            ]
        ),
        "fast_mode": True,
    }
    review = _rule_based_review(df)
    return df, audit, review

def _ai_review_and_transform(
    llm: ChatOpenAI,
    df: pd.DataFrame,
    sheet_name: str,
    file_stem: str,
    allow_auto_header: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    version = _extract_version(file_stem)
    auto_headers = _auto_header_rows(df) if allow_auto_header else []
    _, repeated_groups = _detect_repeated_group_pattern(
        [str(c) for c in df.columns], min_pairs=7
    )
    long_hint = _long_format_hint(df)
    diag = _structure_diagnostics(df)
    max_cols = 20 if df.shape[1] > 50 else df.shape[1]
    sample_csv = _summarize_df(df, max_rows=20, max_cols=max_cols)
    sample_cols = list(_sample_df(df, max_rows=1, max_cols=max_cols).columns)
    prompt = (
        "你是表结构审核专家。请根据表名与列名判断主题，并给出必要的清洗建议。"
        "只输出 JSON（不要代码块、不要多余文本），字段："
        "topic, table_name, header_rows, rename_map, transpose, summary。"
        "规则：列名必须有中文语义；不破坏信息；可读性差才转置；"
        "当列数>50且存在重复组（同名+编号）时，建议使用 long_format；"
        "只改变表现形式与输出结构，禁止改变数据内容；"
        "禁止删除数据、禁止清空或删除行列；"
        "保留原始重复列名，不要主动添加数字后缀或组序号；"
        "仅为空列名补充语义化名称；"
        "注意：当前列名可能是数字索引，表头可能位于前几行；"
        "header_rows 为表头所在的行号列表（从 1 开始计数），用于多行表头合并；"
        "rename_map 是旧列名到新列名的映射；summary 用中文并保持简洁。"
        "summary 必须包含三行：总目标/结构要点/结论。\n\n"
        f"文件名: {file_stem}\n工作表: {sheet_name}\n版本: {version}\n"
        f"表形状: {df.shape}\n重复组计数: {repeated_groups}\n"
        f"长表化建议: {long_hint}\n"
        f"结构诊断: {_format_diagnostics(diag)}\n"
        f"列名(截断): {sample_cols}\n"
        f"建议表头行(自动识别): {auto_headers}\n"
        f"样例(前20行前20列):\n{sample_csv}"
    )
    try:
        content = _invoke_llm(
            llm,
            prompt,
            run_name="excel_to_csv.initial_review",
            metadata={"file": file_stem, "sheet": str(sheet_name)},
        )
        data = _extract_json(content)
    except Exception as exc:
        data = {
            "topic": file_stem,
            "table_name": sheet_name,
            "rename_map": {},
            "transpose": False,
            "summary": f"LLM 解析失败，未进行结构调整。原因: {exc}",
        }

    rename_raw = data.get("rename_map") or {}
    rename_map: dict[str, str] = {}
    if isinstance(rename_raw, dict):
        rename_map = {str(k): str(v) for k, v in rename_raw.items()}
    if allow_auto_header:
        header_rows_raw = data.get("header_rows")
        if header_rows_raw is None:
            header_rows_raw = auto_headers
    else:
        header_rows_raw = []
    header_rows: list[int] = []
    if isinstance(header_rows_raw, list):
        for item in header_rows_raw:
            if isinstance(item, (int, float)) or (
                isinstance(item, str) and item.isdigit()
            ):
                header_rows.append(_safe_int(item, -1))
    header_rows = [row for row in header_rows if row > 0]
    if header_rows:
        df = _merge_header_rows(df, header_rows)
    df = df.rename(columns=rename_map)
    df = _normalize_columns(df)
    if data.get("transpose") and not header_rows:
        df = _transpose_if_needed(df)
    audit = {
        "topic": data.get("topic"),
        "table_name": data.get("table_name"),
        "version": version,
        "sheet_name": sheet_name,
        "header_rows": header_rows,
        "rename_map": rename_map,
        "transpose": bool(data.get("transpose")),
        "summary": data.get("summary", ""),
    }
    return df, audit


def _render_final_review(review: dict[str, object]) -> str:
    issues_raw = review.get("issues") or []
    suggestions_raw = review.get("suggestions") or []
    issues = [str(item) for item in issues_raw] if isinstance(issues_raw, list) else []
    suggestions = (
        [str(item) for item in suggestions_raw]
        if isinstance(suggestions_raw, list)
        else []
    )
    issues_text = "\n".join([f"- {item}" for item in issues]) if issues else "- 无"
    suggestions_text = (
        "\n".join([f"- {item}" for item in suggestions]) if suggestions else "- 无"
    )
    lines = [
        "## 最终审核",
        f"- 通过: {bool(review.get('approved'))}",
        "",
        "### 问题",
        issues_text,
        "",
        "### 建议",
        suggestions_text,
        "",
        "### 总结",
        str(review.get("summary", "")),
        "",
    ]
    return "\n".join(lines)


def _render_plan(plan: dict[str, object]) -> str:
    rename_raw = plan.get("rename_map") or {}
    rename_map: dict[str, str] = {}
    if isinstance(rename_raw, dict):
        rename_map = {str(k): str(v) for k, v in rename_raw.items()}
    rename_text = "\n".join([f"- {k} -> {v}" for k, v in rename_map.items()])
    if not rename_text:
        rename_text = "- 无"
    header_rows = plan.get("header_rows") or []
    forward_raw = plan.get("forward_fill_cols") or []
    forward_cols = (
        [str(item) for item in forward_raw] if isinstance(forward_raw, list) else []
    )
    forward_text = "、".join(forward_cols) if forward_cols else "无"
    long_stats = plan.get("long_format_stats") or {}
    lines = [
        "## 结构整改计划",
        f"- 表头行: {header_rows}",
        f"- 转置: {bool(plan.get('transpose'))}",
        f"- 长表化: {bool(plan.get('long_format'))}",
        f"- 向下填充列: {forward_text}",
        "",
        "### 列名调整",
        rename_text,
        "",
        "### 原因",
        str(plan.get("rationale", "")),
        f"### 长表化评估: {long_stats}" if long_stats else "",
        "",
    ]
    return "\n".join([line for line in lines if line != ""])


def _ai_final_review(
    llm: ChatOpenAI, df: pd.DataFrame, topic: str, table_name: str, version: str
) -> dict[str, object]:
    rule_review = _rule_based_review(df)
    if rule_review.get("approved"):
        return rule_review
    max_cols = 20 if df.shape[1] > 50 else df.shape[1]
    sample_csv = _summarize_df(df, max_rows=20, max_cols=max_cols)
    prompt = (
        "你是最终审核员。判断当前表结构是否清晰易懂，是否满足基本表格守则。"
        "只输出 JSON（不要代码块、不要多余文本），字段："
        "approved, issues, suggestions, summary。"
        "approved 为 true/false。issues/suggestions 为中文列表。"
        "审核不涉及内容质量，只关注结构与可读性；"
        "只改变表现形式与输出结构，禁止建议修改数据内容；"
        "禁止建议删除数据或清空行列；"
        "summary 必须包含三行：总目标/结构要点/结论。\n\n"
        f"命名: {topic}_{table_name}_{version}\n"
        f"列名(截断): {list(_sample_df(df, max_rows=1, max_cols=max_cols).columns)}\n"
        f"样例(前20行前20列):\n{sample_csv}"
    )
    content = _invoke_llm(
        llm,
        prompt,
        run_name="excel_to_csv.final_review",
        metadata={"topic": topic, "table_name": table_name, "version": version},
    )
    return _extract_json(content)


def _rule_based_review(df: pd.DataFrame) -> dict[str, object]:
    columns = [str(col) for col in df.columns]
    issues = []
    seen = set()
    for col in columns:
        if col in seen:
            issues.append(f"列名重复: {col}")
        seen.add(col)
        if col.strip() == "" or col.lower().startswith("unnamed"):
            issues.append(f"列名为空或无意义: {col}")
        if col.isdigit():
            issues.append(f"列名为纯数字: {col}")
    approved = not issues
    summary = "\n".join(
        [
            "总目标: 输出结构清晰可读的表格。",
            "结构要点: 列名语义明确且不重复。",
            "结论: 通过。" if approved else "结论: 未通过。",
        ]
    )
    return {
        "approved": approved,
        "issues": issues,
        "suggestions": [],
        "summary": summary,
    }


def _ai_feedback_plan(
    llm: ChatOpenAI,
    df: pd.DataFrame,
    feedback: dict[str, object],
    topic: str,
    table_name: str,
    version: str,
) -> dict[str, object]:
    _, repeated_groups = _detect_repeated_group_pattern(
        [str(c) for c in df.columns], min_pairs=7
    )
    long_hint = _long_format_hint(df)
    diag = _structure_diagnostics(df)
    max_cols = 20 if df.shape[1] > 50 else df.shape[1]
    sample_csv = _summarize_df(df, max_rows=20, max_cols=max_cols)
    prompt = (
        "你是结构整改执行者。基于最终审核反馈，生成可执行的结构整改计划。"
        "只输出 JSON（不要代码块、不要多余文本），字段："
        "header_rows, rename_map, transpose, long_format, forward_fill_cols, rationale。"
        "规则：只能调整列名、转置或长表化；"
        "forward_fill_cols 仅允许包含 [关系, 英文编码, 反关系, 反关系英文编码]；"
        "仅当长表化后这些列出现空值且逻辑上需要重复时才向下填充；"
        "只改变表现形式与输出结构，禁止改变数据内容；"
        "禁止删除/清空任何数据或行列；"
        "保留原始重复列名，不要主动添加数字后缀或组序号；"
        "仅为空列名补充语义化名称。\n\n"
        "注意：当前列名可能是数字索引，表头可能位于前几行。\n\n"
        f"命名: {topic}_{table_name}_{version}\n"
        f"表形状: {df.shape}\n重复组计数: {repeated_groups}\n"
        f"长表化建议: {long_hint}\n"
        f"结构诊断: {_format_diagnostics(diag)}\n"
        f"列名(截断): {list(_sample_df(df, max_rows=1, max_cols=max_cols).columns)}\n"
        f"审核反馈: {feedback}\n"
        f"样例(前20行前20列):\n{sample_csv}"
    )
    content = _invoke_llm(
        llm,
        prompt,
        run_name="excel_to_csv.feedback_plan",
        metadata={"topic": topic, "table_name": table_name, "version": version},
    )
    return _extract_json(content)


def _apply_feedback(
    df: pd.DataFrame,
    plan: dict[str, object],
    allow_long_format: bool,
    allow_auto_header: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if allow_auto_header:
        header_rows_raw = plan.get("header_rows")
        if header_rows_raw is None:
            header_rows_raw = _auto_header_rows(df)
    else:
        header_rows_raw = []
    header_rows: list[int] = []
    if isinstance(header_rows_raw, list):
        for item in header_rows_raw:
            if isinstance(item, (int, float)) or (
                isinstance(item, str) and item.isdigit()
            ):
                header_rows.append(_safe_int(item, -1))
    header_rows = [row for row in header_rows if row > 0]
    if header_rows:
        df = _merge_header_rows(df, header_rows)
    rename_raw = plan.get("rename_map") or {}
    rename_map: dict[str, str] = {}
    if isinstance(rename_raw, dict):
        rename_map = {str(k): str(v) for k, v in rename_raw.items()}
    if rename_map:
        df = df.rename(columns=rename_map)
    df = _normalize_columns(df)
    if plan.get("transpose") and not header_rows:
        df = _transpose_if_needed(df)
    if allow_long_format and plan.get("long_format"):
        min_bases = _safe_int(plan.get("long_format_min_bases"), 2)
        min_suffixes = _safe_int(plan.get("long_format_min_suffixes"), 7)
        df_long, meta = _maybe_long_format(
            df, min_bases=min_bases, min_suffixes=min_suffixes
        )
        if meta.get("applied"):
            df = df_long
            plan["long_format_stats"] = meta
    df = _post_process_output(df, plan)
    forward_fill_cols: list[str] = []
    forward_raw = plan.get("forward_fill_cols") or []
    if isinstance(forward_raw, list):
        forward_fill_cols = [str(item) for item in forward_raw]
    return df, {
        "header_rows": header_rows,
        "rename_map": rename_map,
        "transpose": bool(plan.get("transpose")),
        "long_format": bool(plan.get("long_format")),
        "forward_fill_cols": forward_fill_cols,
        "rationale": plan.get("rationale", ""),
    }


def _process_dataframe(
    df: pd.DataFrame,
    source_path: Path,
    sheet_label: str,
    output_dir: Path,
    llm_config: dict[str, object],
    max_rounds: int,
    skip_llm: bool,
    is_csv: bool,
) -> None:
    trace_token = _start_task_trace(source_path, str(sheet_label), output_dir)
    print(
        f"[excel_to_csv] start file={source_path.name} sheet={sheet_label}",
        flush=True,
    )
    try:
        allow_auto_header = (not is_csv) and _should_auto_header(df)
        version = _extract_version(source_path.stem)
        inferred_topic, inferred_table = _infer_topic_table_from_name(
            source_path.stem, version
        )
        final_review: dict[str, object] = {}
        final_plan: dict[str, object] = {}
        if skip_llm:
            print(
                f"[excel_to_csv] fast_mode enabled file={source_path.name} sheet={sheet_label}",
                flush=True,
            )
            df, audit, final_review = _fast_review_and_transform(
                df,
                sheet_name=str(sheet_label),
                file_stem=source_path.stem,
                allow_auto_header=allow_auto_header,
            )
            topic = str(audit.get("topic") or inferred_topic or source_path.stem)
            table_name = str(audit.get("table_name") or inferred_table or sheet_label)
            version = str(audit.get("version") or version or "v1")
        else:
            model = _select_model(df, sheet_label, llm_config)
            llm = _get_llm(llm_config, model_override=model)
            df, audit = _ai_review_and_transform(
                llm,
                df,
                sheet_name=str(sheet_label),
                file_stem=source_path.stem,
                allow_auto_header=allow_auto_header,
            )
            print(
                f"[excel_to_csv] initial_review done file={source_path.name} sheet={sheet_label}",
                flush=True,
            )
            topic = str(audit.get("topic") or inferred_topic or source_path.stem)
            table_name = str(audit.get("table_name") or inferred_table or sheet_label)
            version = str(audit.get("version") or version or "v1")
            allow_long_format = bool(llm_config.get("allow_long_format", True))
            seen_states: set[str] = set()
            for _round in range(1, max_rounds + 1):
                print(
                    f"[excel_to_csv] final_review round={_round} file={source_path.name} sheet={sheet_label}",
                    flush=True,
                )
                final_review = _ai_final_review(llm, df, topic, table_name, version)
                if bool(final_review.get("approved")):
                    break
                print(
                    f"[excel_to_csv] feedback_plan round={_round} file={source_path.name} sheet={sheet_label}",
                    flush=True,
                )
                plan = _ai_feedback_plan(llm, df, final_review, topic, table_name, version)
                before_sig = _df_signature(df)
                df, final_plan = _apply_feedback(
                    df, plan, allow_long_format, allow_auto_header
                )
                after_sig = _df_signature(df)
                state_sig = _stable_signature(
                    {
                        "review": final_review,
                        "plan": final_plan,
                        "df": after_sig,
                    }
                )
                if state_sig in seen_states or after_sig == before_sig:
                    final_review = {
                        "approved": True,
                        "issues": final_review.get("issues", []),
                        "suggestions": final_review.get("suggestions", []),
                        "summary": "\n".join(
                            [
                                "总目标: 输出结构清晰可读的表格。",
                                "结构要点: 检测到重复整改且结构未变化。",
                                "结论: 检测到循环，自动通过。",
                            ]
                        ),
                    }
                    print(
                        f"[excel_to_csv] loop_detected auto_approve file={source_path.name} sheet={sheet_label}",
                        flush=True,
                    )
                    break
                seen_states.add(state_sig)

        out_name = f"{topic}_{table_name}_{version}.csv"
        out_path = output_dir / out_name
        df.to_csv(
            str(out_path),
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        audit_path = output_dir / f"{topic}_{table_name}_{version}__audit.md"
        audit_text = _render_audit(audit)
        audit_text += _render_final_review(final_review)
        if final_plan:
            audit_text += _render_plan(final_plan)
        audit_path.write_text(audit_text, encoding="utf-8")
        _end_task_trace(
            trace_token,
            outputs={
                "output_csv": str(out_path),
                "approved": bool(final_review.get("approved")),
            },
        )
        print(
            f"[excel_to_csv] done file={source_path.name} sheet={sheet_label} out={out_path.name}",
            flush=True,
        )
    except Exception as exc:
        _end_task_trace(trace_token, error=str(exc))
        raise


def _process_sheet(
    source_path: Path,
    sheet_name: str | None,
    output_dir: Path,
    llm_config: dict[str, object],
    max_rounds: int,
    skip_llm: bool = False,
) -> None:
    sheet_label = sheet_name or source_path.stem
    is_csv = source_path.suffix.lower() == ".csv"
    if is_csv:
        df = _read_csv_preserve_columns(source_path)
    else:
        sheet_key = str(sheet_name) if sheet_name is not None else 0
        df = pd.read_excel(
            source_path,
            sheet_name=sheet_key,
            dtype=object,
            header=None,
            keep_default_na=False,
        )
    _process_dataframe(
        df,
        source_path,
        str(sheet_label),
        output_dir,
        llm_config,
        max_rounds,
        skip_llm,
        is_csv,
    )


def convert_excel_dir(
    input_dir: Path,
    output_dir: Path,
    llm_config: dict[str, object],
    max_rounds: int = 5,
    max_workers: int | None = None,
    max_file_workers: int | None = None,
    skip_llm: bool = False,
) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    def _process_file(source_path: Path) -> None:
        rel_path = source_path.relative_to(input_dir)
        out_parent = output_dir / rel_path.parent
        out_parent.mkdir(parents=True, exist_ok=True)

        if source_path.suffix.lower() == ".csv":
            _process_sheet(
                source_path,
                None,
                out_parent,
                llm_config,
                max_rounds,
                skip_llm,
            )
            return

        excel = pd.ExcelFile(source_path)
        configured_workers = _safe_int(
            llm_config.get("max_workers", min(4, (os.cpu_count() or 2))), 2
        )
        worker_limit = (
            _safe_int(max_workers, configured_workers)
            if max_workers is not None
            else configured_workers
        )
        if worker_limit <= 1:
            for sheet_name in excel.sheet_names:
                df = excel.parse(
                    sheet_name=sheet_name,
                    dtype=object,
                    header=None,
                    keep_default_na=False,
                )
                _process_dataframe(
                    df,
                    source_path,
                    str(sheet_name),
                    out_parent,
                    llm_config,
                    max_rounds,
                    skip_llm,
                    is_csv=False,
                )
            return

        with ThreadPoolExecutor(max_workers=worker_limit) as executor:
            futures = []
            for sheet_name in excel.sheet_names:
                futures.append(
                    executor.submit(
                        _process_sheet,
                        source_path,
                        str(sheet_name),
                        out_parent,
                        llm_config,
                        max_rounds,
                        skip_llm,
                    )
                )
            for future in as_completed(futures):
                future.result()

    files = list(_iter_input_files(input_dir))
    if not files:
        return
    if max_file_workers is None:
        max_file_workers = _safe_int(llm_config.get("max_file_workers", 1), 1)
    if max_file_workers <= 1:
        for source_path in files:
            _process_file(source_path)
    else:
        with ThreadPoolExecutor(max_workers=max_file_workers) as executor:
            futures = [executor.submit(_process_file, path) for path in files]
            for future in as_completed(futures):
                future.result()


def _run(payload: dict[str, object]) -> dict[str, str]:
    input_dir = Path(payload["input_dir"])
    output_dir = Path(payload["output_dir"])
    base_dir = Path(__file__).parent.resolve()
    default_llm = str(base_dir / "config" / "llm.yaml")
    default_langsmith = str(base_dir / "config" / "langsmith.yaml")
    config_path = Path(payload.get("llm_config", default_llm))
    langsmith_path = Path(payload.get("langsmith_config", default_langsmith))
    max_rounds = _safe_int(payload.get("max_rounds"), 5)
    max_workers_raw = payload.get("max_workers")
    max_workers = (
        _safe_int(max_workers_raw, 0) if max_workers_raw is not None else None
    )
    if max_workers == 0:
        max_workers = None
    skip_llm_flag = str(payload.get("skip_llm", "")).lower()
    skip_llm = skip_llm_flag in {"1", "true", "yes", "y", "on"}
    langsmith_config = _load_langsmith_config(langsmith_path)
    _configure_langsmith(langsmith_config)
    if langsmith_config:
        key = str(langsmith_config.get("api_key", ""))
        masked = f"{key[:4]}...{key[-4:]}" if len(key) >= 8 else "missing"
        print(
            "[langsmith] enabled="
            f"{bool(langsmith_config.get('enabled'))} "
            f"project={langsmith_config.get('project')} "
            f"endpoint={langsmith_config.get('endpoint')} "
            f"api_key={masked}",
            flush=True,
        )
    llm_config = _load_llm_config(config_path)
    convert_excel_dir(
        input_dir,
        output_dir,
        llm_config,
        max_rounds=max_rounds,
        max_workers=max_workers,
        skip_llm=skip_llm,
    )
    return {"status": "ok", "input_dir": str(input_dir), "output_dir": str(output_dir)}


def build_agent() -> RunnableLambda:
    from langchain_core.runnables import RunnableLambda

    return RunnableLambda(_run)


def _run_cli() -> None:
    parser = argparse.ArgumentParser(description="Convert Excel files to CSV.")
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--llm-config",
        default=str(Path(__file__).parent.resolve() / "config" / "llm.yaml"),
        help="Path to LLM config YAML",
    )
    parser.add_argument(
        "--langsmith-config",
        default=str(Path(__file__).parent.resolve() / "config" / "langsmith.yaml"),
        help="Path to LangSmith config YAML",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum AI review iterations per sheet (default: 5)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Override per-file sheet worker count",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM review for faster processing",
    )
    args = parser.parse_args()

    agent = build_agent()
    agent.invoke(
        {
            "input_dir": args.input,
            "output_dir": args.output,
            "llm_config": args.llm_config,
            "langsmith_config": args.langsmith_config,
            "max_rounds": args.max_rounds,
            "max_workers": args.max_workers,
            "skip_llm": args.skip_llm,
        }
    )


if __name__ == "__main__":
    _run_cli()
