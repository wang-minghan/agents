import ast
import yaml
import re
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

def _looks_like_stub_python(content: str) -> bool:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return False
    stub_lines = 0
    def_or_class = 0
    for line in lines:
        if line.startswith("def ") or line.startswith("class "):
            def_or_class += 1
        if line in {"...", "pass"} or line.endswith(": ..."):
            stub_lines += 1
    if def_or_class >= 3 and stub_lines >= max(3, int(def_or_class * 0.6)):
        return True
    if stub_lines >= 8 and def_or_class == 0:
        return True
    return False

def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    cleaned = cleaned.strip("`")
    newline_idx = cleaned.find("\n")
    if newline_idx != -1:
        cleaned = cleaned[newline_idx + 1 :]
    return cleaned.strip()

def _parse_function_patch_blocks(content: str) -> Dict[str, str]:
    pattern = r"<function name=[\"'](.*?)[\"']>\\s*(.*?)\\s*</function>"
    matches = re.findall(pattern, content, re.DOTALL)
    return {name.strip(): body.strip() for name, body in matches if name.strip()}

def _index_functions(tree: ast.Module) -> Dict[str, ast.AST]:
    mapping: Dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            mapping[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    mapping[f"{node.name}.{item.name}"] = item
    return mapping

def _extract_patch_body_nodes(patch_code: str) -> List[ast.stmt] | None:
    cleaned = _strip_code_fence(patch_code)
    try:
        module = ast.parse(cleaned)
    except Exception:
        return None
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return list(node.body)
    return None

def _is_stub_body(body_nodes: List[ast.stmt]) -> bool:
    if len(body_nodes) != 1:
        return False
    node = body_nodes[0]
    if isinstance(node, ast.Pass):
        return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        return node.value.value is Ellipsis
    return False

def _render_body_lines(body_nodes: List[ast.stmt], indent: str) -> List[str]:
    rendered: List[str] = []
    for stmt in body_nodes:
        text = ast.unparse(stmt)
        if not text:
            continue
        for line in text.splitlines():
            if line:
                rendered.append(indent + line)
            else:
                rendered.append("")
    if not rendered:
        rendered = [indent + "pass"]
    return rendered

def _apply_function_patches(
    original_text: str,
    patches: Dict[str, str],
) -> Tuple[str, List[str], List[str], List[str]]:
    try:
        tree = ast.parse(original_text)
    except Exception:
        return original_text, [], list(patches.keys()), []
    index = _index_functions(tree)
    lines = original_text.splitlines()
    replacements: List[Tuple[int, int, List[str]]] = []
    applied: List[str] = []
    missing: List[str] = []
    skipped: List[str] = []

    for name, patch_code in patches.items():
        target = index.get(name)
        if target is None:
            missing.append(name)
            continue
        body_nodes = _extract_patch_body_nodes(patch_code)
        if not body_nodes or _is_stub_body(body_nodes):
            skipped.append(name)
            continue
        body = getattr(target, "body", None)
        if not body:
            missing.append(name)
            continue
        start_line = body[0].lineno
        end_line = getattr(body[-1], "end_lineno", body[-1].lineno)
        indent = " " * body[0].col_offset
        new_body_lines = _render_body_lines(body_nodes, indent)
        replacements.append((start_line, end_line, new_body_lines))
        applied.append(name)

    for start, end, new_lines in sorted(replacements, key=lambda x: x[0], reverse=True):
        lines[start - 1 : end] = new_lines

    updated = "\n".join(lines)
    if original_text.endswith("\n"):
        updated += "\n"
    return updated, applied, missing, skipped

def find_project_root(start_path: Path) -> Path:
    """
    Traverse up to find the project root key markers (like .git or pyproject.toml).
    """
    current = start_path.resolve()
    for _ in range(10): # Max depth safety
        if (current / "pyproject.toml").exists():
            return current
        if (current / ".git").exists():
            return current

        parent = current.parent
        if parent == current: # System root reached
            break
        current = parent

    # Fallback to 2 levels up if not found (legacy behavior)
    return start_path.parent.parent


def _find_llm_config(base_dir: Path) -> Optional[Path]:
    current = base_dir.resolve()
    for _ in range(8):
        for candidate in (
            current / "config" / "llm.yaml",
            current / "config" / "llm.yml",
        ):
            if candidate.exists():
                return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _resolve_llm_api_key(profile: Dict[str, Any]) -> Optional[str]:
    api_key = str(profile.get("api_key") or "").strip()
    if api_key and not api_key.startswith("<"):
        return api_key
    for env_name in (
        "LLM_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    ):
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value
    return api_key or None

def load_config(config_path: str = None, base_dir: Path = None) -> Dict[str, Any]:
    """
    Unified config loader. 
    If base_dir is provided, it resolves relative config_path against it.
    It automatically looks for llm.yaml in the nearest agent config folder.
    """
    if base_dir is None:
        base_dir = Path(__file__).parent
        
    if config_path is None:
        # Default assumption: config.yaml in config/ folder relative to base_dir
        # But this might vary per agent. 
        # Note: If called from common.py without explicit path, this default might be wrong for specific agents.
        # So it is better if agents explicitly pass their default path or we assume a standard.
        # For now, let's keep the logic simple: if None, try to find it in likely places or error?
        # To match previous behavior, we might check if base_dir has a config/config.yaml
        possible_path = base_dir / "config" / "config.yaml"
        if possible_path.exists():
            config_path = str(possible_path)
        else:
             # Fallback to just "config.yaml"
            config_path = str(base_dir / "config.yaml")

    path_obj = Path(config_path)
    if not path_obj.is_absolute():
        path_obj = base_dir / config_path
            
    if not path_obj.exists():
        # Just return empty or warning if strictly needed, but better to fail if config is expected
        print(f"Warning: Config file not found at {path_obj}, returning empty config.")
        config = {}
    else:
        with open(path_obj, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    # 加载 agent 独立的 LLM 配置
    llm_config_path = _find_llm_config(base_dir)

    if llm_config_path and llm_config_path.exists():
        with open(llm_config_path, "r", encoding="utf-8") as f:
            llm_settings = yaml.safe_load(f) or {}
            active_profile = llm_settings.get("active_profile", "grok")
            profile = llm_settings.get("profiles", {}).get(active_profile, {})
            resolved_api_key = _resolve_llm_api_key(profile)
            if resolved_api_key:
                profile["api_key"] = resolved_api_key

            # 注入配置: Global override or detailed injection
            # dev_team uses config["llm"] = profile
            # architect injects into roles if missing.
            # We support both: put it in "llm" key, and agents can use it as they see fit.
            config["llm"] = profile

            # Helper for task planner structure:
            if "roles" in config:
                for role_name, role_cfg in config["roles"].items():
                    if isinstance(role_cfg, dict):
                        if "model" not in role_cfg:
                            role_cfg["model"] = profile.get("model")
                        if "api_key" not in role_cfg:
                            role_cfg["api_key"] = profile.get("api_key")
                        if "api_base" not in role_cfg:
                            role_cfg["api_base"] = profile.get("api_base")

    return config

def parse_code_blocks(text: str) -> List[Tuple[str, str]]:
    """
    Parses code blocks from text. Supports both XML-style <file path="...">...</file> 
    and Markdown-style ```python:<path> ... ``` (future proofing).
    
    Returns:
        List of (file_path, content) tuples.
    """
    results = []
    
    # 1. XML Style: <file path="path/to/file">content</file>
    xml_pattern = r'<file path=[\"\'](.*?)[\"\']>\s*(.*?)\s*</file>'
    matches = re.findall(xml_pattern, text, re.DOTALL)
    results.extend(matches)
    
    return results

def save_files_from_content(
    content: str,
    output_base_dir: Path,
    update_mode: bool | None = None,
    reserved_paths: set[str] | None = None,
) -> List[str]:
    """
    Extracts file blocks from content and saves them to the output directory.
    Replaces RoleAgent.extract_and_save_files.
    """
    matches = parse_code_blocks(content)
    saved_files = []
    
    if not output_base_dir:
        print("Warning: No output directory provided, skipping file save.")
        return []

    base_path = output_base_dir.resolve()
    if update_mode is None:
        update_mode = bool(os.environ.get("DEV_TEAM_ITERATION_TARGET"))
    if reserved_paths is None:
        reserved_paths = set()

    warned_paths = set()

    for path, file_content in matches:
        # 清理路径 & 安全检查
        clean_path = path.strip().replace("\\", "/")
        if not clean_path:
            continue

        try:
            path_obj = Path(clean_path)
            if path_obj.is_absolute():
                full_path = path_obj.resolve()
                try:
                    rel_path = full_path.relative_to(base_path)
                except ValueError:
                    print(f"    ⚠️ 忽略不安全的文件路径: {clean_path}")
                    continue
                if clean_path not in warned_paths:
                    print(f"    ⚠️ 绝对路径已归一化: {clean_path} -> {rel_path.as_posix()}")
                    warned_paths.add(clean_path)
            else:
                if ".." in path_obj.parts:
                    print(f"    ⚠️ 忽略不安全的文件路径: {clean_path}")
                    continue
                full_path = (base_path / path_obj).resolve()
                try:
                    full_path.relative_to(base_path)
                except ValueError:
                    print(f"    ⚠️ 路径越界保护: {clean_path}")
                    continue

            if full_path.exists() and full_path.is_dir():
                print(f"    ⚠️ 忽略目录路径: {clean_path}")
                continue
            if str(full_path) in reserved_paths:
                print(f"    ⚠️ 跳过冲突文件(已由其他角色生成): {clean_path}")
                continue

            function_patches = _parse_function_patch_blocks(file_content)
            patch_mode = bool(function_patches)
            if patch_mode:
                if not full_path.exists():
                    print(f"    ⚠️ 更新片段但文件不存在: {clean_path}")
                    continue
                try:
                    original = full_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    original = ""
                updated_text, applied, missing, skipped = _apply_function_patches(
                    original, function_patches
                )
                if missing:
                    print(f"    ⚠️ 未找到函数: {', '.join(missing)} in {clean_path}")
                if skipped:
                    print(f"    ⚠️ 跳过疑似占位函数: {', '.join(skipped)} in {clean_path}")
                if not applied:
                    print(f"    ⚠️ 未应用任何函数更新: {clean_path}")
                    continue
                file_content = updated_text
            else:
                if full_path.suffix == ".py" and _looks_like_stub_python(file_content):
                    print(f"    ⚠️ 跳过疑似占位代码: {clean_path}")
                    continue

            if update_mode and full_path.exists() and full_path.is_file() and not patch_mode:
                try:
                    original = full_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    original = ""
                if original and len(file_content) < int(len(original) * 0.6):
                    print(f"    ⚠️ 更新模式下跳过明显缩减的文件: {clean_path}")
                    continue
            
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 安全机制：如果文件存在，先创建备份
            if full_path.exists():
                backup_path = full_path.with_suffix(full_path.suffix + ".bak")
                shutil.copy2(full_path, backup_path)
                print(f"    ⚠️ 文件已存在，已备份至: {backup_path}")

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(file_content)
            saved_files.append(str(full_path))
            reserved_paths.add(str(full_path))
            print(f"    └── 💾 已保存: {full_path}")
        except Exception as e:
            print(f"    ❌ 保存文件失败 {full_path}: {e}")

    return saved_files
