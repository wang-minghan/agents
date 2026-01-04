import yaml
import re
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

def find_project_root(start_path: Path) -> Path:
    """
    Traverse up to find the project root key markers (like .git or configs/).
    """
    current = start_path.resolve()
    for _ in range(10): # Max depth safety
        if (current / "configs" / "llm.yaml").exists():
            return current
        if (current / ".git").exists():
            return current
        
        parent = current.parent
        if parent == current: # System root reached
            break
        current = parent
    
    # Fallback to 2 levels up if not found (legacy behavior)
    return start_path.parent.parent

def load_config(config_path: str = None, base_dir: Path = None) -> Dict[str, Any]:
    """
    Unified config loader. 
    If base_dir is provided, it resolves relative config_path against it.
    It automatically looks for llm.yaml in the project root found relative to base_dir or __file__.
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

    # 加载统一的 LLM 配置
    project_root = find_project_root(base_dir)
    llm_config_path = project_root / "configs" / "llm.yaml"
    
    if llm_config_path.exists():
        with open(llm_config_path, "r", encoding="utf-8") as f:
            llm_settings = yaml.safe_load(f)
            active_profile = llm_settings.get("active_profile", "grok")
            profile = llm_settings.get("profiles", {}).get(active_profile, {})

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

def save_files_from_content(content: str, output_base_dir: Path) -> List[str]:
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
            
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 安全机制：如果文件存在，先创建备份
            if full_path.exists():
                backup_path = full_path.with_suffix(full_path.suffix + ".bak")
                shutil.copy2(full_path, backup_path)
                print(f"    ⚠️ 文件已存在，已备份至: {backup_path}")

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(file_content)
            saved_files.append(str(full_path))
            print(f"    └── 💾 已保存: {full_path}")
        except Exception as e:
            print(f"    ❌ 保存文件失败 {full_path}: {e}")

    return saved_files
