from pathlib import Path

from agents.common import find_project_root, load_config as common_load_config, parse_code_blocks

def load_config(config_path: str = None):
    base_dir = Path(__file__).parent.resolve()
    config = common_load_config(config_path, base_dir)
    if "agent_root" not in config:
        config["agent_root"] = str(base_dir)
    return config

__all__ = ["find_project_root", "load_config", "parse_code_blocks"]
