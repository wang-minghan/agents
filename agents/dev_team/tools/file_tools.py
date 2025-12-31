from pathlib import Path
from typing import Optional

def write_file(file_path: str, content: str) -> str:
    """
    将内容写入文件。
    :param file_path: 文件路径（相对于 output/codebase）
    :param content: 文件内容
    :return: 成功或失败的消息
    """
    try:
        # 强制将所有文件写入 output/codebase 目录
        base_dir = Path("agents/dev_team/output/codebase")
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # 移除路径中的 ../ 防止目录遍历攻击
        safe_path = file_path.replace("..", "")
        if safe_path.startswith("/"):
            safe_path = safe_path[1:]
            
        full_path = base_dir / safe_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return f"File successfully written to: {safe_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def read_file(file_path: str) -> str:
    """
    读取文件内容。
    :param file_path: 文件路径
    :return: 文件内容
    """
    try:
        base_dir = Path("agents/dev_team/output/codebase")
        safe_path = file_path.replace("..", "")
        if safe_path.startswith("/"):
            safe_path = safe_path[1:]
            
        full_path = base_dir / safe_path
        
        if not full_path.exists():
            return f"Error: File {safe_path} does not exist."
            
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def list_files(directory: str = ".") -> str:
    """
    列出目录下的文件。
    """
    try:
        base_dir = Path("agents/dev_team/output/codebase")
        target_dir = base_dir / directory
        
        if not target_dir.exists():
            return f"Directory {directory} does not exist."
            
        files = [str(f.relative_to(base_dir)) for f in target_dir.rglob("*") if f.is_file()]
        return "\n".join(files) if files else "No files found."
    except Exception as e:
        return f"Error listing files: {str(e)}"
