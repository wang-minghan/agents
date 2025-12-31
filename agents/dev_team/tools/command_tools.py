import subprocess
import os
from pathlib import Path

def execute_command(command: str) -> str:
    """
    在 output/codebase 目录下执行 shell 命令。
    """
    try:
        base_dir = Path("agents/dev_team/output/codebase").absolute()
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # 切换工作目录并执行命令
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            timeout=30  # 避免命令挂死
        )
        
        output = result.stdout
        error = result.stderr
        
        if result.returncode == 0:
            return f"Command executed successfully.\nOutput:\n{output}"
        else:
            return f"Command failed (Return Code: {result.returncode}).\nError:\n{error}\nOutput:\n{output}"
            
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"
