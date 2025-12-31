import subprocess
from pathlib import Path
from agents.dev_team.interfaces import CodeExecutor

class SafeExecutor(CodeExecutor):
    """
    Default executor that refuses to run code.
    Used for safety by default.
    """
    def run_tests(self, test_dir: str) -> str:
        return (
            "SKIPPED: Automated execution is disabled in SafeExecutor mode.\n"
            "Result: Code generation complete, but manual verification is required."
        )

class LocalUnsafeExecutor(CodeExecutor):
    """
    Executor that runs code locally via subprocess.
    WARNING: Only use this in a sandboxed or trusted environment.
    """
    def run_tests(self, test_dir: str, test_cmd: list = None) -> str:
        path = Path(test_dir)
        if not path.exists():
            return "SKIPPED: Target directory for tests does not exist."
            
        try:
            # We assume pytest is installed in the environment
            cmd = test_cmd or ["pytest", str(path)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            report = result.stdout + "\n" + result.stderr
            if result.returncode == 0:
                return f"SUCCESS: All tests passed.\n{report[-1000:]}"
            else:
                return f"FAIL: Tests failed.\n{report[-2000:]}"
        except Exception as e:
            return f"ERROR: Execution failed: {str(e)}"
