import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Union

from agents.dev_team.interfaces import CodeExecutor

class LocalUnsafeExecutor(CodeExecutor):
    """
    Executor that runs code locally via subprocess.
    WARNING: Only use this in a sandboxed or trusted environment.
    """
    def __init__(self, test_cmd: Optional[Union[str, Iterable[str], Iterable[Iterable[str]]]] = None, timeout: int = 60):
        self.test_cmd = test_cmd
        self.timeout = timeout

    def _discover_pytests(self, root: Path) -> List[Path]:
        patterns = ("test_*.py", "*_test.py")
        matches: List[Path] = []
        for pattern in patterns:
            matches.extend(sorted(root.rglob(pattern)))
        return sorted({p.resolve() for p in matches})

    def _normalize_cmd(self, test_dir: Path) -> Optional[Union[str, list]]:
        if self.test_cmd:
            if isinstance(self.test_cmd, str):
                return self.test_cmd
            command = list(self.test_cmd)
            if command and isinstance(command[0], (list, tuple)):
                return command
            return command

        if not self._discover_pytests(test_dir):
            return None
        return ["pytest", str(test_dir)]

    def run_tests(self, test_dir: str) -> str:
        path = Path(test_dir)
        if not path.exists():
            return "SKIPPED: Target directory for tests does not exist."

        if not self.test_cmd:
            test_files = self._discover_pytests(path)
            if not test_files:
                return "SKIPPED: No tests found."
            if len(test_files) > 1:
                cmd = [["pytest", str(test_file)] for test_file in test_files]
            else:
                cmd = ["pytest", str(test_files[0])]
        else:
            cmd = self._normalize_cmd(path)
        if not cmd:
            return "SKIPPED: No tests found."

        try:
            if isinstance(cmd, list) and cmd and isinstance(cmd[0], (list, tuple)):
                outputs = []
                for item in cmd:
                    result = subprocess.run(
                        list(item),
                        capture_output=True,
                        text=True,
                        timeout=self.timeout,
                    )
                    report = result.stdout + "\n" + result.stderr
                    outputs.append(report)
                    if result.returncode != 0:
                        return f"FAIL: Tests failed.\n{report[-2000:]}"
                merged = "\n".join(outputs)
                return f"SUCCESS: All tests passed.\n{merged[-1000:]}"

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                shell=isinstance(cmd, str),
            )
            report = result.stdout + "\n" + result.stderr
            if result.returncode == 0:
                return f"SUCCESS: All tests passed.\n{report[-1000:]}"
            return f"FAIL: Tests failed.\n{report[-2000:]}"
        except Exception as e:
            return f"ERROR: Execution failed: {str(e)}"

    def run_user_simulation(self, test_dir: str) -> str:
        path = Path(test_dir)
        if not path.exists():
            return "SKIPPED: Target directory for simulation does not exist."

        candidates = [
            path / "user_simulation.py",
            path / "tests" / "user_simulation.py",
        ]
        target = next((item for item in candidates if item.exists()), None)
        if not target:
            return "SKIPPED: No user simulation script found."

        try:
            result = subprocess.run(
                ["python", str(target)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            report = result.stdout + "\n" + result.stderr
            if result.returncode == 0:
                return f"SUCCESS: User simulation passed.\n{report[-1000:]}"
            return f"FAIL: User simulation failed.\n{report[-2000:]}"
        except Exception as e:
            return f"ERROR: Simulation failed: {str(e)}"
