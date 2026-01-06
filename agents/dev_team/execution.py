import subprocess
import shlex
import fnmatch
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Union

from agents.dev_team.interfaces import CodeExecutor

class LocalUnsafeExecutor(CodeExecutor):
    """
    Executor that runs code locally via subprocess.
    WARNING: Only use this in a sandboxed or trusted environment.
    """
    def __init__(
        self,
        test_cmd: Optional[Union[str, Iterable[str], Iterable[Iterable[str]]]] = None,
        timeout: int = 60,
        per_file: bool = True,
        ui_test_patterns: Optional[Iterable[str]] = None,
        coverage_cmd: Optional[Union[str, Iterable[str]]] = None,
        coverage_timeout: int = 120,
    ):
        self.test_cmd = test_cmd
        self.timeout = timeout
        self.per_file = per_file
        self.ui_test_patterns = list(ui_test_patterns or [])
        self.coverage_cmd = coverage_cmd
        self.coverage_timeout = coverage_timeout

    def _discover_pytests(self, root: Path) -> List[Path]:
        patterns = ("test_*.py", "*_test.py")
        matches: List[Path] = []
        for pattern in patterns:
            matches.extend(sorted(root.rglob(pattern)))
        return sorted({p.resolve() for p in matches})

    def _normalize_cmd(self, test_dir: Path) -> Optional[list]:
        if self.test_cmd:
            if isinstance(self.test_cmd, str):
                cmd = shlex.split(self.test_cmd)
                return cmd if cmd else None
            command = list(self.test_cmd)
            if command and isinstance(command[0], (list, tuple)):
                return command
            return command if command else None

        if not self._discover_pytests(test_dir):
            return None
        return ["pytest", str(test_dir)]

    def _discover_ui_tests(self, root: Path) -> List[Path]:
        if not self.ui_test_patterns:
            return []
        matches: List[Path] = []
        for path in root.rglob("*.py"):
            rel = str(path.relative_to(root))
            if any(fnmatch.fnmatch(rel, pattern) for pattern in self.ui_test_patterns):
                matches.append(path)
        return sorted({p.resolve() for p in matches})

    def _ensure_pytest(self) -> Optional[str]:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return None
        except Exception:
            pass
        try:
            install = subprocess.run(
                [sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if install.returncode == 0:
                return None
            return f"ERROR: pytest install failed.\n{install.stdout}\n{install.stderr}"
        except Exception as e:
            return f"ERROR: pytest install failed: {str(e)}"

    def run_tests(self, test_dir: str) -> str:
        path = Path(test_dir)
        if not path.exists():
            return "SKIPPED: Target directory for tests does not exist."

        install_error = self._ensure_pytest()
        if install_error:
            return install_error

        if not self.test_cmd:
            test_files = self._discover_pytests(path)
            if not test_files:
                return "SKIPPED: No tests found."
            if self.per_file:
                cmd = [[sys.executable, "-m", "pytest", "-x", str(test_file)] for test_file in test_files]
            else:
                cmd = [sys.executable, "-m", "pytest", "-x", str(path)]
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
                        cwd=path,
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
                cwd=path,
            )
            report = result.stdout + "\n" + result.stderr
            if result.returncode == 0:
                return f"SUCCESS: All tests passed.\n{report[-1000:]}"
            return f"FAIL: Tests failed.\n{report[-2000:]}"
        except Exception as e:
            return f"ERROR: Execution failed: {str(e)}"

    def run_ui_tests(self, test_dir: str) -> str:
        path = Path(test_dir)
        if not path.exists():
            return "SKIPPED: Target directory for UI tests does not exist."

        install_error = self._ensure_pytest()
        if install_error:
            return install_error

        ui_tests = self._discover_ui_tests(path)
        if not ui_tests:
            return "SKIPPED: No UI tests found."

        try:
            for test_file in ui_tests:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-x", str(test_file)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=path,
                )
                report = result.stdout + "\n" + result.stderr
                if result.returncode != 0:
                    return f"FAIL: UI tests failed.\n{report[-2000:]}"
            return "SUCCESS: UI tests passed."
        except Exception as e:
            return f"ERROR: UI test execution failed: {str(e)}"

    def run_coverage(self, test_dir: str) -> str:
        path = Path(test_dir)
        if not path.exists():
            return "SKIPPED: Target directory for coverage does not exist."

        install_error = self._ensure_pytest()
        if install_error:
            return install_error

        if not self._discover_pytests(path):
            return "SKIPPED: No tests found."

        cmd = self.coverage_cmd
        if cmd:
            if isinstance(cmd, str):
                cmd = shlex.split(cmd)
        else:
            cmd = [sys.executable, "-m", "pytest", "--cov=.", "--cov-report=term-missing"]

        try:
            result = subprocess.run(
                list(cmd),
                capture_output=True,
                text=True,
                timeout=self.coverage_timeout,
                cwd=path,
            )
            report = result.stdout + "\n" + result.stderr
            if result.returncode == 0:
                return f"SUCCESS: Coverage passed.\n{report[-2000:]}"
            return f"FAIL: Coverage failed.\n{report[-2000:]}"
        except Exception as e:
            return f"ERROR: Coverage execution failed: {str(e)}"

    def run_input_contract_tests(self, test_dir: str) -> str:
        path = Path(test_dir)
        if not path.exists():
            return "SKIPPED: Target directory for input tests does not exist."

        candidates = [
            path / "input_contract_tests.py",
            path / "input_contract.py",
            path / "tests" / "input_contract.py",
            path / "tests" / "test_input_contract.py",
        ]
        target = next((item for item in candidates if item.exists()), None)
        if not target:
            return "SKIPPED: No input contract tests found."

        try:
            result = subprocess.run(
                ["python", str(target)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=path,
            )
            report = result.stdout + "\n" + result.stderr
            if result.returncode == 0:
                return f"SUCCESS: Input contract tests passed.\n{report[-1000:]}"
            return f"FAIL: Input contract tests failed.\n{report[-2000:]}"
        except Exception as e:
            return f"ERROR: Input contract tests failed: {str(e)}"

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
                cwd=path,
            )
            report = result.stdout + "\n" + result.stderr
            if result.returncode == 0:
                return f"SUCCESS: User simulation passed.\n{report[-1000:]}"
            return f"FAIL: User simulation failed.\n{report[-2000:]}"
        except Exception as e:
            return f"ERROR: Simulation failed: {str(e)}"


class DisabledExecutor(CodeExecutor):
    def __init__(self, reason: str = "Execution disabled by config.") -> None:
        self.reason = reason

    def run_tests(self, test_dir: str) -> str:
        return f"SKIPPED: {self.reason}"

    def run_input_contract_tests(self, test_dir: str) -> str:
        return f"SKIPPED: {self.reason}"

    def run_user_simulation(self, test_dir: str) -> str:
        return f"SKIPPED: {self.reason}"

    def run_ui_tests(self, test_dir: str) -> str:
        return f"SKIPPED: {self.reason}"

    def run_coverage(self, test_dir: str) -> str:
        return f"SKIPPED: {self.reason}"
