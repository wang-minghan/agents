from __future__ import annotations

from typing import Any, Dict


class VerificationService:
    def __init__(self, executor: Any):
        self.executor = executor

    @staticmethod
    def _classify(output: str) -> str:
        if not output:
            return "unknown"
        normalized = output.strip().upper()
        if normalized.startswith("SUCCESS"):
            return "passed"
        if normalized.startswith("FAIL"):
            return "failed"
        if normalized.startswith("ERROR"):
            return "error"
        if normalized.startswith("SKIPPED"):
            return "skipped"
        if "FAILED" in normalized or "FAIL" in normalized:
            return "failed"
        if "ERROR" in normalized:
            return "error"
        if "SKIPPED" in normalized:
            return "skipped"
        return "unknown"

    @staticmethod
    def _summary(output: str) -> str:
        if not output:
            return "No output"
        lines = output.splitlines()
        return lines[0] if lines else "No output"

    def run_tests(self, output_dir: str, enabled: bool) -> Dict[str, str]:
        if not enabled:
            output = "SKIPPED: Testing disabled by config."
        else:
            output = self.executor.run_tests(output_dir)
        return {"status": self._classify(output), "summary": self._summary(output), "output": output}

    def run_ui_tests(self, output_dir: str, enabled: bool) -> Dict[str, str]:
        if not enabled or not hasattr(self.executor, "run_ui_tests"):
            output = "SKIPPED: No UI tests."
        else:
            output = self.executor.run_ui_tests(output_dir)
        return {"status": self._classify(output), "summary": self._summary(output), "output": output}

    def run_coverage(self, output_dir: str) -> Dict[str, str]:
        if not hasattr(self.executor, "run_coverage"):
            output = "SKIPPED: No coverage run."
        else:
            output = self.executor.run_coverage(output_dir)
        return {"status": self._classify(output), "summary": self._summary(output), "output": output}

    def run_input_contract_tests(self, output_dir: str) -> Dict[str, str]:
        if not hasattr(self.executor, "run_input_contract_tests"):
            output = "SKIPPED: No input contract tests."
        else:
            output = self.executor.run_input_contract_tests(output_dir)
        return {"status": self._classify(output), "summary": self._summary(output), "output": output}

    def run_user_simulation(self, output_dir: str, enabled: bool) -> Dict[str, str]:
        if not enabled or not hasattr(self.executor, "run_user_simulation"):
            output = "SKIPPED: No user simulation."
        else:
            output = self.executor.run_user_simulation(output_dir)
        return {"status": self._classify(output), "summary": self._summary(output), "output": output}

    def run_round(self, output_dir: str, *, ui_required: bool, testing_enabled: bool) -> Dict[str, Dict[str, str]]:
        return {
            "tests": self.run_tests(output_dir, testing_enabled),
            "ui_tests": self.run_ui_tests(output_dir, ui_required),
            "coverage": self.run_coverage(output_dir),
            "input_contract": self.run_input_contract_tests(output_dir),
            "user_simulation": self.run_user_simulation(output_dir, ui_required),
        }
