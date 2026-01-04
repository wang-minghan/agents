from pathlib import Path

from agents.dev_team.commander import Commander
from agents.dev_team.execution import LocalUnsafeExecutor
from agents.dev_team.interfaces import Agent, CodeExecutor
from agents.common import save_files_from_content
from agents.dev_team.utils import find_project_root


def _write_prompt(tmp_path: Path, name: str) -> str:
    prompt_path = tmp_path / f"{name}.txt"
    prompt_path.write_text(f"mock {name} prompt", encoding="utf-8")
    return str(prompt_path)


def test_architecture_and_safety(tmp_path: Path):
    test_output_dir = tmp_path / "output"
    test_output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "output_dir": str(test_output_dir),
        "llm": {"model": "mock", "api_key": "mock"},
        "roles": {
            "engineer": {"prompt_path": _write_prompt(tmp_path, "engineer")},
            "qa": {"prompt_path": _write_prompt(tmp_path, "qa")},
        },
    }

    # 1. Default Execution (LocalUnsafeExecutor)
    orch = Commander(config, output_dir=str(test_output_dir))
    assert isinstance(orch.code_executor, LocalUnsafeExecutor)

    # 2. Dependency Injection (AgentFactory)

    class MockAgent:
        def __init__(self, jd, cfg, mem, out):
            self.role_name = jd.get("role_name")
            self.role_type = "ENGINEER"
            self.output_dir = out
            self._memory = mem

        def run(self) -> str:
            content = '<file path="safe.py">content</file>'
            self._memory.add_output(self.role_name, content)
            return content

    def mock_factory(jd, cfg, mem, out) -> Agent:
        return MockAgent(jd, cfg, mem, out)

    orch_di = Commander(config, output_dir=str(test_output_dir), agent_factory=mock_factory)
    planner_result = {
        "final_jds": [{"role_name": "MockEngineer", "role_type": "ENGINEER"}],
        "requirements": {},
    }
    orch_di.initialize_team(planner_result)
    assert len(orch_di.agents) == 1
    assert isinstance(orch_di.agents[0], MockAgent)

    # 3. Path Safety in save_files_from_content (Regression Test)
    safe_files = save_files_from_content('<file path="safe.py">content</file>', test_output_dir)
    assert len(safe_files) == 1
    assert (test_output_dir / "safe.py").exists()

    abs_path = test_output_dir / "abs.txt"
    abs_files = save_files_from_content(f'<file path="{abs_path}">content</file>', test_output_dir)
    assert len(abs_files) == 1
    assert abs_path.exists()

    unsafe_files = save_files_from_content('<file path="../unsafe.py">content</file>', test_output_dir)
    assert unsafe_files == []
    assert not (test_output_dir / "../unsafe.py").exists()

    # 4. Robust Path Finding
    fake_root = test_output_dir / "fake_project"
    (fake_root / "configs").mkdir(parents=True, exist_ok=True)
    (fake_root / "configs" / "llm.yaml").touch()

    deep_path = fake_root / "a" / "b" / "c"
    deep_path.mkdir(parents=True, exist_ok=True)

    found = find_project_root(deep_path)
    assert found.resolve() == fake_root.resolve()

    # 5. Commander Early Exit
    class MockPassExecutor(CodeExecutor):
        def run_tests(self, test_dir: str) -> str:
            return "SUCCESS: All tests passed."

    class MockMemoryAgent(MockAgent):
        def run(self) -> str:
            content = '<file path="a.txt">ok</file>'
            self._memory.add_output(self.role_name, content)
            return content

    orch_early = Commander(config, output_dir=str(test_output_dir), code_executor=MockPassExecutor())
    orch_early.agents = [MockMemoryAgent({"role_name": "Dev"}, config, orch_early.shared_memory, test_output_dir)]
    result = orch_early.run_collaboration(max_rounds=5)
    assert any(path.endswith("a.txt") for path in orch_early.shared_memory.saved_files)
    assert result["status"] == "passed"


def test_local_executor_skips_without_tests(tmp_path: Path):
    executor = LocalUnsafeExecutor()
    result = executor.run_tests(str(tmp_path))
    assert "SKIPPED: No tests found." in result


def test_user_simulation_skips_when_missing(tmp_path: Path):
    executor = LocalUnsafeExecutor()
    result = executor.run_user_simulation(str(tmp_path))
    assert "SKIPPED: No user simulation script found." in result
