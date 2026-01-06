from __future__ import annotations

from pathlib import Path

from agents.dev_team.app.state_machine import MilestoneStateMachine
from agents.dev_team.app.workflow import WorkflowRouter
from agents.dev_team.capabilities import CapabilityRegistry, CAP_RESULT_VERIFIABLE
from agents.dev_team.gates.quality_gate import QualityGate
from agents.dev_team.domain.models import Evidence, GateRule
from agents.dev_team.policies.compliance_policy import CompliancePolicyImpl
from agents.dev_team.policies.ui_baseline_policy import UIBaselinePolicy
from agents.dev_team.services.execution_service import ExecutionService
from agents.dev_team.services.verification_service import VerificationService
from agents.dev_team.services.worker_pool import WorkerPool
from agents.dev_team.memory import SharedMemoryStore


class DummyExecutor:
    def run_tests(self, _output_dir: str) -> str:
        return "SUCCESS: tests"

    def run_ui_tests(self, _output_dir: str) -> str:
        return "FAIL: ui tests"

    def run_coverage(self, _output_dir: str) -> str:
        return "SUCCESS: coverage"

    def run_input_contract_tests(self, _output_dir: str) -> str:
        return "SUCCESS: input contract"

    def run_user_simulation(self, _output_dir: str) -> str:
        return "SKIPPED: no simulation"


class DummyAgent:
    role_name = "QA Engineer"
    role_type = "QA"

    def run(self) -> str:
        return "ok"


def test_state_machine_transitions():
    fsm = MilestoneStateMachine()
    assert fsm.transition("start_round").to_state == "executing"
    assert fsm.transition("agents_done").to_state == "verifying"
    assert fsm.transition("tests_done").to_state == "gating"
    assert fsm.transition("gating").to_state == "gating"
    assert fsm.transition("gate_failed").to_state == "failed"
    assert fsm.transition("retry").to_state == "executing"


def test_workflow_router_phases():
    router = WorkflowRouter()
    phases = router.phases(enable_consensus=True, enable_cross_validation=True, has_qa=True)
    assert phases == ["agents", "consensus", "tests", "cross_validation", "qa", "gate"]


def test_quality_gate_evaluate():
    rules = [GateRule("tests", True), GateRule("coverage", True)]
    gate = QualityGate(rules)
    evidence = [
        Evidence(kind="tests", status="passed"),
        Evidence(kind="coverage", status="failed"),
    ]
    decision = gate.evaluate(evidence)
    assert decision.status == "failed"
    assert "coverage" in decision.reasons


def test_verification_service_run_round(tmp_path: Path):
    service = VerificationService(DummyExecutor())
    result = service.run_round(str(tmp_path), ui_required=True, testing_enabled=True)
    assert result["tests"]["status"] == "passed"
    assert result["ui_tests"]["status"] == "failed"
    assert result["coverage"]["status"] == "passed"
    assert result["input_contract"]["status"] == "passed"
    assert result["user_simulation"]["status"] == "skipped"


def test_execution_service_run_agents():
    service = ExecutionService(lambda agent: {"role": agent.role_name})
    results = service.run_agents([DummyAgent(), DummyAgent()])
    assert len(results) == 2
    assert results[0]["role"] == "QA Engineer"


def test_worker_pool_registers_capabilities():
    registry = CapabilityRegistry()
    pool = WorkerPool(registry)
    agent = DummyAgent()
    pool.add_agent(agent, agent.role_name, is_qa=True, is_final_approver=False)
    resolved = registry.resolve(CAP_RESULT_VERIFIABLE)
    assert resolved and resolved[0] is agent


def test_compliance_policy_detects_secret(tmp_path: Path):
    secret_file = tmp_path / "secrets.txt"
    secret_file.write_text("sk-1234567890ABCDEFGHIJKLMNOP", encoding="utf-8")
    policy = CompliancePolicyImpl(
        config={"compliance": {"enabled": True}},
        output_dir=tmp_path,
        memory=SharedMemoryStore({}),
    )
    report = policy.run()
    assert report["status"] == "failed"
    assert report["findings"]


def test_ui_baseline_check_evidence_without_baseline(tmp_path: Path):
    output_dir = tmp_path / "output"
    evidence_dir = output_dir / "evidence" / "ui"
    docs_dir = output_dir / "evidence" / "docs"
    evidence_dir.mkdir(parents=True)
    docs_dir.mkdir(parents=True)
    (evidence_dir / "implementation.png").write_bytes(b"fake")
    policy = UIBaselinePolicy(
        config={"ui_design": {"summary_required": True}},
        output_dir=output_dir,
        memory=SharedMemoryStore({}),
    )
    report = policy.check_ui_evidence()
    assert report["status"] == "passed"
    assert "design_baseline_missing" in report["warnings"]
