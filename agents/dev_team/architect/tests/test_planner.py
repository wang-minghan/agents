import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from agents.dev_team.architect.agent import run_architect

class TestTaskPlanner(unittest.TestCase):
    
    @patch('agents.dev_team.architect.agent.build_analyzer_chain')
    @patch('agents.dev_team.architect.agent.build_classifier_chain')
    @patch('agents.dev_team.architect.agent.build_optimizer_chain')
    @patch('agents.dev_team.architect.agent.build_validator_chain')
    def test_successful_flow(self, mock_validator, mock_optimizer, mock_classifier, mock_analyzer):
        # Mock Config
        config = {
            "workflow": {"max_iterations": 2, "snapshot_enabled": False},
            "roles": {}
        }
        
        # Mock Analyzer Output
        mock_analyzer.return_value.invoke.return_value = {
            "goal": "Test Goal",
            "key_features": ["F1"],
            "constraints": ["C1"],
            "priority": "High"
        }
        
        # Mock Classifier Output
        mock_classifier.return_value.invoke.return_value = {
            "tasks": [{"id": "T1", "name": "Task 1"}],
            "roles": [{"role_name": "Dev", "initial_jd": "Code it"}]
        }
        
        # Mock Optimizer Output
        mock_optimizer.return_value.invoke.return_value = {
            "role_name": "Dev",
            "required_skills": ["Python"]
        }
        
        # Mock Validator Output (Passed)
        mock_validator.return_value.invoke.return_value = {
            "passed": True,
            "score": 0.9,
            "overall_feedback": "Looks good"
        }
        
        input_data = {"user_input": "Do something"}
        result = run_architect(input_data, config)
        
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["tasks"]), 1)
        self.assertEqual(len(result["final_jds"]), 1)

    @patch('agents.dev_team.architect.agent.build_analyzer_chain')
    @patch('agents.dev_team.architect.agent.build_classifier_chain')
    @patch('agents.dev_team.architect.agent.build_optimizer_chain')
    @patch('agents.dev_team.architect.agent.build_validator_chain')
    def test_needs_feedback(self, mock_validator, mock_optimizer, mock_classifier, mock_analyzer):
        config = {"workflow": {"max_iterations": 1, "snapshot_enabled": False}}
        
        # Setup basic mocks to proceed to validation
        mock_analyzer.return_value.invoke.return_value = {"goal": "G"}
        mock_classifier.return_value.invoke.return_value = {"tasks": [], "roles": [{"role_name": "R", "initial_jd": "J"}]}
        mock_optimizer.return_value.invoke.return_value = {}
        
        # Mock Validator (Failed with feedback needed)
        mock_validator.return_value.invoke.return_value = {
            "passed": False,
            "score": 0.5,
            "user_feedback_needed": ["Question 1?"],
            "overall_feedback": "Bad"
        }
        
        result = run_architect({"user_input": "in"}, config)
        
        self.assertEqual(result["status"], "needs_feedback")
        self.assertIn("Question 1?", result["validation_result"]["user_feedback_needed"])

    @patch('agents.dev_team.architect.agent.build_analyzer_chain')
    @patch('agents.dev_team.architect.agent.build_classifier_chain')
    def test_invalid_classification(self, mock_classifier, mock_analyzer):
        config = {"workflow": {"max_iterations": 1, "snapshot_enabled": False}}
        mock_analyzer.return_value.invoke.return_value = {"goal": "G"}
        mock_classifier.return_value.invoke.return_value = {"tasks": "bad", "roles": []}

        result = run_architect({"user_input": "in"}, config)

        self.assertEqual(result["status"], "error")
        self.assertIn("Task classification failed", result["error"])

    @patch('agents.dev_team.architect.agent.build_analyzer_chain')
    @patch('agents.dev_team.architect.agent.build_classifier_chain')
    @patch('agents.dev_team.architect.agent.build_optimizer_chain')
    @patch('agents.dev_team.architect.agent.build_validator_chain')
    def test_stringified_analyzer_output(self, mock_validator, mock_optimizer, mock_classifier, mock_analyzer):
        config = {"workflow": {"max_iterations": 1, "snapshot_enabled": False}}
        mock_analyzer.return_value.invoke.return_value = '{"goal": "Test Goal"}'
        mock_classifier.return_value.invoke.return_value = {
            "tasks": [{"id": "T1", "name": "Task 1"}],
            "roles": [{"role_name": "Dev", "initial_jd": "Code it"}],
        }
        mock_optimizer.return_value.invoke.return_value = {
            "role_name": "Dev",
            "required_skills": ["Python"],
        }
        mock_validator.return_value.invoke.return_value = {
            "passed": True,
            "score": 0.9,
        }

        result = run_architect({"user_input": "in"}, config)

        self.assertEqual(result["status"], "completed")

    @patch('agents.dev_team.architect.agent.build_optimizer_chain')
    @patch('agents.dev_team.architect.agent.build_validator_chain')
    def test_resume_from_feedback(self, mock_validator, mock_optimizer):
        config = {"workflow": {"max_iterations": 1, "snapshot_enabled": False}}

        mock_optimizer.return_value.invoke.return_value = {"role_name": "Dev"}
        mock_validator.return_value.invoke.return_value = {"passed": True, "score": 0.9}

        planner_state = {
            "requirements": {"goal": "G"},
            "tasks": [{"id": "T1", "name": "Task 1"}],
            "roles": [{"role_name": "Dev", "initial_jd": "Do it"}],
            "current_jds": {"Dev": "Do it"},
            "iteration": 1,
        }

        result = run_architect(
            {"user_input": "in", "planner_state": planner_state, "user_feedback": "补充信息"},
            config,
        )

        self.assertEqual(result["status"], "completed")

    @patch('agents.dev_team.architect.agent.build_analyzer_chain')
    @patch('agents.dev_team.architect.agent.build_classifier_chain')
    @patch('agents.dev_team.architect.agent.build_optimizer_chain')
    @patch('agents.dev_team.architect.agent.build_validator_chain')
    def test_snapshot_written(self, mock_validator, mock_optimizer, mock_classifier, mock_analyzer):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "workflow": {
                    "max_iterations": 1,
                    "snapshot_enabled": True,
                    "snapshot_dir": tmpdir,
                },
                "roles": {},
                "agent_root": "/home/minghan/project/agents/agents/dev_team/architect",
            }

            mock_analyzer.return_value.invoke.return_value = {"goal": "G"}
            mock_classifier.return_value.invoke.return_value = {
                "tasks": [{"id": "T1", "name": "Task 1"}],
                "roles": [{"role_name": "Dev", "initial_jd": "Code it"}],
            }
            mock_optimizer.return_value.invoke.return_value = {"role_name": "Dev"}
            mock_validator.return_value.invoke.return_value = {"passed": True, "score": 0.9}

            result = run_architect({"user_input": "in"}, config)

            snapshot_path = result.get("snapshot_path")
            self.assertIsNotNone(snapshot_path)
            self.assertTrue(Path(snapshot_path).exists())

    @patch('agents.dev_team.architect.agent.build_analyzer_chain')
    @patch('agents.dev_team.architect.agent.build_classifier_chain')
    @patch('agents.dev_team.architect.agent.build_optimizer_chain')
    @patch('agents.dev_team.architect.agent.build_validator_chain')
    def test_constraints_injected(self, mock_validator, mock_optimizer, mock_classifier, mock_analyzer):
        config = {"workflow": {"max_iterations": 1, "snapshot_enabled": False}, "roles": {}}

        mock_analyzer.return_value.invoke.return_value = {"goal": "G"}
        mock_classifier.return_value.invoke.return_value = {
            "tasks": [{"id": "T1", "name": "Task 1"}],
            "roles": [{"role_name": "Dev", "initial_jd": "Code it"}],
        }
        mock_optimizer.return_value.invoke.return_value = {"role_name": "Dev"}
        mock_validator.return_value.invoke.return_value = {"passed": True, "score": 0.9}

        run_architect({"user_input": "in", "constraints": {"stack": "python"}}, config)

        call_args = mock_analyzer.return_value.invoke.call_args[0][0]
        self.assertIn("[Constraints]", call_args["user_input"])

if __name__ == '__main__':
    unittest.main()
