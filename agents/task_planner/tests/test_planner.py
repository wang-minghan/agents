import unittest
from unittest.mock import MagicMock, patch
from agents.task_planner.agent import run_task_planner

class TestTaskPlanner(unittest.TestCase):
    
    @patch('agents.task_planner.agent.build_analyzer_chain')
    @patch('agents.task_planner.agent.build_classifier_chain')
    @patch('agents.task_planner.agent.build_optimizer_chain')
    @patch('agents.task_planner.agent.build_validator_chain')
    def test_successful_flow(self, mock_validator, mock_optimizer, mock_classifier, mock_analyzer):
        # Mock Config
        config = {
            "workflow": {"max_iterations": 2},
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
        result = run_task_planner(input_data, config)
        
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["tasks"]), 1)
        self.assertEqual(len(result["final_jds"]), 1)

    @patch('agents.task_planner.agent.build_analyzer_chain')
    @patch('agents.task_planner.agent.build_classifier_chain')
    @patch('agents.task_planner.agent.build_optimizer_chain')
    @patch('agents.task_planner.agent.build_validator_chain')
    def test_needs_feedback(self, mock_validator, mock_optimizer, mock_classifier, mock_analyzer):
        config = {"workflow": {"max_iterations": 1}}
        
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
        
        result = run_task_planner({"user_input": "in"}, config)
        
        self.assertEqual(result["status"], "needs_feedback")
        self.assertIn("Question 1?", result["validation_result"]["user_feedback_needed"])

if __name__ == '__main__':
    unittest.main()
