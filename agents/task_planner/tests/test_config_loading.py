import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

from agents.task_planner.agent import load_config

def test_planner_config():
    print(">>> Testing Task Planner Config Loading...")
    try:
        config = load_config()
        print(f"Config loaded successfully. Keys: {list(config.keys())}")
        if "agent_root" in config:
            print(f"agent_root: {config['agent_root']}")
            assert Path(config['agent_root']).is_absolute()
        assert "roles" in config
        print("✅ Task Planner Config Loading Verified!")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_planner_config()
