import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

print("Testing imports...")
try:
    from agents.common import load_config, find_project_root
    print("✅ agents.common imported")
except ImportError as e:
    print(f"❌ agents.common failed: {e}")

try:
    from agents.dev_team.role_agent import RoleAgent
    print("✅ agents.dev_team.role_agent imported")
except ImportError as e:
    print(f"❌ agents.dev_team.role_agent failed: {e}")

try:
    from agents.task_planner.agent import load_config as tp_load_config
    print("✅ agents.task_planner.agent imported")
    
    # Test config loading
    cfg = tp_load_config()
    if "agent_root" in cfg:
        print("✅ Task Planner Config loaded with agent_root")
    else:
        print("❌ Task Planner Config missing agent_root")
        
except ImportError as e:
    print(f"❌ agents.task_planner.agent failed: {e}")
except Exception as e:
    print(f"❌ Verification failed: {e}")
