from typing import Any, Dict, List
from pathlib import Path
import yaml
import json
from langchain_core.runnables import RunnableLambda

from agents.task_planner.chains.analyzer_chain import build_analyzer_chain
from agents.task_planner.chains.classifier_chain import build_classifier_chain
from agents.task_planner.chains.optimizer_chain import build_optimizer_chain
from agents.task_planner.chains.validator_chain import build_validator_chain


from agents.common import load_config as common_load_config

def load_config(config_path: str = None) -> Dict[str, Any]:
    # Wrapper to maintain compatible signature but use common logic with correct base
    base_dir = Path(__file__).parent.resolve()
    config = common_load_config(config_path, base_dir)
    
    # Inject agent_root for relative path resolution in chains if not present
    if "agent_root" not in config:
        config["agent_root"] = str(base_dir)
        
    return config

def run_task_planner(input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    user_input = input_data.get("user_input")
    execution_feedback = input_data.get("execution_feedback")
    
    if execution_feedback:
        user_input += f"\n\n[Pre-existing Execution Feedback]: {execution_feedback}"

    max_iterations = config.get("workflow", {}).get("max_iterations", 3)
    
    # 1. 需求分析
    print(">>> 正在进行需求分析...")
    analyzer = build_analyzer_chain(config)
    requirements = analyzer.invoke({"user_input": user_input})
    print(f"    核心目标: {requirements.get('goal')}")
    
    # 2. 任务分类与角色拆解
    print(">>> 正在进行任务拆解与角色分配...")
    classifier = build_classifier_chain(config)
    classification = classifier.invoke({"requirements": json.dumps(requirements, ensure_ascii=False)})
    
    tasks = classification.get("tasks", [])
    roles = classification.get("roles", [])
    print(f"    拆解出 {len(tasks)} 个任务, {len(roles)} 个角色")
    
    # 迭代优化流程
    iteration = 0
    # Initialize current JDs with initial JDs
    current_role_jds = {role["role_name"]: role["initial_jd"] for role in roles}
    
    validation_result = {"passed": False}
    # 初始化反馈 (Global)
    global_feedback = "无 (第一轮)"
    role_specific_feedbacks = {}
    
    while iteration < max_iterations:
        iteration += 1
        print(f">>> 第 {iteration} 轮 JD 优化与验证...")
        
        # 3. JD 优化
        optimizer = build_optimizer_chain(config)
        optimized_jds = []
        
        # Use a dictionary to store optimized JDs for the next iteration
        next_round_jds = {}

        for role in roles:
            role_name = role["role_name"]
            # INPUT for optimizer: Use the JD from the CURRENT state (updated in previous loop), not the initial one
            current_jd_content = current_role_jds.get(role_name, "")
            
            # Construct personalized feedback
            # Feedback = Global Feedback + Specific Feedback (if any)
            current_feedback = f"Overall: {global_feedback}"
            if role_name in role_specific_feedbacks:
                current_feedback += f"\nSpecific Advice for {role_name}: {role_specific_feedbacks[role_name]}"
            
            jd_input = {
                "role_name": role_name,
                "initial_jd": current_jd_content, 
                "requirements": json.dumps(requirements, ensure_ascii=False),
                "tasks": json.dumps(tasks, ensure_ascii=False),
                "feedback": current_feedback
            }
            try:
                opt_jd = optimizer.invoke(jd_input)
                optimized_jds.append(opt_jd)
                
                next_round_jds[role_name] = json.dumps(opt_jd, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"    ❌ 优化角色 {role_name} 失败: {e}")
                # Keep previous JD if failed
                optimized_jds.append({"role_name": role_name, "error": str(e)}) # Placeholder
                next_round_jds[role_name] = current_role_jds[role_name]

        current_role_jds = next_round_jds
        final_jds = optimized_jds
        
        # 4. JD 验证
        print("    正在验证 JD...")
        validator = build_validator_chain(config)
        validation_input = {
            "requirements": json.dumps(requirements, ensure_ascii=False),
            "final_jds": json.dumps(final_jds, ensure_ascii=False),
            "tasks": json.dumps(tasks, ensure_ascii=False)
        }
        validation_result = validator.invoke(validation_input)
        
        if validation_result.get("passed", False):
            print("    >>> 验证通过！")
            break
        else:
            print(f"    >>> 验证未通过 (分数: {validation_result.get('score')})")
            print(f"    反馈: {validation_result.get('overall_feedback')}")
            
            # 更新反馈给下一轮
            global_feedback = validation_result.get('overall_feedback', '无具体反馈')
            role_specific_feedbacks = validation_result.get('role_specific_feedback', {})
            
            # 如果存在严重缺失，且有需要用户反馈的问题，则中断流程返回给用户
            if validation_result.get("user_feedback_needed"):
                print("    !!! 需要用户补充信息 !!!")
                return {
                    "status": "needs_feedback",
                    "requirements": requirements,
                    "validation_result": validation_result,
                    "current_jds": final_jds
                }

    
    return {
        "status": "completed",
        "requirements": requirements,
        "tasks": tasks,
        "final_jds": final_jds,
        "validation_result": validation_result
    }

def build_agent():
    config = load_config()
    return RunnableLambda(lambda x: run_task_planner(x, config))

if __name__ == "__main__":
    import sys
    
    user_input = "我想做一个简单的待办事项管理CLI工具，用Python写，数据存本地文件就行。"
    if len(sys.argv) > 1:
        user_input = sys.argv[1]
        
    agent = build_agent()
    result = agent.invoke({"user_input": user_input})
    print("\n========== 最终结果 ==========\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
