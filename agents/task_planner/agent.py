from typing import Any, Dict, List, Tuple
from pathlib import Path
import yaml
import json
from langchain_core.runnables import RunnableLambda

from agents.task_planner.chains.analyzer_chain import build_analyzer_chain
from agents.task_planner.chains.classifier_chain import build_classifier_chain
from agents.task_planner.chains.optimizer_chain import build_optimizer_chain
from agents.task_planner.chains.validator_chain import build_validator_chain


from agents.common import load_config as common_load_config


def _coerce_json(payload: Any) -> Any:
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return payload
    return payload


def _require_dict(name: str, payload: Any) -> Dict[str, Any]:
    payload = _coerce_json(payload)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} output must be a dict, got {type(payload).__name__}")
    return payload


def _validate_requirements(payload: Any) -> Dict[str, Any]:
    data = _require_dict("Requirement analysis", payload)
    goal = data.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("Requirement analysis must include non-empty 'goal' field.")
    return data


def _validate_classification(payload: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    data = _require_dict("Task classification", payload)
    tasks = data.get("tasks", [])
    roles = data.get("roles", [])
    if not isinstance(tasks, list) or not isinstance(roles, list):
        raise ValueError("Task classification must include list fields 'tasks' and 'roles'.")
    for role in roles:
        if not isinstance(role, dict) or not role.get("role_name") or not role.get("initial_jd"):
            raise ValueError("Each role must include 'role_name' and 'initial_jd'.")
    return tasks, roles


def _validate_optimizer_output(role_name: str, payload: Any) -> Dict[str, Any]:
    data = _require_dict(f"JD optimizer ({role_name})", payload)
    if "role_name" not in data:
        data["role_name"] = role_name
    return data


def _validate_validator_output(payload: Any) -> Dict[str, Any]:
    data = _require_dict("JD validator", payload)
    passed = data.get("passed")
    score = data.get("score")
    if passed is None and score is None:
        raise ValueError("JD validator must include 'passed' or 'score'.")
    return data

def load_config(config_path: str = None) -> Dict[str, Any]:
    # Wrapper to maintain compatible signature but use common logic with correct base
    base_dir = Path(__file__).parent.resolve()
    config = common_load_config(config_path, base_dir)
    
    # Inject agent_root for relative path resolution in chains if not present
    if "agent_root" not in config:
        config["agent_root"] = str(base_dir)
        
    return config

def run_task_planner(input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    user_input = input_data.get("user_input") or ""
    execution_feedback = input_data.get("execution_feedback")
    planner_state = input_data.get("planner_state")
    user_feedback = input_data.get("user_feedback")
    
    if execution_feedback:
        user_input += f"\n\n[Pre-existing Execution Feedback]: {execution_feedback}"

    max_iterations = config.get("workflow", {}).get("max_iterations", 3)
    validation_threshold = config.get("workflow", {}).get("validation_threshold", 0.8)
    
    if planner_state:
        try:
            requirements = _require_dict("Planner state requirements", planner_state.get("requirements"))
            tasks = planner_state.get("tasks", [])
            roles = planner_state.get("roles", [])
            current_role_jds = planner_state.get("current_jds", {})
            if not isinstance(tasks, list) or not isinstance(roles, list) or not isinstance(current_role_jds, dict):
                raise ValueError("Planner state must include tasks(list), roles(list), current_jds(dict).")
            for role in roles:
                if not isinstance(role, dict) or not role.get("role_name"):
                    raise ValueError("Planner state roles must include 'role_name'.")
                if "initial_jd" not in role and role.get("role_name") not in current_role_jds:
                    raise ValueError("Planner state role missing 'initial_jd' and no current_jds entry.")
        except Exception as e:
            return {
                "status": "error",
                "error": f"Planner state invalid: {e}",
            }
        print(">>> 已检测到 planner_state，跳过需求分析与任务拆解，直接继续优化/验证...")
    else:
        # 1. 需求分析
        print(">>> 正在进行需求分析...")
        analyzer = build_analyzer_chain(config)
        try:
            requirements = _validate_requirements(analyzer.invoke({"user_input": user_input}))
        except Exception as e:
            return {
                "status": "error",
                "error": f"Requirement analysis failed: {e}"
            }
        print(f"    核心目标: {requirements.get('goal')}")

        # 2. 任务分类与角色拆解
        print(">>> 正在进行任务拆解与角色分配...")
        classifier = build_classifier_chain(config)
        try:
            classification = classifier.invoke({"requirements": json.dumps(requirements, ensure_ascii=False)})
            tasks, roles = _validate_classification(classification)
        except Exception as e:
            return {
                "status": "error",
                "error": f"Task classification failed: {e}"
            }
        print(f"    拆解出 {len(tasks)} 个任务, {len(roles)} 个角色")
    
    # 迭代优化流程
    iteration = int(planner_state.get("iteration", 0)) if planner_state else 0
    if planner_state and user_feedback:
        max_iterations = max(max_iterations, iteration + 1)
    # Initialize current JDs with initial JDs
    if not planner_state:
        current_role_jds = {role["role_name"]: role["initial_jd"] for role in roles}
    
    validation_result = {"passed": False}
    # 初始化反馈 (Global)
    global_feedback = "无 (第一轮)"
    if user_feedback:
        global_feedback = f"用户补充: {user_feedback}"
    role_specific_feedbacks = {}
    
    final_jds = []
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
                opt_jd = _validate_optimizer_output(role_name, optimizer.invoke(jd_input))
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
        try:
            validation_result = _validate_validator_output(validator.invoke(validation_input))
        except Exception as e:
            return {
                "status": "error",
                "error": f"JD validation failed: {e}"
            }

        score = validation_result.get("score")
        is_passed = validation_result.get("passed", False)
        if is_passed or (score is not None and score >= validation_threshold):
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
                    "tasks": tasks,
                    "roles": roles,
                    "validation_result": validation_result,
                    "current_jds": final_jds,
                    "planner_state": {
                        "requirements": requirements,
                        "tasks": tasks,
                        "roles": roles,
                        "current_jds": current_role_jds,
                        "iteration": iteration,
                    },
                }

    
    if not final_jds:
        for role in roles:
            role_name = role.get("role_name", "Unknown")
            jd_content = current_role_jds.get(role_name) if planner_state else role.get("initial_jd", "")
            jd_payload = _coerce_json(jd_content)
            if isinstance(jd_payload, dict):
                final_jds.append(jd_payload)
            else:
                final_jds.append({"role_name": role_name, "content": jd_content})

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
