from typing import Any, Dict, List
from pathlib import Path
import yaml
import json
from langchain_core.runnables import RunnableLambda

from agents.task_planner.chains.analyzer_chain import build_analyzer_chain
from agents.task_planner.chains.classifier_chain import build_classifier_chain
from agents.task_planner.chains.optimizer_chain import build_optimizer_chain
from agents.task_planner.chains.validator_chain import build_validator_chain

def load_config(config_path: str = "configs/task_planner.yaml") -> Dict[str, Any]:
    # 加载基础任务规划配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 加载统一的 LLM 配置
    llm_config_path = "configs/llm.yaml"
    if Path(llm_config_path).exists():
        with open(llm_config_path, "r", encoding="utf-8") as f:
            llm_settings = yaml.safe_load(f)
            active_profile = llm_settings.get("active_profile", "grok")
            profile = llm_settings.get("profiles", {}).get(active_profile, {})
            
            # 将 LLM 配置注入到每个角色的配置中，如果角色没有指定则使用默认
            for role_name in config.get("roles", {}):
                role_cfg = config["roles"][role_name]
                if "model" not in role_cfg:
                    role_cfg["model"] = profile.get("model")
                if "api_key" not in role_cfg:
                    role_cfg["api_key"] = profile.get("api_key")
                if "api_base" not in role_cfg:
                    role_cfg["api_base"] = profile.get("api_base")
    
    return config

def run_task_planner(input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    user_input = input_data.get("user_input")
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
    final_jds = []
    validation_result = {"passed": False}
    
    while iteration < max_iterations:
        iteration += 1
        print(f">>> 第 {iteration} 轮 JD 优化与验证...")
        
        # 3. JD 优化
        optimizer = build_optimizer_chain(config)
        optimized_jds = []
        
        for role in roles:
            # 如果是第一轮，使用初始JD；否则（理论上可以基于反馈微调，这里简化处理，还是基于初始JD优化）
            # 在更复杂的实现中，可以将上一轮的反馈作为输入传给 optimizer
            
            jd_input = {
                "role_name": role["role_name"],
                "initial_jd": role["initial_jd"],
                "requirements": json.dumps(requirements, ensure_ascii=False),
                "tasks": json.dumps(tasks, ensure_ascii=False)
            }
            opt_jd = optimizer.invoke(jd_input)
            optimized_jds.append(opt_jd)
        
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
