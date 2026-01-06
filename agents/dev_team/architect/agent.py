from typing import Any, Dict, List, Tuple, Optional, Callable
import os
import ast
from pathlib import Path
from datetime import datetime, timezone
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from agents.dev_team.architect.chains.analyzer_chain import build_analyzer_chain
from agents.dev_team.architect.chains.classifier_chain import build_classifier_chain
from agents.dev_team.architect.chains.optimizer_chain import build_optimizer_chain
from agents.dev_team.architect.chains.validator_chain import build_validator_chain


from agents.common import load_config as common_load_config


def suggest_default_assumptions(
    config: Dict[str, Any],
    planner_result: Dict[str, Any],
    user_input: str,
    constraints: Optional[Dict[str, Any]] = None,
) -> str:
    llm_cfg = config.get("llm", {})
    role_cfg = config.get("roles", {}).get("requirement_analyzer", {})
    model = role_cfg.get("model") or llm_cfg.get("model", "gpt-4")
    api_key = role_cfg.get("api_key") or llm_cfg.get("api_key")
    api_base = role_cfg.get("api_base") or llm_cfg.get("api_base")
    if not api_key:
        return (
            "默认假设:\n"
            "- 默认提供可用的前端界面与基础交互流程\n"
            "- 核心流程需具备自动化测试与覆盖率报告\n"
            "- 默认目标为可商用交付，需完整文档与验收清单\n"
            "- 默认性能满足中等并发与常规稳定性要求\n"
        )

    llm = ChatOpenAI(
        model=model,
        temperature=0.2,
        api_key=api_key,
        base_url=api_base,
        max_retries=1,
        timeout=60,
    )
    questions = planner_result.get("validation_result", {}).get("user_feedback_needed", [])
    prompt = (
        "你是资深架构师。给出最小可行的默认假设，用于在缺少用户补充时继续推进规划。\n"
        "要求：覆盖性能目标、UI范围、目标用户、角色补齐/交付边界、测试与覆盖率要求，使用中文条目列表。\n"
        "只输出条目列表，不要额外解释。\n\n"
        f"用户需求: {user_input}\n"
        f"待澄清问题: {questions}\n"
        f"约束: {constraints or {}}\n"
    )
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else response
    content = str(content).strip()
    return "默认假设:\n" + content

def _invoke_with_heartbeat(label: str, func: Callable[[], Any], interval: int = 10) -> Any:
    stop = threading.Event()

    def _beat() -> None:
        while not stop.wait(interval):
            print(f">>> {label} 仍在分析...", flush=True)

    worker = threading.Thread(target=_beat, daemon=True)
    worker.start()
    try:
        return func()
    finally:
        stop.set()
        worker.join(timeout=1)


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    cleaned = cleaned.strip("`")
    newline_idx = cleaned.find("\n")
    if newline_idx != -1:
        cleaned = cleaned[newline_idx + 1 :]
    return cleaned.strip()

def _sanitize_json_text(text: str) -> str:
    replaced = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", '"')
        .replace("’", '"')
    )
    return replaced

def _coerce_json(payload: Any) -> Any:
    if isinstance(payload, str):
        cleaned = _strip_code_fence(payload)
        cleaned = _sanitize_json_text(cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = cleaned[start : end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    return payload
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

def _format_constraints(constraints: Any) -> Optional[str]:
    if constraints is None:
        return None
    payload = _coerce_json(constraints)
    if isinstance(payload, (dict, list)):
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return str(payload)

def _build_function_inventory(root: Path, max_files: int = 120, max_items: int = 200) -> Dict[str, Any]:
    skip_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", "output", "data", "evidence"}
    functions: List[Dict[str, Any]] = []
    files_scanned = 0
    root = root.resolve()

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in skip_dirs]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            files_scanned += 1
            if files_scanned > max_files:
                break
            path = Path(dirpath) / filename
            rel = str(path.relative_to(root))
            try:
                source = path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source)
            except Exception:
                continue
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append(
                        {
                            "file": rel,
                            "name": node.name,
                            "lineno": getattr(node, "lineno", None),
                            "end_lineno": getattr(node, "end_lineno", None),
                        }
                    )
                elif isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            functions.append(
                                {
                                    "file": rel,
                                    "name": f"{node.name}.{item.name}",
                                    "lineno": getattr(item, "lineno", None),
                                    "end_lineno": getattr(item, "end_lineno", None),
                                }
                            )
            if len(functions) >= max_items:
                break
        if files_scanned > max_files or len(functions) >= max_items:
            break

    return {
        "root": str(root),
        "files_scanned": files_scanned,
        "functions": functions[:max_items],
    }

def _inject_function_plan(requirements: Dict[str, Any], constraints: Optional[Dict[str, Any]]) -> None:
    if not isinstance(requirements, dict) or not constraints:
        return
    existing_project = constraints.get("existing_project")
    if not existing_project:
        return
    root = Path(str(existing_project))
    if not root.exists():
        return
    if requirements.get("function_inventory"):
        return
    inventory = _build_function_inventory(root)
    requirements["function_inventory"] = inventory
    requirements.setdefault(
        "function_plan_rules",
        [
            "优先做函数级别的任务拆解，再做目录/模块级规划。",
            "按 AST/目录结构识别可改动的函数与依赖关系。",
            "每轮只完成少量函数的可验证改动，避免大范围改写。",
        ],
    )

def _limit_tasks_and_roles(
    tasks: List[Dict[str, Any]],
    roles: List[Dict[str, Any]],
    max_tasks: int,
    max_roles: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if max_tasks > 0 and len(tasks) > max_tasks:
        tasks = tasks[:max_tasks]
    if max_roles > 0 and len(roles) > max_roles:
        roles = roles[:max_roles]
    return tasks, roles

def _resolve_snapshot_dir(config: Dict[str, Any]) -> Optional[Path]:
    workflow_cfg = config.get("workflow", {})
    if not workflow_cfg.get("snapshot_enabled", False):
        return None
    snapshot_dir = workflow_cfg.get("snapshot_dir", "output/snapshots")
    base_dir = Path(config.get("agent_root", Path(__file__).parent))
    path = Path(snapshot_dir)
    if not path.is_absolute():
        path = base_dir / path
    path.mkdir(parents=True, exist_ok=True)
    return path

def _maybe_save_snapshot(config: Dict[str, Any], payload: Dict[str, Any]) -> Optional[Path]:
    snapshot_dir = _resolve_snapshot_dir(config)
    if not snapshot_dir:
        return None
    status = payload.get("result", {}).get("status", "unknown")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot_path = snapshot_dir / f"{timestamp}_{status}.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return snapshot_path

def _finalize_result(result: Dict[str, Any], config: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
    snapshot_payload = {"input": input_data, "result": result}
    snapshot_path = _maybe_save_snapshot(config, snapshot_payload)
    if snapshot_path:
        result["snapshot_path"] = str(snapshot_path)
    return result

def load_config(config_path: str = None) -> Dict[str, Any]:
    # Wrapper to maintain compatible signature but use common logic with correct base
    base_dir = Path(__file__).parent.resolve()
    config = common_load_config(config_path, base_dir)
    
    # Inject agent_root for relative path resolution in chains if not present
    if "agent_root" not in config:
        config["agent_root"] = str(base_dir)
        
    return config

def run_architect(input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    user_input = input_data.get("user_input") or ""
    execution_feedback = input_data.get("execution_feedback")
    planner_state = input_data.get("planner_state")
    user_feedback = input_data.get("user_feedback")
    constraints = input_data.get("constraints")
    
    if execution_feedback:
        user_input += f"\n\n[Pre-existing Execution Feedback]: {execution_feedback}"
    constraints_text = _format_constraints(constraints)
    if constraints_text:
        user_input += f"\n\n[Constraints]\n{constraints_text}"

    workflow_cfg = config.get("workflow", {})
    max_iterations = workflow_cfg.get("max_iterations", 3)
    validation_threshold = workflow_cfg.get("validation_threshold", 0.8)
    max_roles = int(workflow_cfg.get("max_roles", 5) or 5)
    max_tasks = int(workflow_cfg.get("max_tasks", 20) or 20)
    max_jd_chars = int(workflow_cfg.get("max_jd_chars", 4000) or 4000)
    
    if planner_state:
        try:
            requirements = _require_dict("Planner state requirements", planner_state.get("requirements"))
            tasks = planner_state.get("tasks", [])
            roles = planner_state.get("roles", [])
            tasks, roles = _limit_tasks_and_roles(tasks, roles, max_tasks, max_roles)
            current_role_jds = planner_state.get("current_jds", {})
            if not isinstance(tasks, list) or not isinstance(roles, list) or not isinstance(current_role_jds, dict):
                raise ValueError("Planner state must include tasks(list), roles(list), current_jds(dict).")
            for role in roles:
                if not isinstance(role, dict) or not role.get("role_name"):
                    raise ValueError("Planner state roles must include 'role_name'.")
                if "initial_jd" not in role and role.get("role_name") not in current_role_jds:
                    raise ValueError("Planner state role missing 'initial_jd' and no current_jds entry.")
        except Exception as e:
            return _finalize_result({
                "status": "error",
                "error": f"Planner state invalid: {e}",
            }, config, input_data)
        _inject_function_plan(requirements, constraints)
        print(">>> 已检测到 planner_state，跳过需求分析与任务拆解，直接继续优化/验证...")
    else:
        # 1. 需求分析
        print(">>> 正在进行需求分析...")
        analyzer = build_analyzer_chain(config)
        try:
            print(">>> 需求分析输出(流式):", flush=True)
            requirements = _validate_requirements(
                _invoke_with_heartbeat(
                    "需求分析",
                    lambda: analyzer.invoke({"user_input": user_input, "constraints": constraints}),
                )
            )
            _inject_function_plan(requirements, constraints)
            print("\n", flush=True)
        except Exception as e:
            return _finalize_result({
                "status": "error",
                "error": f"Requirement analysis failed: {e}"
            }, config, input_data)
        print(f"    核心目标: {requirements.get('goal')}")

        # 2. 任务分类与角色拆解
        print(">>> 正在进行任务拆解与角色分配...")
        classifier = build_classifier_chain(config)
        try:
            classification = _invoke_with_heartbeat(
                "任务拆解",
                lambda: classifier.invoke({"requirements": json.dumps(requirements, ensure_ascii=False)}),
            )
            tasks, roles = _validate_classification(classification)
        except Exception as e:
            return _finalize_result({
                "status": "error",
                "error": f"Task classification failed: {e}"
            }, config, input_data)
        tasks, roles = _limit_tasks_and_roles(tasks, roles, max_tasks, max_roles)
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

        def _optimize_role(role_name: str, jd_input: Dict[str, Any]):
            try:
                opt_jd = _validate_optimizer_output(
                    role_name,
                    _invoke_with_heartbeat(
                        f"JD优化({role_name})",
                        lambda: optimizer.invoke(jd_input),
                    ),
                )
                return role_name, opt_jd, None
            except Exception as exc:
                return role_name, None, exc

        max_workers = min(4, max(1, len(roles)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for role in roles:
                role_name = role["role_name"]
                current_jd_content = current_role_jds.get(role_name, "")
                if isinstance(current_jd_content, str) and max_jd_chars > 0:
                    current_jd_content = current_jd_content[:max_jd_chars]
                current_feedback = f"Overall: {global_feedback}"
                if role_name in role_specific_feedbacks:
                    current_feedback += f"\nSpecific Advice for {role_name}: {role_specific_feedbacks[role_name]}"
                jd_input = {
                    "role_name": role_name,
                    "initial_jd": current_jd_content,
                    "requirements": json.dumps(requirements, ensure_ascii=False),
                    "tasks": json.dumps(tasks, ensure_ascii=False),
                    "feedback": current_feedback,
                }
                futures[executor.submit(_optimize_role, role_name, jd_input)] = role_name

            results: Dict[str, Dict[str, Any]] = {}
            errors: Dict[str, Exception] = {}
            for future in as_completed(futures):
                role_name, opt_jd, err = future.result()
                if err:
                    errors[role_name] = err
                elif opt_jd:
                    results[role_name] = opt_jd

        for role in roles:
            role_name = role["role_name"]
            if role_name in results:
                opt_jd = results[role_name]
                optimized_jds.append(opt_jd)
                next_round_jds[role_name] = json.dumps(opt_jd, ensure_ascii=False, indent=2)
            else:
                err = errors.get(role_name)
                print(f"    ❌ 优化角色 {role_name} 失败: {err}")
                optimized_jds.append({"role_name": role_name, "error": str(err)})
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
            validation_result = _validate_validator_output(
                _invoke_with_heartbeat(
                    "JD验证",
                    lambda: validator.invoke(validation_input),
                )
            )
        except Exception as e:
            return _finalize_result({
                "status": "error",
                "error": f"JD validation failed: {e}"
            }, config, input_data)

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
                return _finalize_result({
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
                }, config, input_data)

    
    if not final_jds:
        for role in roles:
            role_name = role.get("role_name", "Unknown")
            jd_content = current_role_jds.get(role_name) if planner_state else role.get("initial_jd", "")
            jd_payload = _coerce_json(jd_content)
            if isinstance(jd_payload, dict):
                final_jds.append(jd_payload)
            else:
                final_jds.append({"role_name": role_name, "content": jd_content})

    return _finalize_result({
        "status": "completed",
        "requirements": requirements,
        "tasks": tasks,
        "final_jds": final_jds,
        "validation_result": validation_result
    }, config, input_data)

def build_agent():
    config = load_config()
    return RunnableLambda(lambda x: run_architect(x, config))

if __name__ == "__main__":
    import sys
    
    user_input = "我想做一个简单的待办事项管理CLI工具，用Python写，数据存本地文件就行。"
    if len(sys.argv) > 1:
        user_input = sys.argv[1]
        
    agent = build_agent()
    result = agent.invoke({"user_input": user_input})
    print("\n========== 最终结果 ==========\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
