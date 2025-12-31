import sys
import json
from unittest.mock import MagicMock, patch
from agents.task_planner.agent import run_task_planner

def run_demo():
    print("==================================================")
    print("   Task Planner Agent - 演示模式 (无网络环境)")
    print("==================================================\n")

    user_input = "我想做一个简单的待办事项管理CLI工具，用Python写，数据存本地文件就行。"
    print(f"用户输入: {user_input}\n")

    # 1. 模拟配置
    config = {
        "workflow": {"max_iterations": 3},
        "roles": {}
    }

    # 2. 准备模拟数据 (Mock Data)
    
    # 模拟需求分析结果
    mock_requirements = {
        "goal": "开发一个基于Python的轻量级命令行待办事项管理工具",
        "key_features": [
            "添加新任务 (Add)",
            "查看任务列表 (List)", 
            "标记任务完成 (Complete)",
            "删除任务 (Delete)",
            "数据本地持久化 (JSON/CSV)"
        ],
        "constraints": ["使用 Python 3", "不依赖复杂第三方库", "交互简洁"],
        "ambiguities": [],
        "priority": "Medium"
    }

    # 模拟任务拆解结果
    mock_classification = {
        "tasks": [
            {
                "id": "T-001", 
                "name": "项目初始化与CLI骨架搭建", 
                "description": "配置argparse/click解析命令行参数，建立项目结构",
                "estimated_effort": "2h"
            },
            {
                "id": "T-002", 
                "name": "数据存储模块实现", 
                "description": "封装FileStorage类，处理JSON文件的读写操作",
                "estimated_effort": "3h"
            },
            {
                "id": "T-003", 
                "name": "业务逻辑开发", 
                "description": "实现add, list, complete, delete具体功能逻辑",
                "estimated_effort": "4h"
            }
        ],
        "roles": [
            {
                "role_name": "Python Backend Engineer",
                "initial_jd": "负责Python代码编写，CLI交互逻辑设计以及文件存储实现。"
            }
        ]
    }

    # 模拟 JD 优化结果
    mock_optimized_jd = {
        "role_name": "Python Backend Engineer (CLI Tool)",
        "responsibilities": [
            "负责 CLI 工具的核心功能开发与维护",
            "设计清晰的数据存储结构 (Schema)",
            "编写单元测试确保功能稳定性"
        ],
        "required_skills": [
            "熟练掌握 Python 3 标准库 (argparse, json, os, sys)",
            "良好的代码规范 (PEP 8)",
            "基本的文件I/O操作经验"
        ],
        "preferred_skills": ["熟悉 Click 或 Typer 库", "有发布 PyPI 包的经验"],
        "tool_stack": ["Python 3.10+", "Git", "VS Code"],
        "acceptance_criteria": [
            "所有核心命令 (add, list, done, rm) 均可正常工作",
            "程序重启后数据不丢失",
            "代码通过 pylint 检查"
        ],
        "collaboration_requirements": ["代码需提交至 Git 仓库", "编写 README.md 使用说明"]
    }

    # 模拟验证结果 (通过)
    mock_validation = {
        "passed": True,
        "score": 0.95,
        "overall_feedback": "JD 定义清晰，完全覆盖了简单的 CLI 工具开发需求，技能要求合理。",
        "user_feedback_needed": []
    }

    # 3. 设置 Mock 对象
    # 我们不仅要 mock build_xxx_chain，还要 mock chain.invoke() 的返回值
    
    def mock_chain_builder(return_value):
        chain = MagicMock()
        chain.invoke.return_value = return_value
        return chain

    # 使用 patch 上下文管理器拦截真实的链构建函数
    with patch('agents.task_planner.agent.build_analyzer_chain', return_value=mock_chain_builder(mock_requirements)), \
         patch('agents.task_planner.agent.build_classifier_chain', return_value=mock_chain_builder(mock_classification)), \
         patch('agents.task_planner.agent.build_optimizer_chain', return_value=mock_chain_builder(mock_optimized_jd)), \
         patch('agents.task_planner.agent.build_validator_chain', return_value=mock_chain_builder(mock_validation)):
        
        # 4. 运行 Agent
        result = run_task_planner({"user_input": user_input}, config)
        
        print("\n\n==================================================")
        print("   最终产出结果 (Result)")
        print("==================================================")
        print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    run_demo()
