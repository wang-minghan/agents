import json
from pathlib import Path
from typing import Any, Dict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

def build_optimizer_chain(config: Dict[str, Any]):
    role_config = config.get("roles", {}).get("jd_optimizer", {})
    model_name = role_config.get("model", "gpt-4")
    temperature = role_config.get("temperature", 0.7)
    api_key = role_config.get("api_key")
    api_base = role_config.get("api_base")
    
    llm = ChatOpenAI(
        model=model_name, 
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base=api_base
    )
    
    prompt_path = Path("agents/task_planner/prompts/jd_optimizer.md")
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()
    
    safe_template = template.replace("{", "{{").replace("}", "}}")
    
    prompt = PromptTemplate(
        template=safe_template + "\n\n输入:\n- Role: {role_name}\n- Initial JD: {initial_jd}\n- Requirements: {requirements}\n- Tasks: {tasks}\n\n输出:",
        input_variables=["role_name", "initial_jd", "requirements", "tasks"]
    )
    
    return prompt | llm | JsonOutputParser()
