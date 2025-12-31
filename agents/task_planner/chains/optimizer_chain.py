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
        api_key=api_key,
        base_url=api_base
    )
    
    agent_root = Path(config.get("agent_root", "."))
    prompt_path = agent_root / "prompts/jd_optimizer.md"
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()
    
    # safe_template = template # template is now expected to be valid f-string format (braces escaped where needed)
    
    prompt = PromptTemplate(
        template=template + "\n\n输入:\n- Role: {role_name}\n- Initial JD: {initial_jd}\n- Requirements: {requirements}\n- Tasks: {tasks}\n- Feedback: {feedback}\n\n输出:",
        input_variables=["role_name", "initial_jd", "requirements", "tasks", "feedback"]
    )
    
    return prompt | llm | JsonOutputParser()
