import json
from pathlib import Path
from typing import Any, Dict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

def build_classifier_chain(config: Dict[str, Any]):
    role_config = config.get("roles", {}).get("task_classifier", {})
    model_name = role_config.get("model", "gpt-4")
    temperature = role_config.get("temperature", 0.5)
    api_key = role_config.get("api_key")
    api_base = role_config.get("api_base")
    
    llm = ChatOpenAI(
        model=model_name, 
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base=api_base
    )
    
    prompt_path = Path("agents/task_planner/prompts/task_classifier.md")
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()
    
    safe_template = template.replace("{", "{{").replace("}", "}}")
    
    prompt = PromptTemplate(
        template=safe_template + "\n\n输入: {requirements}\n输出:",
        input_variables=["requirements"]
    )
    
    return prompt | llm | JsonOutputParser()
