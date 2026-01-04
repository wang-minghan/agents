from pathlib import Path
from typing import Any, Dict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

def build_analyzer_chain(config: Dict[str, Any]):
    role_config = config.get("roles", {}).get("requirement_analyzer", {})
    model_name = role_config.get("model", "gpt-4")
    temperature = role_config.get("temperature", 0.3)
    api_key = role_config.get("api_key")
    api_base = role_config.get("api_base")
    
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=api_base
    )
    
    agent_root = Path(config.get("agent_root", "."))
    prompt_path = agent_root / "prompts/requirement_analyzer.md"
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()
    
    # 转义模板中的 JSON 花括号，保留变量占位符
    # 简单的做法是先全部转义，再把 user_input 还原
    safe_template = template.replace("{", "{{").replace("}", "}}").replace("{{user_input}}", "{user_input}")
    
    prompt = PromptTemplate(
        template=safe_template + "\n\n用户输入: {user_input}\n输出:",
        input_variables=["user_input"]
    )
    
    return prompt | llm | JsonOutputParser()
