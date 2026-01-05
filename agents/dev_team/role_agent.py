import json
from pathlib import Path
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from agents.dev_team.interfaces import MemoryStore


class RoleAgent:
    def __init__(self, role_jd: Dict[str, Any], config: Dict[str, Any], memory: MemoryStore, output_dir: Path = None):
        self.role_name = role_jd.get("role_name", "Unknown")
        self.role_type = role_jd.get("role_type", "ENGINEER").upper() # Default to ENGINEER
        self.role_jd = role_jd
        self.config = config
        self.memory = memory
        self.output_dir = output_dir

        # 初始化 LLM - 根据 role_type 决定 temperature
        llm_cfg = config.get("llm", {})
        
        # 兼容旧逻辑：如果 role_type 是 ENGINEER 但名字里有 QA/Test，也当作 QA
        role_upper = self.role_name.upper()
        self.is_qa = self.role_type == "QA" or "QA" in role_upper or "TEST" in role_upper
        
        temp = 0.3 if self.is_qa else 0.7

        self.llm = ChatOpenAI(
            model=llm_cfg.get("model", "gpt-4o"),
            api_key=llm_cfg.get("api_key"),
            base_url=llm_cfg.get("api_base"),
            temperature=temp
        )
        
        # 加载 System Prompt 模板
        # 优先使用 role_type 对应的 prompt
        if self.is_qa:
            prompt_path_str = self.config["roles"]["qa"]["prompt_path"]
        else:
            prompt_path_str = self.config["roles"]["engineer"]["prompt_path"]
            
        # 动态处理路径
        # Assuming config["agent_root"] might be available or we resolve relative
        base_dir = Path(config.get("agent_root", "."))
        if not Path(prompt_path_str).is_absolute():
             prompt_path = base_dir / prompt_path_str
        else:
             prompt_path = Path(prompt_path_str)

        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt_template = f.read()
        self.prompt_template = ChatPromptTemplate.from_template(self.system_prompt_template)

    # extract_and_save_files method removed (Decoupled IO)

    def run(self) -> str:
        # 从共享记忆获取上下文
        context = self.memory.get_context_for_role(self.role_name, self.role_type)

        # 渲染 Prompt
        # 根据不同角色的 Prompt 需要的变量进行填充
        # Engineer prompt 需要: role_jd, requirements
        # QA prompt 需要: role_jd, engineer_output
        # QA prompt 需要: role_jd, engineer_output
        
        engineer_output = {}
        if self.is_qa:
            engineer_output = self.memory.get_peer_output_summaries(self.role_name, include_qa=False)

        prompt_kwargs = {
            "role_jd": json.dumps(self.role_jd, ensure_ascii=False, indent=2),
            "requirements": self.memory.global_context.get("requirements", ""),
            "engineer_output": json.dumps(engineer_output, ensure_ascii=False),
            "test_results": self.memory.global_context.get("latest_test_results", "暂无测试运行结果"),
            "bug_cards": json.dumps(self.memory.global_context.get("bug_cards", []), ensure_ascii=False),
        }
        
        system_instruction = self.prompt_template.format(**prompt_kwargs)

        user_content = (
            f"【当前任务】\n你现在的角色是: {self.role_name}\n"
            f"你的职责是: {json.dumps(self.role_jd.get('responsibilities', []), ensure_ascii=False)}\n\n"
            f"【共享上下文】\n{context}\n\n"
            f"请开始你的工作。如果需要编写代码，请务必使用 <file path='...'>...</file> 格式包裹代码。"
        )

        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=user_content)
        ]

        print(f"\n🤖 [{self.role_name}] 正在思考并输出...\n")
        
        full_response = ""
        try:
            # 使用流式输出
            for chunk in self.llm.stream(messages):
                content = chunk.content
                if isinstance(content, str):
                    print(content, end="", flush=True)
                    full_response += content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, str):
                            print(item, end="", flush=True)
                            full_response += item
                        elif isinstance(item, dict) and "text" in item:
                            text = item["text"]
                            print(text, end="", flush=True)
                            full_response += text
            
            print("\n") # 换行

        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            print(f"\n❌ {error_msg}")
            full_response = error_msg

        # 尝试提取并保存文件 -> moved to coordinator
        # self.extract_and_save_files(full_response)

        # 将输出存入共享记忆
        self.memory.add_output(self.role_name, full_response)
        return str(full_response)
