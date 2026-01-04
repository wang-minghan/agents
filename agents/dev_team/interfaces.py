from typing import Protocol, Any, Dict, List, runtime_checkable

@runtime_checkable
class MemoryStore(Protocol):
    def get_context_for_role(self, role_name: str, role_type: str | None = None) -> str: ...
    def get_peer_output_summaries(
        self,
        requesting_role: str,
        include_qa: bool = False,
        summary_max_chars: int | None = None,
    ) -> Dict[str, str]: ...
    def add_output(self, role_name: str, content: str) -> None: ...
    def get_all_outputs(self) -> Dict[str, List[str]]: ...
    def add_qa_feedback(self, feedback: Dict[str, Any]) -> None: ...

@runtime_checkable
class CodeExecutor(Protocol):
    def run_tests(self, test_dir: str) -> str: ...

@runtime_checkable
class Agent(Protocol):
    role_name: str
    role_type: str
    
    def run(self) -> str: ...
