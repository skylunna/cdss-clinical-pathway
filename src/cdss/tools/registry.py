"""
Tool registry.

Central place where all tools available to the LLM are registered.
Used to look up tools by name during execution, and to produce the
OpenAI-format tool schemas attached to LLM requests.

工具注册表。
所有LLM可用工具的注册中心。  
用于在执行过程中通过名称查找工具，并生成附带LLM请求的OpenAI格式工具架构。
"""
from cdss.tools.base import Tool


class ToolRegistry:
    """包含LLM调用可用的一系列工具。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            available = ", ".join(sorted(self._tools.keys())) or "<none>"
            raise KeyError(
                f"Tool '{name}' not found. Available: {available}"
            )
        return self._tools[name]
    
    def list_all(self) -> list[Tool]:
        return list(self._tools.values())
    
    def to_openai_schemas(self) -> list[dict]:
        """Schemas in OpenAI tool format, to attach to ChatRequest.tools."""
        return [t.to_openai_schema() for t in self._tools.values()]