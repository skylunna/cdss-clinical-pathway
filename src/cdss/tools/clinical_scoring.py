"""
Clinical scoring tools.

Wraps deterministic scoring rules (from cdss.rules.scoring) as LLM tools.
The rules themselves stay free of any tool-system dependency.

临床评分工具。
将确定性评分规则（来自 cdss.rules.scoring）作为 LLM 工具进行封装。  
这些规则本身不依赖任何工具系统。
"""
from cdss.rules.scoring import curb65
from cdss.tools.base import Tool
from cdss.tools.registry import ToolRegistry


curb65_tool = Tool(
    name=curb65.RULE_NAME,
    description=curb65.RULE_DESCRIPTION,
    input_schema=curb65.CURB65Criterion,
    handler=curb65.compute,
)


def register_all(registry: ToolRegistry) -> None:
    """将所有临床评分工具登记到指定的注册系统中。"""
    registry.register(curb65_tool)