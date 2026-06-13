"""
Clinical treatment recommendation tools.

Wraps deterministic treatment rules as LLM-callable tools.

临床治疗建议工具。
将确定性处理规则封装为可调用LLM的工具。
"""
from cdss.rules.treatment import cap_antibiotic
from cdss.tools.base import Tool
from cdss.tools.registry import ToolRegistry


cap_antibiotic_tool = Tool(
    name=cap_antibiotic.RULE_NAME,
    description=cap_antibiotic.RULE_DESCRIPTION,
    input_schema=cap_antibiotic.CAPAntibioticInput,
    handler=cap_antibiotic.compute,
)


def register_all(registry: ToolRegistry) -> None:
    """Register all clinical treatment tools into the given registry."""
    registry.register(cap_antibiotic_tool)