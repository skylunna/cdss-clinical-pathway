"""
Agent endpoint - tool-augmented patient assessment.

Demonstrates the full agent loop: free-text patient info in,
LLM-orchestrated tool calls in the middle, structured final assessment out.

代理端点——工具辅助的患者评估。
展示完整的代理循环：输入自由文本患者信息，中间由大语言模型协调工具调用，最终输出结构化评估结果。
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from cdss.llm.client import LLMClient, LLMError, get_llm_client
from cdss.llm.schemas import Message
from cdss.llm.tool_use import AgentRunResult, run_with_tools
from cdss.tools import get_tool_registry
from cdss.tools.registry import ToolRegistry


router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


SYSTEM_PROMPT = """你是一位临床决策支持助手。根据接诊医师提供的患者信息,
评估病情严重程度并给出处置建议。

## 工作原则

1. **优先使用工具**:需要计算临床评分时,**必须调用对应工具**,不要自己心算。
2. **参数提取要精准**:从患者信息中提取工具所需的具体数值;若关键参数缺失,
   仍可调用工具但在最终回答中明确指出。
3. **综合解读**:工具返回数值后,结合临床背景给出可操作的处置建议。
4. **避免过度调用**:每个评分至多调用一次。若信息不足以调用,直接说明。

## 最终输出应包含

- 患者关键临床信息摘要
- 调用的工具及其结果解读
- 处置建议(明确的下一步行动)
- 数据完整性说明(如有缺失)

## 重要声明

最终输出必须以下面这段话结尾:

> 本建议仅供医师参考,不可替代临床判断。最终处置须由接诊医师综合决定。
"""

class AssessRequest(BaseModel):
    patient_info: str = Field(
        ...,
        min_length=10,
        description="患者自由文本信息:主诉、现病史、查体、辅检等"
    )
    max_iterations: int = Field(default=5, ge=1, le=10)


@router.post("/assess", response_model=AgentRunResult)
async def assess_patient(
    request: AssessRequest,
    client: LLMClient = Depends(get_llm_client),
    registry: ToolRegistry = Depends(get_tool_registry),
) -> AgentRunResult:
    """运行工具增强的患者评估。"""
    messages = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(
            role="user",
            content=f"请评估以下患者:\n\n---\n{request.patient_info}\n---",
        ),
    ]

    try:
        return await run_with_tools(
            client=client,
            initial_messages=messages,
            registry=registry,
            max_iterations=request.max_iterations,
        )
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e