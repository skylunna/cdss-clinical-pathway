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
from cdss.llm.prompts import PromptRegistry, get_prompt_registry
from cdss.llm.tool_use import AgentRunResult, run_with_tools
from cdss.tools import get_tool_registry
from cdss.tools.registry import ToolRegistry


router = APIRouter(prefix="/api/v1/agent", tags=["agent"])

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
    prompts: PromptRegistry = Depends(get_prompt_registry),
) -> AgentRunResult:
    """运行工具增强的患者评估。"""
    template = prompts.get("agent_assessment_v1")
    messages = template.render(patient_info=request.patient_info)

    try:
        return await run_with_tools(
            client=client,
            initial_messages=messages,
            registry=registry,
            max_iterations=request.max_iterations,
        )
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e