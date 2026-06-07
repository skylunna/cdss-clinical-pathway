"""
Diagnosis recommendation endpoint.

Demonstrates prompt-driven LLM use:
- Loads a prompt template by name
- Renders it with user-provided patient info
- Calls the LLM with template-suggested parameters
- Returns the (still unstructured) Markdown response

Next milestone will add structured output (JSON schema) so the response
becomes machine-readable, not just for human reading.

诊断推荐
演示提示驱动的LLM使用：
    - 按名称加载提示模板
    - 使用用户提供的患者信息呈现
    - 使用模板建议的参数调用LLM
    - 返回（仍然是非结构化的）Markdown响应
下一个里程碑将添加结构化输出（JSON模式），以便响应
变得机器可读，而不仅仅是供人类阅读。
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from cdss.llm.client import LLMClient, LLMError, get_llm_client
from cdss.llm.prompts import PromptRegistry, get_prompt_registry
from cdss.llm.schemas import ChatRequest


router = APIRouter(prefix="/api/v1", tags=["diagnose"])


class DiagnoseRequest(BaseModel):
    patient_info: str = Field(
        ...,
        description="患者信息：主诉、病史、体格检查、实验室检查",
        min_length=10,
    )


class DiagnoseResponse(BaseModel):
    diagnosis_markdown: str = Field(description="Markdown格式的鉴别诊断")
    prompt_name: str
    prompt_version: int
    model: str
    total_tokens: int


@router.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(
    request: DiagnoseRequest,
    client: LLMClient = Depends(get_llm_client),
    prompts: PromptRegistry = Depends(get_prompt_registry),
) -> DiagnoseResponse:
    """根据提供的信息为患者生成鉴别诊断。"""
    template = prompts.get("diagnosis_recommendation")
    messages = template.render(patient_info=request.patient_info)

    chat_request = ChatRequest(
        messages=messages,
        temperature=template.model_hints.temperature or 0.3,
        max_tokens=template.model_hints.max_tokens,
    )

    try:
        response = await client.chat(chat_request)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return DiagnoseResponse(
        diagnosis_markdown=response.content,
        prompt_name=template.name,
        prompt_version=template.version,
        model=response.model,
        total_tokens=response.usage.total_tokens,
    )
