"""
Chat endpoint - minimal test endpoint for verifying LLM integration.

This is NOT the production agent endpoint. It's a smoke test.

聊天端点-用于验证LLM集成的最小测试端点。
这不是生产代理端点。这是一个烟雾测试。
"""

from fastapi import APIRouter, Depends, HTTPException

from cdss.llm.client import LLMClient, LLMError, get_llm_client
from cdss.llm.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    client: LLMClient = Depends(get_llm_client),
) -> ChatResponse:
    """
    向LLM发送聊天请求。
    """
    try:
        return await client.chat(request)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(2)) from e
