"""
LLM client wrapper.

Wraps the OpenAI-compatible SDK and exposes a clean interface using our own types.
All LLM calls in the application should go through this client.

LLM客户端包装器。
封装OpenAI兼容的SDK，并使用我们自己的类型公开一个干净的接口。
应用程序中的所有LLM调用都应该通过此客户端。
"""
from functools import lru_cache

from openai import AsyncOpenAI
from openai import APIError, APITimeoutError, RateLimitError

from cdss.core.config import get_settings
from cdss.core.logging import get_logger
from cdss.llm.schemas import ChatRequest, ChatResponse, TokenUsage


logger = get_logger(__name__)


class LLMError(Exception):
    """LLM相关的基本异常"""


class LLMClient:
    """ 
    围绕OpenAI兼容的LLM提供者进行包装。目前支持DeepSeek（以及任何与OpenAI兼容的端点）。
    未来：通过“提供者”字段路由到不同的提供者。
    """
    def __init__(
        self,
        api_key: str,
        base_url: str,
        default_model: str,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._default_model = default_model

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        发送聊天完成请求
        """
        model = request.model or self._default_model

        logger.info(
            "llm_request",
            model=model,
            num_messages=len(request.messages),
            temperature=request.temperature,
        )
        
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=[m.to_openai() for m in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        except (APIError, APITimeoutError, RateLimitError) as e:
            logger.error("llm_request_failed", model=model, error=str(e))
            raise LLMError(f"LLM call failed: {e}") from e
        
        choice = response.choices[0]
        usage = response.usage

        result = ChatResponse(
            content=choice.message.content or "",
            model=response.model,
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
            finish_reason=choice.finish_reason,
        )

        logger.info(
            "llm_response",
            model=result.model,
            total_tokens=result.usage.total_tokens,
            finish_reason=result.finish_reason,
        )

        return result


@lru_cache
def get_llm_client() -> LLMClient:
    """
    获取客户端
    Raises:
        如果配API Key 缺失
    """
    settings = get_settings()
    provider = settings.default_llm_provider

    if provider == "deepseek":
        if not settings.deepseek_api_key:
            raise LLMError(
                "DEEPSEEK_API_KEY is not configured — set it in your .env file"
            )
        return LLMClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            default_model=settings.default_llm_model,
        )
    
    if provider == "openai":
        if not settings.openai_api_key:
            raise LLMError(
                "OPENAI_API_KEY is not configured — set it in your .env file"
            )
        
    raise LLMError(f"Unsupported LLM provider: {provider}")


