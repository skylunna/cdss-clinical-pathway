"""
LLM-related data schemas.

We define our own types instead of leaking OpenAI SDK types throughout the codebase.
This makes provider swaps and testing easier.

LLM相关的数据模式。
我们定义自己的类型，而不是在整个代码库中泄漏OpenAI SDK类型。
这使得提供者交换和测试更容易。
"""

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """A single message in chat conversation
    聊天对话中的一条消息"""

    role: Role
    content: str

    def to_openai(self) -> dict:
        """Convert to OpenAI API format."""
        return {"role": self.role, "content": self.content}


class ChatRequest(BaseModel):
    """Input to a chat completion call."""

    messages: list[Message]
    model: str | None = Field(
        default=None,
        description="Model name. If None, use the default configured model."
    )
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    json_mode: bool = Field(
        default=False,
        description="如果为真，则将 response_format 设置为 JSON 对象。 "
                    "提示必须包含“JSON”（OpenAI兼容API要求）。"
    )


class TokenUsage(BaseModel):
    """Token usage of a chat call. Useful for cost tracking and observability."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    """接口返回数据格式"""

    content: str  # 大模型返回的内容
    model: str  # 使用模型名称
    usage: TokenUsage  # token 消耗统计
    finish_reason: str | None = None  # 结束原因
