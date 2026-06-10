"""
LLM-related data schemas.

We define our own types instead of leaking OpenAI SDK types throughout the codebase.
This makes provider swaps and testing easier.

LLM相关的数据模式。
我们定义自己的类型，而不是在整个代码库中泄漏OpenAI SDK类型。
这使得提供者交换和测试更容易。
"""
from typing import Literal, Any

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """LLM请求的工具调用。"""
    id: str = Field(description="本次通话的唯一标识符（工具结果中返回）")
    name: str = Field(description="要调用的工具名称")
    arguments_json: str = Field(description="参数的原始JSON字符串")

class Message(BaseModel):
    """A single message in chat conversation
    聊天对话中的一条消息"""

    role: Role
    content: str | None = None # 当助手消息仅为工具调用时，返回 None
    tool_call_id: str | None = Field(
        default=None,
        description="对于 role='tool'，表示正在响应的助手工具调用 ID",
    )
    tool_calls: list[ToolCall] | None = Field(
        default=None,
        description="对于角色='assistant'，模型在调用工具时请求的内容"
    )

    def to_openai(self) -> dict[str, Any]:
        """Convert to OpenAI API format."""
        msg: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments_json
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


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
    tools: list[dict] | None = Field(
        default=None,
        description="OpenAI格式的工具定义列表"
    )
    tool_choice: str | None = Field(
        default=None,
        description="'auto（如果工具集设置为默认值）、none，或特定工具名称"
    )


class TokenUsage(BaseModel):
    """Token usage of a chat call. Useful for cost tracking and observability."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    """接口返回数据格式"""

    # 大模型返回的内容
    content: str = Field(description="文本内容（如果模型返回了工具调用，则可能为空）")
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="模型请求的工具调用（如果模型给出了最终文本回答，则为空）"
    )
    # 使用模型名称
    model: str  
    # token 消耗统计
    usage: TokenUsage  
    # 结束原因
    finish_reason: str | None = None  
