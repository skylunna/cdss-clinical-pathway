"""
Structured output helper.

Turns an LLM call into a typed function: input prompt, output a validated
Pydantic instance. This is the foundation for Agent tool calling, rule
engine integration, and reliable multi-step pipelines.

Strategy:
    1. Inject the Pydantic model's JSON Schema into the system message
    2. Enable JSON mode on the underlying API (guarantees valid JSON)
    3. Parse the JSON response and validate against the Pydantic model
    4. On parse or validation failure, retry with error feedback to the model

结构化输出助手。
将LLM调用转换为一个可编程函数：
    输入提示，输出经过验证的Pydantic实例。这是代理工具调用、规则引擎集成以及可靠多步骤流程的基础。
策略：
    1. 将 Pydantic 模型的 JSON Schema 注入系统消息
    2. 在底层 API 上启用 JSON 模式（确保生成有效的 JSON）
    3. 解析 JSON 响应并验证其与 Pydantic 模型的一致性
    4. 在解析或验证失败时，向模型返回错误反馈并重试
"""
import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from cdss.core.logging import get_logger
from cdss.llm.client import LLMClient, LLMError
from cdss.llm.schemas import ChatRequest, Message


logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """当LLM在重试后仍无法生成有效的结构化输出时，会抛出此异常。"""


def _render_schema_instruction(output_schema: type[BaseModel]) -> str:
    """将 Pydantic 模型转换为提示指令片段。"""
    schema = output_schema.model_json_schema()
    schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
    return (
        "\n\n## 输出格式(必须严格遵守)\n\n"
        "你的回答必须是一个**合法的 JSON 对象**,严格匹配以下 JSON Schema。"
        "不要输出 JSON 之外的任何文字(不要解释、不要 Markdown 代码块标记):\n\n"
        f"```json\n{schema_str}\n```"
    )


async def structured_chat(
        client: LLMClient,
        request: ChatRequest,
        output_schema: type[T],
        max_retries: int = 2,
) -> tuple[T, int]:
    """
    使用 Pydantic 输出模式调用 LLM，返回验证后的实例。
    参数：
        client：要使用的LLM客户端。
        request：聊天请求（消息、温度等）。
        json_mode强制设为True；请勿手动设置。
        output_schema：描述期望输出的Pydantic BaseModel子类。
        max_retries：在解析/验证失败时重试的次数。
    返回值：
        一个经过验证的 `output_schema` 实例。
    返回：
        StructuredOutputError：如果所有尝试均失败时。
        LLMError：当底层API出现故障（网络、认证等）时。
    """
    # 1. 将模式指令注入（或插入）系统消息中
    schema_instruction = _render_schema_instruction(output_schema)
    base_messages = list(request.messages)

    if base_messages and base_messages[0].role == "system":
        base_messages[0] = Message(
            role="system",
            content=base_messages[0].content + schema_instruction,
        )
    else:
        base_messages.insert(
            0, Message(role="system", content=schema_instruction.strip())
        )

    last_error: str | None = None

    for attempt in range(max_retries + 1):
        # 为本次尝试构建消息；重试时附加错误反馈
        attempt_message = list(base_messages)
        if last_error is not None:
            attempt_message.append(
                Message(
                    role="user",
                    content=(
                        f"你上一次的输出无法被解析为符合 schema 的 JSON。"
                        f"错误: {last_error}。请严格按照 JSON Schema 重新生成。"
                    ),
                )
            )
        
        attempt_request = request.model_copy(
            update={"messages": attempt_message, "json_mode": True}
        )

        response = await client.chat(attempt_request)

        # 2. 解析 JSON
        try:
            data = json.loads(response.content)
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            logger.warning(
                "structured_output_parse_failed",
                attempt=attempt + 1,
                error=str(e),
            )
            continue

        # 3. 验证 Pydantic 格式
        try:
            validated = output_schema.model_validate(data)
        except ValidationError as e:
            last_error = f"Schema validation error: {e.errors(include_url=False)}"
            logger.warning(
                "structured_output_validation_failed",
                attempt=attempt + 1,
                error=str(e),
            )
            continue

        logger.info(
            "structured_output_success",
            attempt=attempt + 1,
            schema=output_schema.__name__,
        )
        return validated, response.usage.total_tokens
    
    raise StructuredOutputError(
        f"Failed to produce valid {output_schema.__name__} after "
        f"{max_retries + 1} attempts. Last error: {last_error}"
    )