"""
Tool-augmented conversation loop ("the agent loop").

Standard structure:
  1. Send messages + tool defs to LLM
  2. If LLM returns tool_calls:
       - Execute each tool
       - Append assistant message (with tool_calls) and tool result messages
       - Repeat
  3. If LLM returns text content (no tool_calls): that's the final answer
  4. Stop at max_iterations (safety bound against runaway loops)

This is the simplest agent: LLM as planner + executor. More sophisticated
agents add memory, multi-agent coordination, explicit planning, etc.,
but they all build on this core loop.

工具增强的对话循环（“代理循环”）。
标准结构：
    1. 将消息和工具定义发送给LLM
    2. 如果 LLM 返回 tool_calls：
        -- 执行每个工具
        -- 追加助手消息（包含 tool_calls）和工具结果消息
        - 重复此过程
    3. 如果LLM返回文本内容（无工具调用）：那就是最终答案
    4. 在 max_iterations 处停止（防止无限循环的安全限制）
这是最简单的代理：大语言模型作为规划者和执行者。更复杂的代理会添加记忆、多智能体协调、显式规划等功能，但它们都建立在这一核心循环之上。
"""
import json
from typing import Any

from pydantic import BaseModel, Field

from cdss.core.logging import get_logger
from cdss.llm.client import LLMClient
from cdss.llm.schemas import ChatRequest, Message, ToolCall
from cdss.tools.base import ToolExecutionError
from cdss.tools.registry import ToolRegistry


logger = get_logger(__name__)


class ToolInvocation(BaseModel):
    """单次工具调用记录，用于审计和可观测性"""

    tool_name: str
    arguments: dict[str, Any]
    result: str | None = None
    error: str | None = None


class AgentRunResult(BaseModel):
    """工具辅助对话的最终结果。"""

    final_message: str
    iterations: int
    invocations: list[ToolInvocation] = Field(default_factory=list)
    total_tokens: int
    completed: bool = Field(
        description="如果在未获得最终文本答案的情况下达到最大迭代次数，则返回False"
    )


async def run_with_tools(
        client: LLMClient,
        initial_messages: list[Message],
        registry: ToolRegistry,
        max_iterations: int = 5,
        temperature: float = 0.2,
) -> AgentRunResult:
    """进行一次多轮对话，其中大语言模型可以调用工具。"""
    messages = list(initial_messages)
    invocations: list[ToolInvocation] = []
    total_tokens = 0

    tool_schemas = registry.to_openai_schemas()

    for iteration in range(1, max_iterations + 1):
        logger.info(
            "agent_iteration",
            iteration=iteration,
            num_messages=len(messages),
        )

        response = await client.chat(
            ChatRequest(
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
                temperature=temperature,
            )
        )
        total_tokens += response.usage.total_tokens

        # 无需工具调用，LLM直接给出最终答案
        if not response.tool_calls:
            logger.info(
                "agent_complete",
                iteration=iteration,
                total_tokens=total_tokens,
                num_invocations=len(invocations),
            )

            return AgentRunResult(
                final_message=response.content,
                iterations=iteration,
                invocations=invocations,
                total_tokens=total_tokens,
                completed=True,
            )
        
        # 附加包含工具调用的助手消息
        messages.append(
            Message(
                role="assistant",
                content=response.content or None,
                tool_calls=response.tool_calls,
            )
        )

        # 执行每个工具调用，并将结果作为工具消息附加
        for tool_call in response.tool_calls:
            invocation = await _execute_tool_call(registry, tool_call)
            invocations.append(invocation)
            messages.append(
                Message(
                    role="tool",
                    tool_call_id=tool_call.id,
                    content=invocation.result
                    if invocation.result is not None
                    else f"ERROR: {invocation.error}",
                )
            )

    # 最大迭代次数已用完
    logger.warning(
        "agent_max_iterations_reached",
        max_iterations=max_iterations,
        invocation=len(invocations),
    )
    return AgentRunResult(
        final_message=(
            f"达到最大迭代次数({max_iterations}), 未能得出最终答案."
            f"已执行 {len(invocations)} 次工具调用."
        ),
        iterations=max_iterations,
        invocations=invocations,
        total_tokens=total_tokens,
        completed=False,
    )


async def _execute_tool_call(
        registry: ToolRegistry,
        call: ToolCall,
) -> ToolInvocation:
    """执行一次工具调用，并将所有错误记录到调用日志中。"""
    # 无效的参数（LLM生成的JSON可能格式错误）
    try:
        arguments = json.loads(call.arguments_json)
    except json.JSONDecodeError as e:
        logger.warning("tool_arguments_parse_failed", tool=call.name, error=str(e))
        return ToolInvocation(
            tool_name=call.name,
            arguments={},
            error=f"参数 JSON 解析失败: {e}",
        )
    
    # 查找该工具
    try:
        tool = registry.get(call.name)
    except KeyError as e:
        return ToolInvocation(
            tool_name=call.name,
            arguments=arguments,
            error=str(e),
        )
    
    # Execute 执行
    try:
        result = await tool.execute(arguments)
        logger.info("tool_executed", tool=call.name, result_len=len(result))
        return ToolInvocation(
            tool_name=call.name,
            arguments=arguments,
            result=result,
        )
    except ToolExecutionError as e:
        logger.warning("tool_execution_failed", tool=call.name, error=str(e))
        return ToolInvocation(
            tool_name=call.name,
            arguments=arguments,
            error=str(e),
        )
    
