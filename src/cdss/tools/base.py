"""
Tool abstraction for LLM function calling.

A Tool wraps a Python function with:
    - name + description (LLM uses these to decide when to call it)
    - Pydantic input schema (defines arguments, validates them at execution time)
    - handler (the actual function, sync or async)

Tools are kept separate from the rules/services they wrap, so the wrapped
code stays usable independently (e.g., for unit tests, direct API endpoints).


LLM函数调用的工具抽象。
一个工具将 Python 函数包装起来，包含以下内容：
    - 名称 + 描述（LLM 会用这些信息决定何时调用该函数）
    - Pydantic 输入 Schema（定义参数，并在执行时进行验证）
    - 处理器（实际的函数，同步或异步）
工具与它们所封装的规则/服务分开存放，以便被封装的代码能够独立使用（例如用于单元测试、直接的API端点）。
"""
import inspect
import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from cdss.core.logging import get_logger


logger = get_logger(__name__)


class ToolExecutionError(Exception):
    """Raised when a tool fails to execute (invalid args or handler error)."""
    # 当工具执行失败时（无效参数或处理程序错误）抛出。


class Tool:
    """可通过函数调用由LLM调用的函数。"""

    def __init__(
            self,
            name: str,
            description: str,
            input_schema: type[BaseModel],
            handler: Callable[..., Any],
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler

    async def execute(self, arguments: dict[str, Any]) -> str:
        """
        验证参数并调用处理器。
        返回序列化为字符串的结果（LLM 消费字符串）。  
        在验证或执行失败时抛出 ToolExecutionError 异常。
        """
        
        # 验证参数
        try:
            validated = self.input_schema.model_validate(arguments)
        except ValidationError as e:
            raise ToolExecutionError(
                f"Invalid arguments for tool '{self.name}': "
                f"{e.errors(include_url=False)}"
            ) from e
        
        # 调用处理器（支持同步和异步）
        try:
            if inspect.iscoroutinefunction(self.handler):
                # 异步
                result = await self.handler(validated)
            else:
                # 同步
                result = self.handler(validated)
        except Exception as e:
            raise ToolExecutionError(
                f"Tool '{self.name}' execution failed: {e}"
            ) from e
        
        # 将结果序列化以供大语言模型使用
        return self._serialize_result(result)
    
    @staticmethod
    def _serialize_result(result: Any) -> str:
        """将任何结果类型转换为适合LLM的字符串。"""
        if isinstance(result, BaseModel):
            return result.model_dump_json()
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False, default=str)
        return str(result)
    
    def to_openai_schema(self) -> dict[str, Any]:
        """转换为 OpenAI 工具定义格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema.model_json_schema(),
            },
        }

