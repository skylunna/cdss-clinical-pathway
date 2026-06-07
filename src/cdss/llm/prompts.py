"""
Prompt template management.

Prompts are stored as YAML files under configs/prompts/.
On startup they are loaded, validated (via Pydantic) and indexed by name.
At call time they are rendered with Jinja2 and turned into a list of Messages.

Why externalize prompts:
- Non-developers can edit them
- Easy to version, diff, A/B test
- Centralized discovery (one directory, easy to audit)
- Decouples prompt engineering iteration from code releases

提示模板管理。
提示以YAML文件的形式存储在configs/promises/下。
启动时，它们会被加载、验证（通过Pydantic）并按名称索引。
在调用时，它们会被Jinja2渲染，并变成一个消息列表。
为什么要外部化提示：
    -非开发人员可以编辑它们
    -易于版本、差异、A/B测试
    -集中发现（一个目录，易于审计）
    -将快速工程迭代与代码发布解耦
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel, Field

from cdss.core.logging import get_logger
from cdss.llm.schemas import Message


logger = get_logger(__name__)

DEFAULT_PROMPT_DIR = Path("configs/prompts")


class PromptVariable(BaseModel):
    """提示模板所需的变量"""

    name: str
    required: bool = True
    description: str | None = None


class ModelHints(BaseModel):
    """此提示的建议模型参数（调用者可覆盖）。"""

    temperature: float | None = None
    max_tokens: int | None = None


class PromptTemplate(BaseModel):
    """一个已加载并解析的提示模板"""

    name: str  # 模版名称 必填
    version: int = 1  # 版本号
    description: str | None = None  # 描述 可选
    variables: list[PromptVariable] = Field(default_factory=list)  # 模版变量列表
    model_hints: ModelHints = Field(default_factory=ModelHints)  # 模型参数配置
    system: str | None = None  # 系统提示词 (system prompt)
    user: str  # 用户问题模版 必填

    def render(self, **variables: Any) -> list[Message]:
        """
        使用 Jinja2 将模板渲染为消息列表。
        使用 StrictUndefined：如果模板引用了未提供的变量，将抛出明确的错误（而不是静默地返回空字符串）。
        """
        # 在渲染前检查所需变量，以获得更清晰的错误信息
        provided = set(variables.keys())
        for var in self.variables:
            if var.required and var.name not in provided:
                raise ValueError(
                    f"Prompt '{self.name}' requires variable '{var.name}' (not provided)"
                )

        env = Environment(undefined=StrictUndefined, autoescape=False)
        messages: list[Message] = []
        if self.system:
            rendered_system = env.from_string(self.system).render(**variables)
            messages.append(Message(role="system", content=rendered_system))

        rendered_user = env.from_string(self.user).render(**variables)
        messages.append(Message(role="user", content=rendered_user))

        return messages


class PromptRegistry:
    """
    从目录中加载并提供提示模板。
    模板在启动时加载一次并保留在内存中；目前不支持热重载（有意为之：我们希望有明确且可追溯的更改）
    """

    def __init__(self, prompt_dir: Path = DEFAULT_PROMPT_DIR) -> None:
        self._prompt_dir = prompt_dir
        self._templates: dict[str, PromptTemplate] = {}

    def load_all(self) -> None:
        """加载 prompt_dir 下的所有 .yaml 文件（递归）。"""
        if not self._prompt_dir.exists():
            logger.warning("prompt_dir_missing", path=str(self._prompt_dir))
            return

        count = 0
        for yaml_file in sorted(self._prompt_dir.rglob("*.yaml")):
            with yaml_file.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            template = PromptTemplate(**data)
            if template.name in self._templates:
                raise ValueError(
                    f"Duplicate prompt name '{template.name}' (latest from {yaml_file})"
                )
            self._templates[template.name] = template
            logger.info(
                "prompt_loaded",
                name=template.name,
                version=template.version,
                file=str(yaml_file),
            )
            count += 1
        logger.info("prompts_loaded", total=count)

    def get(self, name: str) -> PromptTemplate:
        """按名称获取模板。如果未找到，则引发KeyError。"""
        if name not in self._templates:
            available = ", ".join(sorted(self._templates.keys())) or "<none>"
            raise KeyError(f"Prompt template '{name}' not found. Available: {available}")
        return self._templates[name]

    def list_names(self) -> list[str]:
        return sorted(self._templates.keys())


@lru_cache
def get_prompt_registry() -> PromptRegistry:
    """获取应用程序范围的提示词注册表（缓存的单例）。"""
    registry = PromptRegistry()
    registry.load_all()
    return registry
