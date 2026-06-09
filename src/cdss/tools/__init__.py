"""
__init__.py

Copyright (c) 2026 skylunna

Author      : skylunna <tianlunnn@gmail.com>
Created     : 2026-06-03 21:22:30
Modified    : 2026-06-03 21:22:30

Description : Function Calling 工具集

工具子系统。
提供注册表单例，并在启动时注册所有内置工具。

"""
from functools import lru_cache

from cdss.core.logging import get_logger
from cdss.tools import clinical_scoring
from cdss.tools.registry import ToolRegistry


logger = get_logger(__name__)


@lru_cache
def get_tool_registry() -> ToolRegistry:
    """获取全局工具注册表（缓存单例）。"""
    registry = ToolRegistry()
    clinical_scoring.register_all(registry)
    logger.info(
        "tool_registry_initialized",
        tools=[t.name for t in registry.list_all()],
    )
    return registry