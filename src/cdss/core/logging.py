"""
Structured logging configuration using structlog.

Use this everywhere instead of the standard logging module:

    from cdss.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("event_name", key1=value1, key2=value2)

    
使用structlog的结构化日志配置。
在任何地方都使用它，而不是标准的日志模块：
从cdss.core.logging导入get_logger
logger=get_logger（__name__）
logger.info（“事件名称”，按键1=value1，按键2=value2）
"""
import logging
import sys

import structlog

from cdss.core.config import get_settings


def configure_logging() -> None:
    """
    Configure structlog. Call this once at application startup.

    In development: human-readable colored console output.
    In production: JSON output for log aggregation system.

    配置结构日志。在应用程序启动时调用一次。
    开发中：彩色控制台输出。
    在生产环境中：日志聚合系统的JSON输出。
    """
    settings = get_settings()
    log_level = getattr(logging, settings.log_level)

    # Configure standard library logging (used by libraries like uvicorn,sqlalchemy
    # 配置标准库日志记录（由uvicorn、sqlalchemy等库使用
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Shared processors for both dev and prod
    # 开发和生产共享处理器
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.is_development:
        # Pretty-printed, colored console output
        # 印刷精美，彩色控制台输出
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # JSON output for production
        processors = shared_processors + [
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger. Use module's __name__ as the name.

    Example:
        logger = get_logger(__name__)
        logger.info("user_login", user_id=123, ip="1.2.3.4")
    """
    return structlog.get_logger(name)