"""
__init__.py

Copyright (c) 2026 skylunna

Author      : skylunna <tianlunnn@gmail.com>
Created     : 2026-06-03 21:22:30
Modified    : 2026-06-03 21:22:30

Description : FastAPI应用入口
FastAPI application entry point.

Run with:
    uv run uvicorn cdss.main:app --reload

"""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from cdss.core.config import get_settings
from cdss.core.logging import get_logger, configure_logging
from cdss.api.chat import router as chat_router
from cdss.api.diagnose import router as diagnose_router
from cdss.api.score import router as score_router
from cdss.api.agent import router as agent_router
from cdss.api.search import router as search_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown logic."""
    # === startup ===
    configure_logging()  # 配置日志
    settings = get_settings()  # 读取日志
    logger = get_logger(__name__)  # 创建日志 (带文件名)
    logger.info(
        "application_starting",  # 日志事件名
        env=settings.app_env,
        log_level=settings.log_level,
        default_llm_provider=settings.default_llm_provider,
    )

    yield  # Application is running

    # === shutdown ===
    logger.info("application_shutdown")


app = FastAPI(
    title="CDSS Clinical Pathway",
    description="Agent-based Clinical Decision Support System with clinical pathway navigation",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(chat_router)
app.include_router(diagnose_router)
app.include_router(score_router)
app.include_router(agent_router)
app.include_router(search_router)


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """
    Liveness probe. Returns 200 if the application is running.

    Use this for Kubernetes/Docker health checks and basic monitoring.
    """
    return {"status": "ok"}


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    """Root endpoint with basic info."""
    settings = get_settings()
    return {
        "name": "CDSS Clinical Pathway",
        "version": "0.1.0",
        "env": settings.app_env,
        "docs": "/docs",
    }
