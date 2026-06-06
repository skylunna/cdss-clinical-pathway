"""
Application configuration loaded from environment variables and .env file.

Uses pydantic-settings for type-safe configuration management.
All configuration should be accessed through the `get_settings()` function.

从环境变量和.env文件加载的应用程序配置。使用pydantic设置进行类型安全配置管理。
所有配置都应该通过`get_settings()`函数访问。
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root is 3 levels up from this file: src/cdss/core/config.py
_PROJECT_ROOT = Path(__file__).parents[3]

class Settings(BaseSettings):
    """Application settings. Auto-loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore", #忽略 .env 里多余的变量, 避免报错
    )

    # Application
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # LLM Provider Selection
    default_llm_provider: Literal["deepseek", "openai", "qwen"] = "deepseek"
    default_llm_model: str = "deepseek-chat"

    # LLM API Keys (all optional for now, validated on actual use)
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"

    # Database (used later)
    database_url: str = Field(
        default="postgresql+asyncpg://cdss_user:cdss_pass@localhost:5432/cdss_db",
        description="PostgreSQL connection string (with asyncpg driver)",
    )

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

@lru_cache
def get_settings() -> Settings:
    """
    Get application settings (cached).

    Use this function everywhere instead of instantiating Settings() directly,
    so the .env file is only parsed once per process.

    获取应用程序设置（缓存）。
    在任何地方都使用此函数，而不是直接实例化Settings()，
    因此，每个进程只解析一次.env文件。
    """
    return Settings()

