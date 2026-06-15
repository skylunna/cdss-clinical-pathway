"""
Async database session management (SQLAlchemy 2.0 style).

Provides:
- engine:               application-wide async DB engine (singleton)
- AsyncSessionLocal:    session factory bound to that engine
- get_session():        FastAPI dependency that yields a session per request

Usage in routes:
    async def my_route(session: AsyncSession = Depends(get_session)):
        result = await session.execute(...)


异步数据库会话管理（SQLAlchemy 2.000 风格）。
提供：
    - engine：全局异步数据库引擎（单例）
    - AsyncSessionLocal：绑定到该引擎的会话工厂
    - get_session()：FastAPI 依赖项，每次请求返回一个会话
在路由中的使用：
    async def my_route(session: AsyncSession = Depends(get_session)):
        result = await session.execute(...)
"""
from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from cdss.core.config import get_settings
from cdss.core.logging import get_logger


logger = get_logger(__name__)


@lru_cache
def get_engine() -> AsyncEngine:
    """获取全局异步引擎（缓存单例）。"""
    settings = get_settings()
    engine = create_async_engine(
        settings.database_url,
        echo=False,
        pool_pre_ping=True, # 避免数据库重启后出现过时的连接
        pool_size=5,
        max_overflow=10,
    )
    logger.info("db_engine_created", url=_sanitize(settings.database_url))
    return engine

@lru_cache
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """获取与应用程序引擎绑定的会话工厂。"""
    return async_sessionmaker(
        bind=get_engine(),
        # 提交后对象仍可使用
        expire_on_commit=False,
        class_ = AsyncSession,
    )

async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖：返回会话，成功时提交，出错时回滚。"""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

def _sanitize(url: str) -> str:
    """从数据库URL中移除密码以用于日志记录。"""
    # postgresql+asyncpg://user:pass@host -> postgresql+asyncpg://user:***@host
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    if "@" not in rest:
        return url
    creds, host = rest.rsplit("@", 1)
    if ":" in creds:
        user, _ = creds.split(":", 1)
        creds = f"{user}:***"
    return f"{scheme}://{creds}@{host}"