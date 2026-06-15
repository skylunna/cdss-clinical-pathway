"""
Database schema initialization.

For this learning project we use Base.metadata.create_all() rather than
Alembic migrations — simpler, but does NOT handle schema changes
(altering an existing table). Switch to Alembic before production.



数据库模式初始化。

我们使用 Base.metadata.create_all() 而不是 Alembic 迁移——更简单，
但不支持对现有表的结构变更。生产环境前切换至 Alembic。
"""
from cdss.core.logging import get_logger
from cdss.db.base import Base
from cdss.db.session import get_engine

# 导入所有模型模块，以便它们的表能够注册到 Base.metadata 中。  
# 这一点很重要：如果没有这些导入，create_all()() 将无法识别它们。
from cdss.models import knowledge   # noqa: F401


logger = get_logger(__name__)

async def init_db() -> None:
    """创建 Base.metadata 中定义的所有表（幂等性）。"""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("db_schema_initialized", tables=list(Base.metadata.tables.keys()))