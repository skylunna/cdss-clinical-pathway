"""
Repository for KnowledgeChunk persistence operations.

The repository pattern separates "how data is stored" from "what we do with it".
Higher layers (ingestion, retrieval) depend on these methods, not on raw SQLAlchemy.

知识块持久化操作的存储库。
仓库模式将“数据的存储方式”与“我们对数据的操作”分开。  
高层模块（如数据摄入、检索）依赖于这些方法，而不是原始的 SQLAlchemy。
"""
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from cdss.models.knowledge import KnowledgeChunk


class KnowledgeChunkRepository:
    """Persistence operations for KnowledgeChunk."""
    # 知识块的持久化操作。

    def __init__(self, session: AsyncSession) -> None:
        # 外部传入已创建好的异步数据库会话，存为私有成员 _session
        # 整个仓储类服用同一个会话，不用每次方法新建连接
        # 符合依赖注入规范，配合 get_session 依赖使用
        self._session = session

    async def add_all(self, chunks: list[KnowledgeChunk]) -> None:
        """一次性插入多个片段。"""
        # 批量插入多条知识库 chunk 记录
        if not chunks:
            return
        self._session.add_all(chunks) # 批量把ORM对象加入会话待提交队列
        # 立刻把数据发送到数据库执行 insert，但不commit提交事务
        # 真正提交事物交给外层get_session的自动 commit 逻辑
        await self._session.flush()

    async def delete_by_source(self, source: str) -> int:
        """Delete all chunks for a given source. Returns number deleted."""
        # 删除指定源的所有片段。返回已删除的片段数量。
        # 删除某一份文档全部分片 （例如重新导入指南前清空旧数据）
        # 构造 DELETE FROM knowledge_chunks WHERE source = xxx;
        # result.rowcount 返回本次删除的行数，无删除返回 0
        # 适用场景：重复上传同一PDF/指南，先删旧数据再新增
        result = await self._session.execute(
            select(func.count(KnowledgeChunk.id)).where(
                KnowledgeChunk.source == source
            )
        )
        return int(result.scalar() or 0)
    
    async def count_by_source(self, source: str) -> int:
        # SELECT count(id) FROM knowledge_chunks WHERE source = ?
        # func.count: SQLALchemy 内置聚合计数函数
        # .scalar() 直接取出单行单列数字，用途：导入前判断该文档是否已存在、有多少分片
        result = await self._session.execute(
            select(func.count(KnowledgeChunk.id)).where(
                KnowledgeChunk.source == source
            )
        )
        return int(result.scalar() or 0)

    async def count_all(self) -> int:
        result = await self._session.execute(
            select(func.count(KnowledgeChunk.id))
        )
        return int(result.scalar() or 0)

    async def list_sources(self) -> list[str]:
        result = await self._session.execute(
            select(KnowledgeChunk.source).distinct().order_by(KnowledgeChunk.source)
        )
        return list(result.scalars().all())