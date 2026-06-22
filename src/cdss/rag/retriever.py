"""
Semantic retrieval service over the knowledge_chunks table.

Embeds a query, runs cosine similarity search via pgvector's <=> operator,
and applies a distance threshold so unrelated queries return [] instead
of confusing the caller with weakly-related chunks.

基于知识块表的语义检索服务。
嵌入查询，通过 pgvector 的 <=> 运算符执行余弦相似度搜索，并设置距离阈值，
使无关查询返回空数组，而非让调用者看到关联性较弱的片段。
"""
from functools import lru_cache

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker


from cdss.core.logging import get_logger
from cdss.db.session import get_session_factory
from cdss.models.knowledge import KnowledgeChunk
from cdss.rag.embedder import Embedder, get_embedder


logger = get_logger(__name__)


# Default distance threshold tuned to BGE-small-zh-v1.5 + this domain.
# 默认距离阈值已调优至 BGE-small-zh-v1.5 + 该领域。
# Adjust based on retrieval evaluation when you swap embedding models.
# 在更换嵌入模型时，请根据检索评估结果进行调整。
# distance <= 0.5: 判定相关，保留
# distance > 0.5: 语义无关，直接丢弃
# 换其他embedding模型 (m3e/jina) 需要重新调这个阈值
DEFAULT_MAX_DISTANCE = 0.5

class RetrievedChunk(BaseModel):
    """检索返回的块，以及其与查询的距离。"""

    content: str    # 分片原文
    source: str     # 文档来源文件名
    section: str | None # Markdown 层级章节路径，用于溯源原文位置
    chunk_index: int    # 统一文档内分片序号
    # 查询向量与该分片向量的余弦距离，Field仅用于 FastAPI 接口文档展示说明，无数值校验
    distance: float = Field(description="查询的余弦距离（0=完全相同，越小越好）")




class Retriever:

    def __init__(
            self,
            embedder: Embedder,
            session_factory: async_sessionmaker,
    ) -> None:
        # 依赖注入 两个依赖，外部传入，解耦: 
        self._embedder = embedder   # 向量编码器 (BGEEmbedder)，用来把文字转向量；
        self._session_factory = session_factory # 数据库会话工厂，每次检索创建异步会话；
        # 好处：单元测试时可以传入 Mock Embedder / Mock 会话，不依赖真实模型与数据库

    async def search(
            self,
            query: str, # 用户提问文本  
            k: int = 5, # 最多取前 k 个最相似分片 (默认5)
            max_distance: float = DEFAULT_MAX_DISTANCE, # 距离过滤阈值，默认0.5
    ) -> list[RetrievedChunk]:
        """
        返回与目标最相似的前K个片段，按最大距离过滤。
        策略：先获取前K个结果（通过HNSW索引高效），然后按距离进行过滤。
        如果大量结果被过滤掉，这种方法会稍显浪费，但简单且正确。
        对于高并发使用场景，可将过滤条件通过WHERE子句推入SQL中。
        """
        # 调用嵌入模型异步编码，得到 512 维浮点向量
        query_vec = await self._embedder.aembed_one(query)

        # 异步数据库查询 pgvector 余弦检索
        async with self._session_factory() as session:
            stmt = (
                select(
                    KnowledgeChunk,
                    KnowledgeChunk.embedding.cosine_distance(query_vec).label("distance"),
                )
                .order_by("distance")
                .limit(k)
            )
            result = await session.execute(stmt)
            rows = result.all()

        retrieved = [
            RetrievedChunk(
                content=chunk.content,
                source=chunk.source,
                section=chunk.section,
                chunk_index=chunk.chunk_index,
                distance=float(distance),
            )
            for chunk, distance in rows
            if distance <= max_distance
        ]

        logger.info(
            "retrieval_done",
            query=query[:60],
            k=k,
            returned=len(retrieved),
            filtered_out=len(rows) - len(retrieved),
        )
        return retrieved
    

@lru_cache
def get_retriever() -> Retriever:
    """获取全局检索器（缓存的单例）。"""
    return Retriever(
        embedder=get_embedder(),
        session_factory=get_session_factory(),
    )