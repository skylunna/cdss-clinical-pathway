"""
Smoke test: semantic retrieval over the ingested knowledge base.

For each test query, find the top-K most similar chunks and print them.
This verifies the end-to-end RAG retrieval path works before we build
the actual retrieval API (next step).

Usage:
    uv run python scripts/test_retrieval.py

烟雾测试：对摄入的知识库进行语义检索。
对于每个测试查询，找出最相似的前K个片段并打印出来。  
这可以在构建实际的检索API（下一步）之前，验证端到端的RAG检索路径是否正常工作。
使用方法：
    uv run python scripts/test_retrieval.py
"""
import asyncio

from sqlalchemy import select


from cdss.core.logging import configure_logging, get_logger
from cdss.db.session import get_session_factory
from cdss.models.knowledge import KnowledgeChunk
from cdss.rag.embedder import get_embedder


logger = get_logger(__name__)


async def search(query: str, k: int = 3) -> None:
    embedder = get_embedder()
    query_vec = await embedder.aembed_one(query)

    factory = get_session_factory()
    async with factory() as session:
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

    print(f"\n{'=' * 80}")
    print(f"Query: {query}")
    print(f"{'=' * 80}")
    for i, (chunk, dist) in enumerate(rows, 1):
        print(f"\n[{i}] distance={dist:.4f}  section={chunk.section}")
        print(f"    {chunk.content[:200]}...")


async def main() -> None:
    configure_logging()

    queries = [
        "重症肺炎应该用什么抗生素",
        "青霉素过敏的患者怎么选抗生素",
        "什么是 CURB-65 评分",
        "什么时候考虑铜绿假单胞菌感染",
        "肺炎治疗多少天能停药",
        # 一个非主题的查询，用于检查它是否还能检索到内容（会检索到，但距离会更远，且内容不相关）
        "番茄炒蛋的做法",
    ]

    for q in queries:
        await search(q, k=3)


if __name__ == "__main__":
    asyncio.run(main())