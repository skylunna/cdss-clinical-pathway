"""
Smoke test: insert a random-vector chunk, read it back, run a similarity query.

Usage:
    uv run python scripts/test_db.py
"""
import asyncio
import random

from sqlalchemy import select

from cdss.core.logging import configure_logging, get_logger
from cdss.db.session import get_session_factory
from cdss.models.knowledge import EMBEDDING_DIM, KnowledgeChunk


logger = get_logger(__name__)


def random_vector() -> list[float]:
    """一个随机的单位向量——仅用于管道测试。"""
    return [random.uniform(-1, 1) for _ in range(EMBEDDING_DIM)]

async def main() -> None:
    configure_logging()
    factory = get_session_factory()


    # --- Insert ---
    async with factory() as session:
        chunk = KnowledgeChunk(
            content="这是一个测试用 chunk(随机向量),用于验证数据库链路。",
            source="smoke-test",
            section="N/A",
            chunk_index=0,
            embedding=random_vector(),
        )
        session.add(chunk)
        await session.commit()
        await session.refresh(chunk)
        logger.info("inserted_chunk", id=chunk.id, source=chunk.source)
        inserted_id = chunk.id 

    # --- Read back ---
    async with factory() as session:
        result = await session.execute(
            select(KnowledgeChunk).where(KnowledgeChunk.id == inserted_id)
        )
        row = result.scalar_one()
        logger.info(
            "read_back",
            id=row.id,
            content=row.content,
            embedding_dim=len(row.embedding),
        )

    # --- Similarity search (random query vector) ---
    async with factory() as session:
        query_vec = random_vector()
         # cosine distance via the <=> operator (pgvector)
        stmt = (
             select(
                 KnowledgeChunk,
                 KnowledgeChunk.embedding.cosine_distance(query_vec).label("distance"),
             )
             .order_by("distance")
             .limit(3)
         )
        result = await session.execute(stmt)
        rows = result.all()
        for chunk_obj, distance in rows:
            logger.info(
                "similarity_result",
                id=chunk_obj.id,
                distance=round(distance, 4),
                source=chunk_obj.source,
            )
    
        # ---- Cleanup ----
    async with factory() as session:
        await session.execute(
            select(KnowledgeChunk).where(KnowledgeChunk.source == "smoke-test")
        )
        # Re-fetch and delete (simpler than ad-hoc delete syntax here)
        result = await session.execute(
            select(KnowledgeChunk).where(KnowledgeChunk.source == "smoke-test")
        )
        for c in result.scalars().all():
            await session.delete(c)
        await session.commit()
        logger.info("cleaned_up_smoke_test_rows")


if __name__ == "__main__":
    asyncio.run(main())