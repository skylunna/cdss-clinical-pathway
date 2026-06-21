"""
Document ingestion pipeline.

Takes raw text or markdown files, runs them through:
    chunk -> embed -> persist (idempotent per source)

Idempotency: re-running ingest on the same source replaces all its chunks.
This makes "fix the source doc and re-ingest" a safe, repeatable operation.

文档导入流水线。
接收原始文本或 Markdown 文件，经过以下处理流程：
分块 → → 内嵌 → → → 持久化（对每个源文件具有幂等性）
幂等性：对同一数据源重新执行导入操作会替换其所有数据块。  
这使得“修复源文档并重新导入”成为一种安全且可重复的操作
"""
from pathlib import Path

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import async_sessionmaker

from cdss.core.logging import get_logger
from cdss.models.knowledge import KnowledgeChunk
from cdss.rag.chunker import MarkdownChunker
from cdss.rag.embedder import Embedder
from cdss.rag.repository import KnowledgeChunkRepository


logger = get_logger(__name__)


class IngestionResult(BaseModel):
    """单个文档导入摘要。"""

    source: str
    chunks_added: int
    chunks_replaced: int


class IngestionPipeline:
    """将数据分块 -> 内嵌 -> 持久化"""

    def __init__(
            self,
            chunker: MarkdownChunker,
            embedder: Embedder,
            session_factory: async_sessionmaker,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._session_factory = session_factory

    async def ingest_text(self, text: str, source: str) -> IngestionResult:
        """根据文本和逻辑源名称导入单个文档。"""
        # 1. Chunk
        chunk_datas = self._chunker.chunk(text, source)
        logger.info("chunked", source=source, num_chunks = len(chunk_datas))

        if not chunk_datas:
            logger.warning("empty_document", source=source)
            return IngestionResult(source=source, chunks_added=0, chunks_replaced=0)
        
        # 2. embed (batched, async via thread pool)
        contents = [c.content for c in chunk_datas]
        embeddings = await self._embedder.aembed_batch(contents)
        logger.info("embedded", source=source, num_vectors=len(embeddings))

        # 3. build ORM rows
        orm_chunks = [
            KnowledgeChunk(
                content=cd.content,
                source=cd.source,
                section=cd.section,
                chunk_index=cd.chunk_index,
                embedding=emb,
            )
            for cd, emb in zip(chunk_datas, embeddings, strict=True)
        ]

        # 4. 持续幂等（先删除此源的现有内容，再插入）
        async with self._session_factory() as session:
            repo = KnowledgeChunkRepository(session)
            replaced = await repo.delete_by_source(source)
            await repo.add_all(orm_chunks)
            await session.commit()

        logger.info(
            "ingested",
            source=source,
            chunks_added=len(orm_chunks),
            chunks_replaced=replaced,
        )

        return IngestionResult(
            source=source,
            chunks_added=len(orm_chunks),
            chunks_replaced=replaced,
        )
    
    async def ingest_file(self, path: Path) -> IngestionResult:
        """输入一个 Markdown/文本文件。来源 = 文件名"""
        text = path.read_text(encoding="utf-8")
        return await self.ingest_text(text, source=path.name)
    
    async def ingest_directory(
            self,
            directory: Path,
            pattern: str = "*.md",
    ) -> list[IngestionResult]:
        """将目录中的所有匹配文件进行导入。"""
        results: list[IngestionResult] = []
        for path in sorted(directory.glob(pattern)):
            results.append(await self.ingest_file(path))
        return results