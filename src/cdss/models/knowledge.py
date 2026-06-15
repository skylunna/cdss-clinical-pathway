"""
Knowledge base ORM models.

A `KnowledgeChunk` is a single retrievable unit of text from a knowledge
source (a guideline, textbook, etc.), with its embedding vector for
similarity search.

知识库 ORM 模型。
`KnowledgeChunk` 是从知识源（如指南、教科书等）中提取的一个可检索的文本单元，
包含用于相似度搜索的嵌入向量。
"""
from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from cdss.db.base import Base


# Must match the embedding model's output dimension.
# 必须与嵌入模型的输出维度匹配。
# bge-small-zh-v1.5 produces 512-dim vectors.
# bge-small-zh-v1.5 生成的是 512 维向量。
EMBEDDING_DIM = 512

class KnowledgeChunk(Base):
    """一段可检索的知识片段及其嵌入内容。"""

    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    content: Mapped[str] = mapped_column(
        Text, nullable=False, comment="The chunk text"
    )

    source: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Source document identifier, e.g., 'cap-guideline-2016.md'",
    )

    section: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Optional section path, e.g., 'Treatment > Severe CAP'",
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="0-based index of this chunk within the source document",
    )

    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIM),
        nullable=False,
        comment=f"{EMBEDDING_DIM}-dim embedding vector (model: bge-small-zh-v1.5)",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    def __repr__(self) -> str:
        preview = self.content[:30].replace("\n", " ")
        return (
            f"<KnowledgeChunk id={self.id} source={self.source!r} "
            f"chunk={self.chunk_index} content={preview!r}...>"
        )