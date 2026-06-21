"""
Knowledge base ingestion CLI.

Usage:
    uv run python scripts/ingest_knowledge.py            # ingest data/raw/ default
    uv run python scripts/ingest_knowledge.py path/to/dir
"""
import asyncio
import sys
from pathlib import Path


from cdss.core.logging import configure_logging, get_logger
from cdss.db.session import get_session_factory
from cdss.rag.chunker import MarkdownChunker
from cdss.rag.embedder import get_embedder
from cdss.rag.ingestion import IngestionPipeline



logger = get_logger(__name__)

DEFAULT_DATA_DIR = Path("data/raw")


async def main() -> None:
    configure_logging()

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_DIR
    if not data_dir.exists():
        logger.error("data_dir_missing", path=str(data_dir))
        sys.exit(1)

    pipeline = IngestionPipeline(
        chunker=MarkdownChunker(target_size=400, overlap=50),
        embedder=get_embedder(),
        session_factory=get_session_factory(),
    )

    logger.info("ingestion_start", data_dir=str(data_dir))
    results = await pipeline.ingest_directory(data_dir)
    total_added = sum(r.chunks_added for r in results)
    total_replaced = sum(r.chunks_replaced for r in results)
    logger.info(
        "ingestion_complete",
        files=len(results),
        total_chunks_added=total_added,
        total_chunks_replaced=total_replaced,
    )

if __name__ == "__main__":
    asyncio.run(main())