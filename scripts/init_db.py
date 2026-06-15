"""
One-time database initialization.

Usage:
    uv run python scripts/init_db.py

一次性数据库初始化。
"""
import asyncio

from cdss.core.logging import configure_logging
from cdss.db.init import init_db


async def main() -> None:
    configure_logging()
    await init_db()


if __name__ == "__main__":
    asyncio.run(main())