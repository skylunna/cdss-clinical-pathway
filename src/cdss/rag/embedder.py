"""
Embedding service abstraction.

An `Embedder` converts text into fixed-dimension vectors suitable for
similarity search. Subclasses adapt different providers (local models,
API services) to a uniform interface.

Concurrency note:
- Sync methods (`embed_one` / `embed_batch`) are CPU/GPU bound and block.
- Async methods delegate to a worker thread via asyncio.to_thread(),
  so calling them from FastAPI routes will not block the event loop.


嵌入服务抽象。
`Embedder` 用于将文本转换为适合相似度搜索的固定维度向量。
其子类通过统一接口适配不同的提供者（本地模型、API 服务）。
并发说明：
    -- 同步方法（`embed_one` / `embed_batch`）是 CPU/GPU 限制的，并会阻塞。
    -- 异步方法通过 `asyncio.to_thread()`()`()` 转发给工作线程，
        因此在 FastAPI路由中调用这些方法不会阻塞事件循环。
"""
import asyncio
from abc import ABC, abstractmethod


class Embedder(ABC):
    """Abstract base for text embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """输出向量维度。必须与数据库模式中的向量(N)列匹配。"""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """嵌入模型的稳定标识符（用于追踪来源）。"""

    @abstractmethod
    def embed_one(self, text: str) -> list[float]:
        """嵌入单个文本。同步。"""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """嵌入多个文本。同步执行。实现应内部批量处理。"""

    # --- 异步包装器（委托给线程以避免阻塞事件循环） ---
    async def aembed_one(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_one, text)
    
    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_batch, texts)
    

class BGEEmbedder(Embedder):
    """
    嵌入器由本地句子转换模型（BGE系列）提供支持。
        默认模型：BAAI/bge-small-zh-v1.5（512维，中文微调）。
        首次使用时从Hugging Face Hub下载该模型。
        （如需使用中国大陆镜像，请设置HF_ENDPOINT=https://hf-mirror.com）。
        嵌入向量经过L2归一化，使得点积等于余弦相似度。  
        这与pgvector中 <=> 运算符的使用方式一致
    """

    MODEL_NAME = "BAAI/bge-small-zh-v1.5"
    IMENSION = 512

    def __init__(self) -> None:
        # 懒加载：避免将 torch 引入每个 cdss.*导入路径，
        # 从而防止例如不涉及 RAG 的单元测试变慢。
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.MODEL_NAME)

        @property
        def dimension(self) -> int:
            return self.DIMENSION
        
        @property
        def model_name(self) -> str:
            return self.MODEL_NAME
        
        def embed_one(self, text: str) -> list[float]:
            # encode() 返回一个 numpy 数组；为 Pydantic/JSON 安全起见，转换为普通列表
            vec = self._model.encode(text, normalize_embeddings=True)
            return vec.tolist()
        
        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            if not texts:
                return []
            matrix = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
            return matrix.tolist()
        
