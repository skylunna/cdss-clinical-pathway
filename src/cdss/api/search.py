"""
Direct retrieval endpoint.

Exposes the knowledge base search as a plain HTTP API,
independent of the Agent loop. Useful for:
- Debugging retrieval quality
- Building a search UI
- Testing distance thresholds

直接检索端点。
将知识库搜索暴露为一个普通的 HTTP API，独立于代理循环。适用于：
        -- 调试检索质量
        -- 构建搜索用户界面
        -- 测试距离阈值
"""
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from cdss.rag.retriever import RetrievedChunk, Retriever, get_retriever


router = APIRouter(prefix="/api/v1", tags=["search"])


class SearchResponse(BaseModel):
    """约束接口返回JSON结构，格式固定。
    FastAPI 根据 response_model 自动过滤多余字段、生成接口文档
    前端固定解析字段，不用兼容各种不规则返回值"""
    query: str          # 用户原始查询词
    num_returned: int   # 过滤后返回的分片数量
    chunks: list[RetrievedChunk]    # 检索到的分片列表


@router.get("/search", response_model=SearchResponse)
# GET 请求接口，路径拼劲前缀完整为 /api/v1/search
# 强制返回值按照该  Pydantic 模型序列化，自动生成 OpenAPI 文档
async def search_knowledge(
    # ... 必填参数；限制最少输入2个字，防止空/单字无意义检索
    # 自动在校验失败时返回友好报错给前端
    q: str = Query(..., min_length=2, description="Search query"), 
    # 默认取前5条；限制范围 1 ～ 10，防止前端传超大值压垮数据库
    k: int = Query(default=5, ge=1, le=10),
    max_distance: float = Query(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Distance threshold: results with distance > this are filtered out"
    ),
    # FastAPI 会自动执行 get_retriever() 依赖注入
    # 因为函数加了 @lru_cache, 全局只会初始化一次模型与数据库工厂
    # 所有接口请求公用一个 Retriever 单列，不会重复加载 BGE 大模型
    # 解耦：接口不用手动创建检索器，框架自动注入
    retriever: Retriever = Depends(get_retriever),
) -> SearchResponse:
    """对知识库进行语义搜索。"""
    # 调用检索器异步语义检索，自动完成：向量化 -> pgvector HNSW 索引找回 -> 距离过滤
    # 把原始查询、返回条数、分片列表包装成标准SearchResponse返回JSON
    chunks = await retriever.search(q, k=k, max_distance=max_distance)
    return SearchResponse(query=q, num_returned=len(chunks), chunks=chunks)