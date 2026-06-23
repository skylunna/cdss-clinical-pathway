"""
Knowledge search tool — wraps the Retriever as an LLM-callable tool.

The tool's description is critical: it tells the LLM WHEN to call it
(distinguishing from rule-based tools) and HOW to query (focused phrases,
not full patient cases). This is a real pattern of prompt engineering at
the tool-definition level.

知识搜索工具——将Retriever封装为可调用LLM的工具。
工具的描述至关重要：它告诉大语言模型何时调用该工具（与基于规则的工具不同），
以及如何查询（使用聚焦短语，而非完整的患者案例）。这是在工具定义层面进行提示工程的一个真实模式。
"""
from pydantic import BaseModel, Field

from cdss.rag.retriever import RetrievedChunk, get_retriever
from cdss.tools.base import Tool
from cdss.tools.registry import ToolRegistry


class SearchClinicalKnowledgeInput(BaseModel):
    query: str = Field(
        min_length=2,
        description=(
            "A SHORT, FOCUSED clinical question or topic phrase. "
            "Do NOT paste the entire patient case. "
            "Good examples: 'CURB-65 评分标准', '青霉素过敏的替代抗生素', "
            "'重症CAP抗生素疗程'. "
            "Bad example: pasting a multi-paragraph patient history."
        ),
    )
    k: int = Field(
        default=3,
        ge=1,
        le=8,
        description="Number of top relevant chunks to return",
    )


class SearchClinicalKnowledgeResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    note: str | None = None


async def _handler(
        data: SearchClinicalKnowledgeInput,
) -> SearchClinicalKnowledgeResult:
    """Tool handler: delegates to the retriever and shapes the result."""
    # 工具处理器：将请求委托给检索器，并对结果进行格式化
    retriever = get_retriever()
    chunks = await retriever.search(data.query, k=data.k)
    note = None
    if not chunks:
        note = (
            "知识库中未找到足够相关的内容。请基于通用临床知识回答,"
            "并在最终回答中说明'此结论未在知识库中找到直接依据'。"
        )
    return SearchClinicalKnowledgeResult(query=data.query, chunks=chunks, note=note)


search_clinical_knowledge_tool = Tool(
    name="search_clinical_knowledge",
    description=(
        "Search the clinical knowledge base (guidelines, treatment protocols, "
        "clinical references) for information relevant to a focused query. "
        "Use this tool when you need to:\n"
        "- Look up specific clinical criteria, definitions, or recommendations\n"
        "- Answer 'what is', 'why', 'how long' style knowledge questions\n"
        "- Verify or cite specific clinical details before answering\n"
        "\n"
        "Pass a SHORT, FOCUSED query (a topic or question phrase, "
        "NOT the whole patient case). Returns top-K relevant text chunks, "
        "each with `source` and `section` fields you SHOULD cite in your "
        "final answer."
    ),
    input_schema=SearchClinicalKnowledgeInput,
    handler=_handler,
)

def register_all(registry: ToolRegistry) -> None:
    registry.register(search_clinical_knowledge_tool)