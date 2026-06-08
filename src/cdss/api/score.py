"""
Scoring endpoints — exposes deterministic clinical scoring rules.

These endpoints are thin wrappers over the pure functions in cdss.rules.
The HTTP layer adds nothing semantic; it just makes the rule callable
via REST (useful for demos and as a target for future LLM tool-calling).

评分端点 — 提供确定性的临床评分规则。
这些端点是 cdss.rules 中纯函数的轻量级封装。  
HTTP 层并未增加任何语义内容，仅使规则可通过 REST 调用（适用于演示，也可作为未来调用 LLM 工具的目标）。
"""
from fastapi import APIRouter

from cdss.rules.scoring import curb65


router = APIRouter(prefix="/api/v1/score", tags=["scoring"])


@router.post(
    "/curb65",
    response_model=curb65.CURB65Result,
    summary="Compute CURB-65 pneumonia severity score",
)
async def score_curb65(data: curb65.CURB65Input) -> curb65.CURB65Result:
    """Compute CURB-65 score from patient parameters"""
    return curb65.compute(data)