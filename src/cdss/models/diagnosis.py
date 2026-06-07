"""
Domain models for diagnostic reasoning.

These Pydantic models serve a triple role:
1. LLM structured output schema (passed to structured_chat)
2. API response model (returned from FastAPI endpoints)
3. Internal data exchange between application modules

The same model used as both LLM schema and API schema ensures consistency.

用于诊断推理的领域模型。
这些 Pydantic 模型具有三重作用：
    1. LLM 结构化输出方案（传递给 structured_chat）
    2. API响应模型（来自FastAPI端点返回）
    3. 应用程序模块之间的内部数据交换
使用相同的模型作为LLM架构和API架构，以确保一致性。
"""
from typing import Literal

from pydantic import BaseModel, Field


ConfidenceLevel = Literal["VERY_HIGH", "PROBABLE", "POSSIBLE", "UNLIKELY"]


class Diagnosis(BaseModel):
    """单一鉴别诊断，有支持性证据"""

    name: str = Field(
        description="规范医学诊断名称(中文, 如: 社区获得性肺炎)",
    )
    confidence: ConfidenceLevel = Field(
        description="置信度: 高 / 中 / 低",
    )
    supporting_evidence: list[str] = Field(
        min_length=1,
        description="从患者信息中提取的、支持此诊断的具体证据(每条1-2句话)",
    )
    discriminating_tests: list[str] = Field(
        default_factory=list,
        description="为确认或排除此诊断，建议补充的检查或信息"
    )


class DiagnosisRecommendation(BaseModel):
    """基于患者信息的诊断推理结构化输出。"""

    key_findings: list[str] = Field(
        min_length=1,
        description="从患者信息中提炼的关键临床发现(症状、体征、辅检异常)",
    )
    diagnoses: list[Diagnosis] = Field(
        min_length=1,
        max_length=5,
        description="按可能性从高到低排序的鉴别诊断列表",
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="如有关键信息缺失需补充, 在此列出(可选)"
    )