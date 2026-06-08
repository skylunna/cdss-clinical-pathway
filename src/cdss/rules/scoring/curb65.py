"""
CURB-65 pneumonia severity score.

A clinical prediction rule used to assess severity of community-acquired
pneumonia (CAP) and guide site-of-care decisions (outpatient / ward / ICU).

Reference: Lim WS, et al. Defining community-acquired pneumonia severity
on presentation to hospital: an international derivation and validation study.
Thorax 2003;58:377-382.

Criteria (1 point each):
    C  - Confusion (new-onset altered mental status)
    U  - Urea > 7 mmol/L  (or BUN > 19 mg/dL)
    R  - Respiratory rate >= 30 /min
    B  - Blood pressure: systolic < 90  OR  diastolic <= 60 mmHg
    65 - Age >= 65 years

Score interpretation:
    0-1: Low risk        -> outpatient treatment usually appropriate
    2:   Intermediate    -> consider hospitalization
    3-5: Severe          -> hospitalize (consider ICU at 4-5)

Design notes:
- Pure function: same input always produces same output, no side effects
- Returns *per-criterion detail* (not just total score), so downstream
  consumers (LLM, audit log, UI) can show *why* the score is what it is
- Missing urea is handled explicitly with a warning, rather than silently
  treating it as zero


CURB-65肺炎严重程度评分。
    用于评估社区获得性肺炎（CAP）严重程度并指导治疗场所选择（门诊/病房/ICU）的临床预测规则。
    参考文献：Lim WS 等。定义社区获得性肺炎在入院时的严重程度：一项国际衍生与验证研究。胸科学报 2003;58:377-382。
标准（每项1分）：
    C - 混乱（新发意识障碍）
    U - 尿素 > 7 mmol/L（或BUN > 19 mg/dL）
    R - 呼吸频率 ≥ 30 次/分钟
    B - 血压：收缩压 < 90 mmHg 或 舒张压 ≤ 60 mmHg
    65 - 年龄 ≥ 65岁
评分解读：
    0-1：低风险 → 通常可门诊治疗
    2：中等风险 → 考虑住院
    3-5：严重风险 → 应住院（4-5级时考虑重症监护）
设计说明：  
    - 纯函数：相同输入始终产生相同输出，无副作用  
    - 返回按标准的详细信息（而不仅仅是总分），以便下游消费者（如LLM、审计日志、用户界面）能够显示评分为何如此  
    - 缺失尿素的情况会通过警告明确处理，而非静默地将其视为零
"""
from typing import Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["low", "intermediate", "severe"]


# --- Input / Output Models ---
class CURB65Input(BaseModel):
    """CURB-65评分所需输入信息"""

    age: int = Field(ge=0, le=150, description="患者年龄")
    confusion: bool = Field(
        description="新发意识障碍/精神状态改变（例如，AMTS ≤ 8）"
    )
    urea_mmol_l: float | None = Field(
        default=None,
        ge=0,
        # 尿素血清浓度（mmol/L）。如未测量则为0。
        description="Serum urea in mmol/L. None if not measured.",
    )
    respiratory_rate: int = Field(
        # 呼吸频率（每分钟呼吸次数）
        ge=0, le=80, description="Respiratory rate (breaths per minute)"
    )
    systolic_bp: int = Field(
        # 收缩压（mmHg）
        ge=0, le=300, description="Systolic blood pressure (mmHg)"
    )
    diastolic_bp: int = Field(
        # 舒张压（mmHg）
        ge=0, le=200, description="Diastolic blood pressure (mmHg)"
    )

class CURB65Criterion(BaseModel):
    """一个标准对得分的贡献。"""

    # 字母代码（C/U/R/B/65）
    code: str = Field(description="Letter code (C/U/R/B/65)")
    # 可读性标准描述
    description: str = Field(description="Human-readable criterion description")
    # 该标准是否满足（1分）
    met: bool = Field(description="Whether this criterion is satisfied (1 point)")
    # 审计/显示的观察值
    value_observed: str = Field(description="Observed value, for audit/display")

class CURB65Result(BaseModel):
    """CURB-65评分结果"""

    score: int = Field(ge=0, le=5)
    risk_level: RiskLevel
    # 按 标准 分类明细
    criteria: list[CURB65Criterion] = Field(description="Per-criterion breakdown")
    # 护理地点建议
    recommendation: str = Field(description="Site-of-care recommendation")
    # 数据质量警告（例如，输入缺失）
    warnings: list[str] = Field(
        default_factory=list,
        description="Data-quality warnings (e.g., missing inputs)",
    )


# --- Public Metadata (for future tool registry) ---
# 公共元数据（用于未来工具注册）
RULE_NAME = "curb65"
RULE_DESCRIPTION = (
    "计算社区获得性肺炎的严重程度CURB-65评分。  "
    " 返回总分（0-5）、风险等级、各标准项详细分解以及护理场所建议。"
)

# --- Core Logic ---
def compute(data: CURB65Input) -> CURB65Result:
    """Compute CURB-65 score from patient data. Pure function."""
    # 根据患者数据计算CURB-65评分。纯函数。
    criteria = [
        _eval_confusion(data),
        _eval_urea(data),
        _eval_respiratory_rate(data),
        _eval_blood_pressure(data),
        _eval_age(data),
    ]

    score = sum (1 for c in criteria if c.met)
    risk_level, recommendation = _interpret(score)

    warnings: list[str] = []
    if data.urea_mmol_l is None:
        warnings.append(
            # “尿素未提供；‘U’标准视为0。”  “评分可能低估了实际严重程度。”
            "Urea not provided; 'U' criterion treated as 0. "
            "Score may underestimate true severity."
        )

    return CURB65Result(
        score=score,
        risk_level=risk_level,
        criteria=criteria,
        recommendation=recommendation,
        warnings=warnings,
    )


# ---------- Per-criterion helpers (each one trivially testable) ----------


def _eval_confusion(data: CURB65Input) -> CURB65Criterion:
    return CURB65Criterion(
        code="C",
        description="新发意识障碍",
        met=data.confusion,
        value_observed=("是" if data.confusion else "否"),
    )


def _eval_urea(data: CURB65Input) -> CURB65Criterion:
    if data.urea_mmol_l is None:
        return CURB65Criterion(
            code="U",
            description="血尿素 > 7 mmol/L",
            met=False,
            value_observed="未测",
        )
    return CURB65Criterion(
        code="U",
        description="血尿素 > 7 mmol/L",
        met=data.urea_mmol_l > 7.0,
        value_observed=f"{data.urea_mmol_l} mmol/L",
    )


def _eval_respiratory_rate(data: CURB65Input) -> CURB65Criterion:
    return CURB65Criterion(
        code="R",
        description="呼吸频率 ≥ 30 次/分",
        met=data.respiratory_rate >= 30,
        value_observed=f"{data.respiratory_rate} 次/分",
    )


def _eval_blood_pressure(data: CURB65Input) -> CURB65Criterion:
    met = data.systolic_bp < 90 or data.diastolic_bp <= 60
    return CURB65Criterion(
        code="B",
        description="收缩压 < 90 mmHg 或 舒张压 ≤ 60 mmHg",
        met=met,
        value_observed=f"{data.systolic_bp}/{data.diastolic_bp} mmHg",
    )


def _eval_age(data: CURB65Input) -> CURB65Criterion:
    return CURB65Criterion(
        code="65",
        description="年龄 ≥ 65 岁",
        met=data.age >= 65,
        value_observed=f"{data.age} 岁",
    )


def _interpret(score: int) -> tuple[RiskLevel, str]:
    """Map score to risk level and recommendation."""
    if score <= 1:
        return (
            "low",
            "低危(评分 0-1):门诊治疗通常适合。结合临床判断决定。",
        )
    if score == 2:
        return (
            "intermediate",
            "中危(评分 2):建议考虑住院治疗或短期观察。",
        )
    # score 3-5
    return (
        "severe",
        "重症(评分 ≥3):建议住院治疗;评分 ≥4 时考虑ICU治疗。",
    )