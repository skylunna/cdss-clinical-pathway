"""
Unit tests for CURB-65 scoring rule.

Strategy:
    - Test each criterion's edge cases (boundary values)
    - Test the overall score calculation and risk-level mapping
    - Test data-quality handling (missing urea -> warning)
    - Use parametrize for thresholds to keep tests DRY

CURB-65评分规则的单元测试。
策略：
    - 测试每个标准的边界情况（临界值）
    - 测试整体得分计算和风险等级映射
    - 测试数据质量处理（尿素缺失 → 发出警告）
    - 使用参数化方式设置阈值，以保持测试代码的DRY原则
"""
import pytest


from cdss.rules.scoring.curb65 import (
    CURB65Input,
    CURB65Result,
    compute,
)


# --- Fixtures ---
@pytest.fixture
def healthy_young_adult() -> CURB65Input:
    """基准输入，应得分为0（未满足任何标准）。"""
    return CURB65Input(
        age=30,
        confusion=False,
        urea_mmol_l=5.0,
        respiratory_rate=16,
        systolic_bp=120,
        diastolic_bp=80,
    )


# --- Score boundary tests ---
# 得分边界测试
def test_score_zero(healthy_young_adult: CURB65Input) -> None:
    """未满足标准 -> 得分0，低风险。"""
    result = compute(healthy_young_adult)
    assert result.score == 0
    assert result.risk_level == "low"
    assert all(not c.met for c in result.criteria)
    assert result.warnings == []


def test_score_max_all_criteria_met() -> None:
    """All 5 criteria met -> score 5, severe."""
    # 所有5项标准均满足 → 评分5，严重。
    data = CURB65Input(
        age=80,
        confusion=True,
        urea_mmol_l=12.0,
        respiratory_rate=35,
        systolic_bp=80,
        diastolic_bp=50,
    )
    result = compute(data)
    assert result.score == 5
    assert result.risk_level == "severe"
    assert all(c.met for c in result.criteria)


# --- Boundary value tests (parametrized) ---
# 边界值测试（参数化）
@pytest.mark.parametrize(
    "age,expected_met",
    [
        (64, False),  # just below threshold
        (65, True),   # exactly threshold (>= 65)
        (66, True),   # above
    ],
)
def test_age_boundary(
    healthy_young_adult: CURB65Input,
    age: int,
    expected_met: bool,
) -> None:
    data = healthy_young_adult.model_copy(update={"age": age})
    result = compute(data)
    age_criterion = next(c for c in result.criteria if c.code == "65")
    assert age_criterion.met is expected_met

@pytest.mark.parametrize(
    "urea,expected_met",
    [
        (7.0, False),  # exactly at threshold (rule is strict: > 7) 恰好在阈值处（规则严格：> 7）
        (7.01, True),  # just above 
        (15.0, True),  # well above 远高于
    ],
)
def test_urea_boundary(
    healthy_young_adult: CURB65Input,
    urea: float,
    expected_met: bool,
) -> None:
    data = healthy_young_adult.model_copy(update={"urea_mmol_l": urea})
    result = compute(data)
    urea_criterion = next(c for c in result.criteria if c.code == "U")
    assert urea_criterion.met is expected_met


@pytest.mark.parametrize(
    "rr,expected_met",
    [(29, False), (30, True), (40, True)],
)
def test_respiratory_rate_boundary(
    healthy_young_adult: CURB65Input,
    rr: int,
    expected_met: bool,
) -> None:
    data = healthy_young_adult.model_copy(update={"respiratory_rate": rr})
    result = compute(data)
    rr_criterion = next(c for c in result.criteria if c.code == "R")
    assert rr_criterion.met is expected_met


@pytest.mark.parametrize(
    "sbp,dbp,expected_met",
    [
        (120, 80, False),  # both normal
        (89, 80, True),    # systolic < 90
        (120, 60, True),   # diastolic <= 60
        (120, 61, False),  # just above diastolic threshold
        (90, 70, False),   # systolic exactly 90 (rule is strict: < 90)
        (85, 55, True),    # both criteria met (still counts once)
    ],
)
def test_blood_pressure_logic(
    healthy_young_adult: CURB65Input,
    sbp: int,
    dbp: int,
    expected_met: bool,
) -> None:
    """B criterion: systolic<90 OR diastolic<=60 (either one is enough)."""
    # B标准：收缩压<90 或 舒张压≤60（任一条件即可）
    data = healthy_young_adult.model_copy(
        update={"systolic_bp": sbp, "diastolic_bp": dbp}
    )
    result = compute(data)
    bp_criterion = next(c for c in result.criteria if c.code == "B")
    assert bp_criterion.met is expected_met


# ---------- Risk level mapping ----------


@pytest.mark.parametrize(
    "score,expected_risk",
    [
        (0, "low"),
        (1, "low"),
        (2, "intermediate"),
        (3, "severe"),
        (4, "severe"),
        (5, "severe"),
    ],
)
def test_risk_level_mapping(score: int, expected_risk: str) -> None:
    """Verify risk level mapping by constructing inputs of known scores."""
    # We achieve the desired score by toggling criteria deterministically
    # 通过构建已知分数的输入来验证风险等级映射。  
    # 我们通过确定性地切换标准来实现所需的分数。
    data = CURB65Input(
        age=65 if score >= 1 else 30,
        confusion=score >= 2,
        urea_mmol_l=12.0 if score >= 3 else 5.0,
        respiratory_rate=35 if score >= 4 else 16,
        systolic_bp=80 if score >= 5 else 120,
        diastolic_bp=80,
    )
    result = compute(data)
    assert result.score == score
    assert result.risk_level == expected_risk


# ---------- Data-quality handling ----------


def test_missing_urea_produces_warning(healthy_young_adult: CURB65Input) -> None:
    """When urea is None, U is treated as 0 but a warning is emitted."""
    # 当尿素为 None 时，U 被视为 0，但会发出警告。
    data = healthy_young_adult.model_copy(update={"urea_mmol_l": None})
    result = compute(data)
    urea_criterion = next(c for c in result.criteria if c.code == "U")
    assert urea_criterion.met is False
    assert urea_criterion.value_observed == "未测"
    assert len(result.warnings) == 1
    assert "Urea" in result.warnings[0]


def test_no_warning_when_all_data_present(
    healthy_young_adult: CURB65Input,
) -> None:
    result = compute(healthy_young_adult)
    assert result.warnings == []


# ---------- Input validation (Pydantic-level) ----------


def test_invalid_age_rejected() -> None:
    """Pydantic should reject obviously invalid inputs at construction time."""
    # Pydantic 应在构造时拒绝明显无效的输入
    with pytest.raises(ValueError):
        CURB65Input(
            age=-5,  # invalid
            confusion=False,
            urea_mmol_l=5.0,
            respiratory_rate=16,
            systolic_bp=120,
            diastolic_bp=80,
        )


def test_returns_curb65_result_type(healthy_young_adult: CURB65Input) -> None:
    """compute() must return a CURB65Result (smoke test on contract)."""
    # compute() 必须返回一个 CURB65Result（合同烟雾测试）
    result = compute(healthy_young_adult)
    assert isinstance(result, CURB65Result)