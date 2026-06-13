"""
Unit tests for CAP empirical antibiotic recommendation.

CAP实证抗生素推荐的单元测试。
"""
import pytest

from cdss.rules.treatment.cap_antibiotic import(
    CAPAntibioticInput,
    CAPAntibioticResult,
    SeverityLevel,
    compute,
)


# --- Setting and duration mapping ---
@pytest.mark.parametrize(
    # 严重程度，预期设置，预期持续时间关键词
    "severity,expected_setting,expected_duration_keyword",
    [
        ("low", "门诊", "5-7"),
        ("intermediate", "普通病房", "7-10"),
        ("severe", "ICU", "10-14"),
    ],
)

def test_setting_and_duration_for_severity(
    severity: SeverityLevel,
    expected_setting: str,
    expected_duration_keyword: str,
) -> None:
    result = compute(CAPAntibioticInput(severity=severity))
    assert result.treatment_setting == expected_setting
    assert expected_duration_keyword in result.duration_days

# --- Always-present default warning ---
def test_always_includes_general_disclaimer() -> None:
    result = compute(CAPAntibioticInput(severity="low"))
    assert any("经验性方案" in w for w in result.warnings)


# --- Penicillin allergy branch ---
# 青霉素过敏分支
@pytest.mark.parametrize("severity", ["low", "intermediate", "severe"])
def test_penicillin_allergy_avoids_beta_lactams(severity: SeverityLevel) -> None:
    # 避免使用β-内酰胺类抗生素的青霉素过敏测试
    result = compute(
        CAPAntibioticInput(severity=severity, has_penicillin_allergy=True)
    )
    # No regimen should mention 阿莫西林 / 头孢
    for r in result.regimens:
        joined = " ".join(r.drugs)
        assert "阿莫西林" not in joined
        assert "头孢" not in joined
    # Should warn about cross-allergy 应警惕交叉过敏
    assert any("青霉素过敏" in w for w in result.warnings)


# ---------- Pseudomonas branch ----------
def test_severe_with_pseudomonas_uses_antipseudomonal() -> None:
    result = compute(
        CAPAntibioticInput(severity="severe", suspect_pseudomonas=True)
    )
    # Should include an antipseudomonal beta-lactam
    drugs_text = " ".join(d for r in result.regimens for d in r.drugs)
    assert "哌拉西林" in drugs_text
    assert any("铜绿假单胞菌" in w for w in result.warnings)


def test_low_severity_ignores_pseudomonas_flag() -> None:
    """At low severity, even with pseudomonas flag, regimen stays simple
    (you don't go to ICU-grade drugs for an outpatient)."""
    result = compute(
        CAPAntibioticInput(severity="low", suspect_pseudomonas=True)
    )
    # Outpatient regimens shouldn't include antipseudomonal beta-lactams
    drugs_text = " ".join(d for r in result.regimens for d in r.drugs)
    assert "哌拉西林" not in drugs_text


# ---------- Recent antibiotic use branch ----------
# 近期抗生素使用分支 
def test_recent_antibiotic_use_adds_warning() -> None:
    result = compute(
        CAPAntibioticInput(severity="low", has_recent_antibiotic_use=True)
    )
    assert any("近期" in w and "抗生素" in w for w in result.warnings)


def test_recent_antibiotic_use_changes_low_severity_regimen() -> None:
    no_recent = compute(CAPAntibioticInput(severity="low"))
    with_recent = compute(
        CAPAntibioticInput(severity="low", has_recent_antibiotic_use=True)
    )
    # The regimens should differ when there's recent antibiotic exposure
    no_recent_drugs = {d for r in no_recent.regimens for d in r.drugs}
    with_recent_drugs = {d for r in with_recent.regimens for d in r.drugs}
    assert no_recent_drugs != with_recent_drugs

# ---------- Severe case extra warning ----------
# 严重病例额外警告
def test_severe_includes_reassessment_warning() -> None:
    result = compute(CAPAntibioticInput(severity="severe"))
    assert any("24-48" in w for w in result.warnings)

# ---------- Output contract ----------
def test_returns_correct_type_and_has_at_least_one_regimen() -> None:
    result = compute(CAPAntibioticInput(severity="intermediate"))
    assert isinstance(result, CAPAntibioticResult)
    assert len(result.regimens) >= 1