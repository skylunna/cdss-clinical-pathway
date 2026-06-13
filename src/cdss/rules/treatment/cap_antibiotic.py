"""
Empirical antibiotic recommendation for community-acquired pneumonia (CAP).

Based on the 2016 Chinese CAP guidelines (中国成人社区获得性肺炎诊断和治疗指南),
illustratively. Heavily simplified.

LEARNING PROJECT ONLY. NOT FOR CLINICAL USE.

Inputs:
    severity                  - CAP severity (matches CURB-65 risk_level)
    has_penicillin_allergy    - patient is allergic to penicillin
    has_recent_antibiotic_use - antibiotic use within past 3 months
    suspect_pseudomonas       - risk factors for P. aeruginosa
                                (immunocompromise, structural lung disease, recent hosp.)

Output:
    treatment_setting - outpatient / ward / ICU
    regimens          - 1-2 recommended regimen options
    duration_days     - typical treatment duration
    warnings          - clinical caveats and adjustment reminders


社区获得性肺炎（CAP）的抗生素经验性推荐。
根据2016年中国成人社区获得性肺炎诊断和治疗指南，示意图。大幅简化。
仅限学习用途，不可用于临床。
输入：
    严重程度                  - CAP 严重程度（与 CURB-6555 风险等级匹配）
    有青霉素过敏史    - - - 患者对青霉素过敏
    近期使用抗生素     - - 近3个月内使用过抗生素
    怀疑铜绿假单胞菌感染   - P. aeruginosa 的风险因素
    （免疫功能低下、结构性肺部疾病、近期住院）
输出：
    治疗设置 - 门诊 / 住院 / 重症监护室
    方案          - - 推荐的1-2种治疗方案选项
    持续天数     - - - 常规治疗时长
    警告          - 临床注意事项及调整提醒
"""
from typing import Literal

from pydantic import BaseModel, Field

SeverityLevel = Literal["low", "intermediate", "severe"]


# --- Models ---
class CAPAntibioticInput(BaseModel):
    """CAP实证抗生素推荐的输入数据。"""

    severity: SeverityLevel = Field(
        description="CAP严重程度。将其与CURB-65风险等级匹配：“低（0-1）/ 中等（2）/ 严重（>=3） "
    )
    has_penicillin_allergy: bool = Field(
        default=False, description="患者对青霉素过敏" 
    )
    has_recent_antibiotic_use: bool = Field(
        default=False, description="过去3个月内使用抗生素"
    )
    suspect_pseudomonas: bool = Field(
        default=False,
        description="铜绿假单胞菌感染的风险因素：免疫功能低下、结构性肺部疾病（如支气管扩张）、近期住院史"
    )


class AntibioticRegimen(BaseModel):
    # 抗生素疗程
    name: str = Field(description="信件制度标签")
    drugs: list[str] = Field(description="特定药物及其剂量和给药途径")
    rationale: str = Field(description="为何建议采用此方案")


class CAPAntibioticResult(BaseModel):
    # CAP抗生素结果
    treatment_setting: Literal["门诊", "普通病房", "ICU"]
    regimens: list[AntibioticRegimen] = Field(min_length=1)
    duration_days: str
    warnings: list[str] = Field(default_factory=list)


# --- Tool metadata ---
RULE_NAME = "cap_empirical_antibiotic"  # 经验性抗生素

# “推荐社区获得性肺炎（CAP）的 empiricic 抗生素方案。”  
# “**在计算 CURB-655 后使用此方法**：将 curb65 的 risk_level 作为 '严重程度' 参数传入。”  
# “返回 1-22 种方案选项、治疗方案、典型疗程及临床注意事项。”  
# “输入包括严重程度（必填）和修饰因素：青霉素过敏、近期使用抗生素、疑似铜绿假单胞菌感染。”
RULE_DESCRIPTION = (
    "Recommend empirical antibiotic regimens for community-acquired pneumonia (CAP). "
    "**Call this AFTER computing CURB-65**: pass curb65's risk_level as the 'severity' argument. "
    "Returns 1-2 regimen options, treatment setting, typical duration, and clinical caveats. "
    "Inputs include severity (required) and modifiers: penicillin allergy, recent antibiotic "
    "use, suspicion of Pseudomonas."
)

# --- Core Logic ---
def compute(data: CAPAntibioticInput) -> CAPAntibioticResult:
    """计算经验性抗生素推荐"""
    setting = _setting_for(data.severity)
    duration = _duration_for(data.severity)

    warnings: list[str] = [
        "本建议为经验性方案, 具体用药须由医师结合患者情况(肝肾功能、合并用药、 "
        "过敏史细节、当地耐药情况) 调整。",
    ]
    if data.has_penicillin_allergy:
        warnings.append(
            "青霉素过敏: 避免青霉素类; 头孢类有交叉过敏可能, 请谨慎使用。"
        )
    if data.has_recent_antibiotic_use:
        warnings.append(
            "近期使用过抗生素: 耐药风险增加, 建议留取病原学样本, 根据培养调整。"
        )
    if data.suspect_pseudomonas:
        warnings.append(
            "怀疑铜绿假单胞菌:必须留取痰培养、血培养指导后续治疗。"
        )
    if data.severity == "severe":
        warnings.append(
            "重症患者: 24-48 小时内评估治疗反应, 无改善需考虑调整方案。"
        )
    
    regimens = _generate_regimens(data)

    return CAPAntibioticResult(
        treatment_setting=setting,
        regimens=regimens,
        duration_days=duration,
        warnings=warnings,
    )




def _setting_for(severity: SeverityLevel) -> str:
    return {
        "low": "门诊",
        "intermediate": "普通病房",
        "severe": "ICU"
    }[severity]

def _duration_for(severity: SeverityLevel) -> str:
    # 持续时间
    return {
        "low": "5-7 天",
        "intermediate": "7-10 天",
        "severe": "10-14 天(根据病情调整)"
    }[severity]

def _generate_regimens(data: CAPAntibioticInput) -> list[AntibioticRegimen]:
    if data.severity == "low":
        return _regimens_low(data)
    if data.severity == "intermediate":
        return _regimens_intermediate(data)
    return _regimens_severe(data)


def _regimens_low(data: CAPAntibioticInput) -> list[AntibioticRegimen]:
    if data.has_penicillin_allergy:
        return [
            AntibioticRegimen(
                name="呼吸氟喹诺酮单药",
                drugs=["左氧氟沙星 0.5g po qd", "或 莫西沙星 0.4g po qd"],
                rationale="青霉素过敏患者的门诊首选;覆盖典型+非典型病原体",
            ),
        ]
    
    if data.has_recent_antibiotic_use:
        return [
            AntibioticRegimen(
                name="联合方案",
                drugs=["阿莫西林/克拉维酸 0.625g po q8h", "+ 阿奇霉素 0.5g po qd"],
                rationale="近期用药者:加β-内酰胺酶抑制剂,联合大环内酯覆盖非典型",
            ),
        ]
    
    return [
        AntibioticRegimen(
            name="单药方案 A",
            drugs=["阿莫西林 0.5g po q8h"],
            rationale="无合并症的轻症CAP首选;主要覆盖肺炎链球菌",
        ),
        AntibioticRegimen(
            name="单药方案 B",
            drugs=["阿奇霉素 0.5g po qd × 3天"],
            rationale="替代方案,尤其考虑非典型病原体(支原体、衣原体)",
        ),
    ]


def _regimens_intermediate(data: CAPAntibioticInput) -> list[AntibioticRegimen]:
    if data.has_penicillin_allergy:
        return [
            AntibioticRegimen(
                name="氟喹诺酮单药",
                drugs=["左氧氟沙星 0.5g iv/po qd"],
                rationale="青霉素过敏的住院方案",
            ),
        ]
    return [
        AntibioticRegimen(
            name="联合方案(首选)",
            drugs=["头孢曲松 1-2g iv qd", "+ 阿奇霉素 0.5g iv/po qd"],
            rationale="覆盖肺炎链球菌(含耐药)、流感嗜血杆菌、非典型病原体",
        ),
        AntibioticRegimen(
            name="单药方案",
            drugs=["左氧氟沙星 0.5g iv/po qd"],
            rationale="呼吸氟喹诺酮单药替代",
        ),
    ]


def _regimens_severe(data: CAPAntibioticInput) -> list[AntibioticRegimen]:
    if data.suspect_pseudomonas:
        return [
            AntibioticRegimen(
                name="覆盖铜绿假单胞菌方案",
                drugs=[
                    "哌拉西林/他唑巴坦 4.5g iv q8h",
                    "+ 左氧氟沙星 0.75g iv qd",
                ],
                rationale="重症+假单胞菌风险,需抗假单胞菌β-内酰胺联合喹诺酮",
            ),
        ]
    if data.has_penicillin_allergy:
        return [
            AntibioticRegimen(
                name="过敏患者重症方案",
                drugs=["左氧氟沙星 0.75g iv qd"],
                rationale="青霉素过敏的重症方案;如疑 MRSA 加万古霉素",
            ),
        ]
    return [
        AntibioticRegimen(
            name="联合方案(首选)",
            drugs=["头孢曲松 2g iv qd", "+ 阿奇霉素 0.5g iv qd"],
            rationale="重症 CAP 经验性方案;覆盖典型+非典型",
        ),
        AntibioticRegimen(
            name="替代联合方案",
            drugs=["头孢曲松 2g iv qd", "+ 左氧氟沙星 0.75g iv qd"],
            rationale="氟喹诺酮替代大环内酯",
        ),
    ]