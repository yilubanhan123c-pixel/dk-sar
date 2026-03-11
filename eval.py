"""
DK-SAR 评估模块
实现论文中的四个核心评估指标：
  - PCC  (Physical Concept Coverage)    物理概念覆盖率
  - CCC  (Causal Chain Completeness)    因果链完整性
  - RDI  (Recommendation Detail Index)  建议措施详细度
  - LCC  (Literature Case Correlation)  案例关联度

用法：
    from eval import evaluate_report
    scores = evaluate_report(report, reference=None)
    print(scores)
"""

import json
import re
from utils.logger import eval_log
from utils.llm import call_llm_json


# ── 物理概念关键词库（用于 PCC 计算）─────────────────────────────
PHYSICS_CONCEPTS = {
    "热力学": ["温度", "热量", "焓", "熵", "热平衡", "绝热", "等温", "热失控",
               "Arrhenius", "活化能", "放热", "吸热", "冷却", "散热"],
    "传热传质": ["传热系数", "Nu", "Re", "对流", "导热", "辐射", "换热",
                "传质", "扩散", "蒸发", "冷凝", "闪蒸"],
    "流体力学": ["压力", "流量", "流速", "伯努利", "压降", "节流",
                "汽蚀", "水锤", "湍流", "层流"],
    "守恒定律": ["质量守恒", "能量守恒", "动量守恒", "物料平衡", "热平衡"],
    "化学反应": ["反应速率", "转化率", "选择性", "催化", "失控反应",
                "爆炸极限", "LEL", "UEL", "闪点", "自燃温度"],
    "设备完整性": ["腐蚀", "疲劳", "蠕变", "氢损伤", "应力", "裂纹", "泄漏"],
}

# ── 因果链完整性检查项 ─────────────────────────────────────────
CAUSAL_CHAIN_ELEMENTS = ["初始原因", "触发事件", "扩展后果", "最终后果"]


# ════════════════════════════════════════════════════════════════
#  指标1：PCC — 物理概念覆盖率
#  计算报告中涉及的物理概念类别数 / 总类别数
# ════════════════════════════════════════════════════════════════

def compute_pcc(report: dict) -> float:
    """
    Physical Concept Coverage (PCC)
    衡量报告对物理机理的覆盖广度
    
    返回: 0.0 ~ 1.0
    """
    report_text = _report_to_plain_text(report).lower()
    
    covered_categories = 0
    total_categories = len(PHYSICS_CONCEPTS)
    details = {}
    
    for category, keywords in PHYSICS_CONCEPTS.items():
        hits = [kw for kw in keywords if kw.lower() in report_text]
        if hits:
            covered_categories += 1
            details[category] = hits[:3]  # 最多记录3个命中词
    
    pcc = covered_categories / total_categories if total_categories > 0 else 0.0
    
    eval_log.debug(f"PCC: {covered_categories}/{total_categories} 类别覆盖 → {pcc:.3f}")
    eval_log.debug(f"命中详情: {details}")
    
    return round(pcc, 4)


# ════════════════════════════════════════════════════════════════
#  指标2：CCC — 因果链完整性
#  检查后果链是否包含完整的事故演化路径
# ════════════════════════════════════════════════════════════════

def compute_ccc(report: dict) -> float:
    """
    Causal Chain Completeness (CCC)
    衡量因果链描述的完整程度
    
    返回: 0.0 ~ 1.0
    """
    scores = []
    
    for deviation in report.get("deviations", []):
        causes       = deviation.get("causes", [])
        consequences = deviation.get("consequences", [])
        
        # 检查原因数量（至少3个）
        cause_score = min(len(causes) / 3, 1.0)
        
        # 检查后果链长度（至少3步）
        consequence_score = min(len(consequences) / 3, 1.0)
        
        # 检查原因类型多样性（覆盖多种类型更好）
        cause_types = set(c.get("type", "") for c in causes)
        diversity_score = min(len(cause_types) / 3, 1.0)
        
        deviation_score = (cause_score + consequence_score + diversity_score) / 3
        scores.append(deviation_score)
    
    ccc = sum(scores) / len(scores) if scores else 0.0
    
    eval_log.debug(f"CCC: {ccc:.3f} （{len(scores)} 个偏差分析）")
    
    return round(ccc, 4)


# ════════════════════════════════════════════════════════════════
#  指标3：RDI — 建议措施详细度
#  检查建议措施的可操作性和优先级分布
# ════════════════════════════════════════════════════════════════

def compute_rdi(report: dict) -> float:
    """
    Recommendation Detail Index (RDI)
    衡量建议措施的具体性和可执行性
    
    返回: 0.0 ~ 1.0
    """
    all_recommendations = []
    
    for deviation in report.get("deviations", []):
        all_recommendations.extend(deviation.get("recommendations", []))
    
    if not all_recommendations:
        return 0.0
    
    scores = []
    for rec in all_recommendations:
        action = rec.get("action", "")
        priority = rec.get("priority", "")
        
        # 动作描述长度得分（越具体越长）
        length_score = min(len(action) / 30, 1.0)
        
        # 是否有优先级
        priority_score = 1.0 if priority in ["高", "中", "低"] else 0.0
        
        # 是否包含具体动词（建立、安装、定期、检查等）
        action_verbs = ["建立", "安装", "定期", "检查", "培训", "制定",
                       "增加", "设置", "更换", "完善", "加强", "实施"]
        verb_score = 1.0 if any(v in action for v in action_verbs) else 0.5
        
        rec_score = (length_score + priority_score + verb_score) / 3
        scores.append(rec_score)
    
    # 优先级分布加分（高中低都有更好）
    priorities = [r.get("priority", "") for r in all_recommendations]
    has_high = "高" in priorities
    has_medium = "中" in priorities
    diversity_bonus = 0.1 if (has_high and has_medium) else 0.0
    
    rdi = min(sum(scores) / len(scores) + diversity_bonus, 1.0)
    
    eval_log.debug(f"RDI: {rdi:.3f} （{len(all_recommendations)} 条建议）")
    
    return round(rdi, 4)


# ════════════════════════════════════════════════════════════════
#  指标4：LCC — 案例关联度
#  衡量检索到的历史案例与当前场景的相关程度
# ════════════════════════════════════════════════════════════════

def compute_lcc(report: dict) -> float:
    """
    Literature Case Correlation (LCC)
    衡量参考案例与分析内容的关联程度
    
    返回: 0.0 ~ 1.0
    """
    meta = report.get("analysis_metadata", {})
    referenced_cases = meta.get("referenced_cases", [])
    
    if not referenced_cases:
        eval_log.debug("LCC: 无参考案例，得分 0.0")
        return 0.0
    
    # 检查案例是否被实际引用在分析中
    report_text = _report_to_plain_text(report)
    
    # 基础分：有案例就给基础分
    base_score = min(len(referenced_cases) / 3, 1.0) * 0.6
    
    # 引用质量分：案例ID是否出现在元数据中
    quality_score = 0.4 if len(referenced_cases) >= 2 else 0.2
    
    lcc = base_score + quality_score
    
    eval_log.debug(f"LCC: {lcc:.3f} （引用案例: {referenced_cases}）")
    
    return round(min(lcc, 1.0), 4)


# ════════════════════════════════════════════════════════════════
#  综合评估入口
# ════════════════════════════════════════════════════════════════

def evaluate_report(report: dict) -> dict:
    """
    对 HAZOP 分析报告进行综合评估
    
    返回:
    {
        "PCC": 0.83,   # 物理概念覆盖率
        "CCC": 0.91,   # 因果链完整性
        "RDI": 0.78,   # 建议详细度
        "LCC": 0.85,   # 案例关联度
        "overall": 0.84,  # 综合得分（加权平均）
        "grade": "良好",
        "details": {...}
    }
    """
    eval_log.info("开始评估 HAZOP 分析报告...")
    
    pcc = compute_pcc(report)
    ccc = compute_ccc(report)
    rdi = compute_rdi(report)
    lcc = compute_lcc(report)
    
    # 加权综合得分（PCC和CCC权重更高，对应论文重点）
    weights = {"PCC": 0.35, "CCC": 0.30, "RDI": 0.20, "LCC": 0.15}
    overall = (
        pcc * weights["PCC"] +
        ccc * weights["CCC"] +
        rdi * weights["RDI"] +
        lcc * weights["LCC"]
    )
    overall = round(overall, 4)
    
    # 评级
    if overall >= 0.85:
        grade = "优秀 ★★★★★"
    elif overall >= 0.70:
        grade = "良好 ★★★★"
    elif overall >= 0.55:
        grade = "一般 ★★★"
    else:
        grade = "较差 ★★"
    
    result = {
        "PCC": pcc,
        "CCC": ccc,
        "RDI": rdi,
        "LCC": lcc,
        "overall": overall,
        "grade": grade,
        "weights": weights,
    }
    
    eval_log.success(
        f"评估完成 | PCC={pcc:.2f} CCC={ccc:.2f} "
        f"RDI={rdi:.2f} LCC={lcc:.2f} | 综合={overall:.2f} {grade}"
    )
    
    return result


def format_scores_markdown(scores: dict) -> str:
    """将评估结果格式化为 Markdown 展示"""
    if not scores:
        return "暂无评估数据"
    
    lines = [
        "### 📊 报告质量评估",
        "",
        f"**综合得分: {scores['overall']*100:.1f}分 / 100分　{scores['grade']}**",
        "",
        "| 指标 | 全称 | 得分 | 说明 |",
        "|------|------|------|------|",
        f"| PCC | 物理概念覆盖率 | {scores['PCC']*100:.1f}% | 涵盖的物理机理类别 |",
        f"| CCC | 因果链完整性 | {scores['CCC']*100:.1f}% | 原因→后果链的完整程度 |",
        f"| RDI | 建议详细度 | {scores['RDI']*100:.1f}% | 措施的可操作性 |",
        f"| LCC | 案例关联度 | {scores['LCC']*100:.1f}% | 历史案例的相关性 |",
    ]
    return "\n".join(lines)


# ── 工具函数 ───────────────────────────────────────────────────

def _report_to_plain_text(report: dict) -> str:
    """将报告 JSON 展平为纯文本，用于关键词匹配"""
    return json.dumps(report, ensure_ascii=False)


# ── 命令行测试 ─────────────────────────────────────────────────

if __name__ == "__main__":
    # 测试用假数据
    sample_report = {
        "node_info": {
            "equipment": "反应釜", "parameter": "温度",
            "deviation_type": "温度过高", "deviation_direction": "过高",
            "normal_value": "80°C", "current_value": "120°C"
        },
        "deviations": [{
            "causes": [
                {"description": "冷却水流量不足导致散热能力下降，热平衡被打破", "type": "设备故障"},
                {"description": "Arrhenius动力学：温度升高使反应速率指数增长，放热加速", "type": "工艺异常"},
                {"description": "操作人员未及时发现温度报警", "type": "操作失误"},
            ],
            "consequences": [
                "冷却能力不足，温度持续升高",
                "反应速率加快，放热量增加，形成正反馈",
                "温度超过设计值，可能引发失控反应"
            ],
            "safeguards": [
                {"measure": "温度报警系统", "effectiveness": "部分有效"},
                {"measure": "紧急冷却联锁", "effectiveness": "有效"},
            ],
            "recommendations": [
                {"action": "建立冷却水流量与温度的联锁保护", "priority": "高"},
                {"action": "定期检查冷却系统换热效率", "priority": "中"},
            ]
        }],
        "analysis_metadata": {
            "referenced_cases": ["R001", "R002"],
            "reflection_rounds": 2,
            "physical_issues_found": [],
            "confidence_level": "高"
        }
    }
    
    scores = evaluate_report(sample_report)
    print("\n评估结果:")
    print(format_scores_markdown(scores))
