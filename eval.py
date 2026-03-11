"""
DK-SAR 评估模块 v2
修复：兼容新版分层报告格式（causes为dict，consequences为dict列表，recommendations为dict）
指标：PCC / CCC / RDI / LCC
"""
import json
from utils.logger import eval_log


PHYSICS_CONCEPTS = {
    "热力学":   ["温度", "热量", "焓", "熵", "热平衡", "绝热", "热失控", "放热", "吸热", "冷却"],
    "传热传质": ["传热", "Nu", "Re", "对流", "导热", "换热", "传质", "扩散", "蒸发", "冷凝"],
    "流体力学": ["压力", "流量", "流速", "伯努利", "压降", "汽蚀", "湍流", "层流"],
    "守恒定律": ["质量守恒", "能量守恒", "动量守恒", "物料平衡", "热平衡"],
    "化学反应": ["反应速率", "转化率", "催化", "失控反应", "爆炸极限", "LEL", "闪点"],
    "设备完整性": ["腐蚀", "疲劳", "泄漏", "裂纹", "应力", "密封"],
}


# ── 工具函数：兼容新旧两种格式 ────────────────────────────────────

def _extract_causes(deviation: dict) -> list:
    """提取所有原因条目，兼容 list 和 dict 分层两种格式"""
    causes_raw = deviation.get("causes", [])
    if isinstance(causes_raw, list):
        return causes_raw
    # 新版分层格式：{"primary": [...], "secondary": [...], "pending": [...]}
    all_causes = []
    for layer in ["primary", "secondary", "pending"]:
        all_causes.extend(causes_raw.get(layer, []))
    return all_causes


def _extract_consequences(deviation: dict) -> list:
    """提取后果链文本列表，兼容 str 和 {"stage":..., "description":...} 两种格式"""
    raw = deviation.get("consequences", [])
    texts = []
    for c in raw:
        if isinstance(c, dict):
            texts.append(c.get("description", ""))
        else:
            texts.append(str(c))
    return [t for t in texts if t]


def _extract_recommendations(deviation: dict) -> list:
    """提取建议措施列表，兼容 list 和 dict 三层格式"""
    recs_raw = deviation.get("recommendations", [])
    if isinstance(recs_raw, list):
        return recs_raw
    # 新版三层格式：{"immediate": [...], "short_term": [...], "long_term": [...]}
    all_recs = []
    for layer in ["immediate", "short_term", "long_term"]:
        all_recs.extend(recs_raw.get(layer, []))
    return all_recs


def _report_to_plain_text(report: dict) -> str:
    return json.dumps(report, ensure_ascii=False)


# ════════════════════════════════════════════════════════════════
#  PCC — 物理概念覆盖率
# ════════════════════════════════════════════════════════════════

def compute_pcc(report: dict) -> float:
    report_text = _report_to_plain_text(report).lower()
    covered = 0
    details = {}
    for category, keywords in PHYSICS_CONCEPTS.items():
        hits = [kw for kw in keywords if kw.lower() in report_text]
        if hits:
            covered += 1
            details[category] = hits[:3]
    pcc = covered / len(PHYSICS_CONCEPTS)
    eval_log.debug(f"PCC: {covered}/{len(PHYSICS_CONCEPTS)} 类别覆盖 → {pcc:.3f}")
    eval_log.debug(f"命中详情: {details}")
    return round(pcc, 4)


# ════════════════════════════════════════════════════════════════
#  CCC — 因果链完整性
# ════════════════════════════════════════════════════════════════

def compute_ccc(report: dict) -> float:
    scores = []
    for deviation in report.get("deviations", []):
        causes = _extract_causes(deviation)
        consequences = _extract_consequences(deviation)

        cause_score = min(len(causes) / 3, 1.0)
        consequence_score = min(len(consequences) / 3, 1.0)

        # 原因类型多样性
        cause_types = set(c.get("type", "") for c in causes if isinstance(c, dict))
        diversity_score = min(len(cause_types) / 3, 1.0)

        scores.append((cause_score + consequence_score + diversity_score) / 3)

    ccc = sum(scores) / len(scores) if scores else 0.0
    eval_log.debug(f"CCC: {ccc:.3f} （{len(scores)} 个偏差）")
    return round(ccc, 4)


# ════════════════════════════════════════════════════════════════
#  RDI — 建议措施详细度
# ════════════════════════════════════════════════════════════════

def compute_rdi(report: dict) -> float:
    all_recs = []
    for deviation in report.get("deviations", []):
        all_recs.extend(_extract_recommendations(deviation))

    if not all_recs:
        return 0.0

    action_verbs = ["建立", "安装", "定期", "检查", "培训", "制定",
                    "增加", "设置", "更换", "完善", "加强", "实施", "停止", "启动"]
    scores = []
    for rec in all_recs:
        if not isinstance(rec, dict):
            continue
        action = rec.get("action", "")
        priority = rec.get("priority", "")
        length_score = min(len(action) / 30, 1.0)
        priority_score = 1.0 if priority in ["高", "中", "低"] else 0.0
        verb_score = 1.0 if any(v in action for v in action_verbs) else 0.5
        scores.append((length_score + priority_score + verb_score) / 3)

    # 三层分级加分（有立即/短期/长期更好）
    recs_raw = report.get("deviations", [{}])[0].get("recommendations", {})
    diversity_bonus = 0.1 if isinstance(recs_raw, dict) and len(recs_raw) >= 2 else 0.0

    rdi = min((sum(scores) / len(scores) if scores else 0.0) + diversity_bonus, 1.0)
    eval_log.debug(f"RDI: {rdi:.3f} （{len(all_recs)} 条建议）")
    return round(rdi, 4)


# ════════════════════════════════════════════════════════════════
#  LCC — 案例关联度
# ════════════════════════════════════════════════════════════════

def compute_lcc(report: dict) -> float:
    meta = report.get("analysis_metadata", {})
    referenced_cases = meta.get("referenced_cases", [])
    if not referenced_cases:
        return 0.0
    base_score = min(len(referenced_cases) / 3, 1.0) * 0.6
    quality_score = 0.4 if len(referenced_cases) >= 2 else 0.2
    lcc = min(base_score + quality_score, 1.0)
    eval_log.debug(f"LCC: {lcc:.3f} （引用案例: {referenced_cases}）")
    return round(lcc, 4)


# ════════════════════════════════════════════════════════════════
#  综合评估入口
# ════════════════════════════════════════════════════════════════

def evaluate_report(report: dict) -> dict:
    eval_log.info("开始评估 HAZOP 分析报告...")
    pcc = compute_pcc(report)
    ccc = compute_ccc(report)
    rdi = compute_rdi(report)
    lcc = compute_lcc(report)

    weights = {"PCC": 0.35, "CCC": 0.30, "RDI": 0.20, "LCC": 0.15}
    overall = round(
        pcc * weights["PCC"] + ccc * weights["CCC"] +
        rdi * weights["RDI"] + lcc * weights["LCC"], 4
    )

    if overall >= 0.85:
        grade = "优秀 ★★★★★"
    elif overall >= 0.70:
        grade = "良好 ★★★★"
    elif overall >= 0.55:
        grade = "一般 ★★★"
    else:
        grade = "较差 ★★"

    result = {"PCC": pcc, "CCC": ccc, "RDI": rdi, "LCC": lcc,
              "overall": overall, "grade": grade, "weights": weights}

    eval_log.success(
        f"评估完成 | PCC={pcc:.2f} CCC={ccc:.2f} "
        f"RDI={rdi:.2f} LCC={lcc:.2f} | 综合={overall:.2f} {grade}"
    )
    return result


def format_scores_markdown(scores: dict) -> str:
    if not scores:
        return "暂无评估数据"
    lines = [
        "### 📊 报告质量评估",
        "",
        f"**综合得分：{scores['overall']*100:.1f} / 100　{scores['grade']}**",
        "",
        "| 指标 | 全称 | 得分 | 权重 | 说明 |",
        "|------|------|------|------|------|",
        f"| PCC | 物理概念覆盖率 | {scores['PCC']*100:.1f}% | 35% | 涵盖的物理机理类别数 |",
        f"| CCC | 因果链完整性  | {scores['CCC']*100:.1f}% | 30% | 原因→后果链的完整程度 |",
        f"| RDI | 建议详细度    | {scores['RDI']*100:.1f}% | 20% | 措施的可操作性 |",
        f"| LCC | 案例关联度    | {scores['LCC']*100:.1f}% | 15% | 历史案例的相关性 |",
    ]
    return "\n".join(lines)
