"""
DK-SAR 用户反馈与迭代记录模块

对应 AI 产品经理核心技能：
  - 收集用户行为数据
  - 分析使用模式，提出迭代建议
  - 跟进产品上线后效果

数据存储在本地 feedback.json，不涉及隐私
"""

import json
import os
from datetime import datetime


FEEDBACK_FILE = "data/feedback.json"
os.makedirs("data", exist_ok=True)


def save_feedback(
    user_input: str,
    report: dict,
    rating: int,           # 1-5 星评分
    comment: str = "",     # 用户文字反馈
    scores: dict = None,   # 系统自动评估分数
):
    """
    保存一条用户反馈
    每次分析完成后调用，记录用户行为数据
    """
    record = {
        "timestamp":   datetime.now().isoformat(),
        "user_input":  user_input[:200],           # 截断避免过长
        "equipment":   report.get("node_info", {}).get("equipment", ""),
        "parameter":   report.get("node_info", {}).get("parameter", ""),
        "rating":      rating,
        "comment":     comment,
        "reflection_rounds": report.get("analysis_metadata", {}).get("reflection_rounds", 0),
        "referenced_cases":  report.get("analysis_metadata", {}).get("referenced_cases", []),
        "auto_scores": scores or {},
    }

    records = _load_all()
    records.append(record)

    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return record


def get_stats() -> dict:
    """
    统计用户行为数据，用于产品迭代决策
    返回可直接展示在界面上的统计摘要
    """
    records = _load_all()
    if not records:
        return {"total": 0, "message": "暂无数据"}

    total = len(records)
    ratings = [r["rating"] for r in records if r.get("rating")]
    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else 0

    # 最常分析的设备类型
    equipments = [r["equipment"] for r in records if r.get("equipment")]
    eq_counts = {}
    for eq in equipments:
        eq_counts[eq] = eq_counts.get(eq, 0) + 1
    top_equipment = sorted(eq_counts.items(), key=lambda x: -x[1])[:3]

    # 平均反思轮次
    rounds = [r["reflection_rounds"] for r in records if "reflection_rounds" in r]
    avg_rounds = round(sum(rounds) / len(rounds), 2) if rounds else 0

    # 近7条反馈
    recent = records[-7:][::-1]

    return {
        "total":         total,
        "avg_rating":    avg_rating,
        "avg_rounds":    avg_rounds,
        "top_equipment": top_equipment,
        "recent_count":  len(recent),
        "recent":        recent,
    }


def format_stats_markdown(stats: dict) -> str:
    """将统计数据格式化为 Markdown"""
    if stats.get("total", 0) == 0:
        return "暂无使用数据，完成第一次分析后这里会显示统计信息。"

    lines = [
        "### 📈 使用数据统计",
        f"- **总分析次数**: {stats['total']} 次",
        f"- **平均用户评分**: {'⭐' * round(stats['avg_rating'])} {stats['avg_rating']} / 5.0",
        f"- **平均反思轮次**: {stats['avg_rounds']} 轮",
    ]

    if stats.get("top_equipment"):
        eq_str = "、".join(f"{e[0]}({e[1]}次)" for e in stats["top_equipment"])
        lines.append(f"- **最常分析设备**: {eq_str}")

    if stats.get("recent"):
        lines.append("\n**最近分析记录：**")
        for r in stats["recent"][:3]:
            t = r["timestamp"][:16].replace("T", " ")
            lines.append(
                f"- `{t}` {r.get('equipment','')} / {r.get('parameter','')} "
                f"| 评分: {'⭐'*r.get('rating',0)} | 轮次: {r.get('reflection_rounds',0)}"
            )

    return "\n".join(lines)


def _load_all() -> list:
    if not os.path.exists(FEEDBACK_FILE):
        return []
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []
