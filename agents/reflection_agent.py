"""
智能体 3：物理反思智能体
双层校验：Layer 1 负样本库匹配 + Layer 2 LLM 物理推理
"""

from fuzzywuzzy import fuzz

from utils.llm import call_llm_json
from utils.vector_store import get_vector_store
import config

DEEP_VERIFY_PROMPT = """你是一名严格的物理审查专家。请检查以下 HAZOP 分析中是否存在物理错误。
## 待检查的分析内容
{analysis_text}

## 检查项目（逐一核查）
1. 质量守恒：物料平衡是否正确？有无凭空产生或消失的物质？
2. 能量守恒：热量产生和散失是否平衡？因果关系是否合理？
3. 时序因果：原因是否发生在结果之前？有无因果颠倒？
4. 数值合理性：温度、压力、浓度等数值是否在合理范围内？
5. 过程机理：化学反应方向、传热传质规律是否正确？

## 输出格式（JSON）
{{
  "has_issues": true 或 false,
  "checks": {{
    "mass_conservation": true 或 false,
    "energy_conservation": true 或 false,
    "causal_order": true 或 false,
    "numeric_reasonableness": true 或 false,
    "process_mechanism": true 或 false
  }},
  "issues": [
    {{
      "issue_type": "质量守恒/能量守恒/时序因果/数值合理性/过程机理",
      "description": "发现的具体问题描述",
      "location": "问题出现在哪个字段",
      "correction_hint": "如何修正的建议"
    }}
  ],
  "summary": "总体评价（一句话）"
}}

规则：
1. 只报告确实违反物理定律、守恒关系或明确因果错误的问题。
2. “还可补充”“建议更严谨”“可以进一步说明”这类优化建议不算问题，不要写入 issues。
3. 如果没有明确物理违规，issues 必须为空，has_issues 必须为 false，checks 必须全部为 true。"""

HARD_VIOLATION_KEYWORDS = (
    "违反",
    "违背",
    "不守恒",
    "守恒错误",
    "质量守恒",
    "能量守恒",
    "因果颠倒",
    "时序错误",
    "数值不合理",
    "过程机理错误",
    "物理错误",
    "物料平衡错误",
    "热平衡错误",
    "热量守恒",
    "凭空产生",
    "凭空消失",
)

SOFT_SUGGESTION_KEYWORDS = (
    "建议",
    "可进一步",
    "可以进一步",
    "可补充",
    "建议补充",
    "建议说明",
    "表述可",
    "更严谨",
    "更完整",
    "更充分",
    "最好",
    "可优化",
    "可改进",
    "进一步说明",
    "进一步论证",
    "建议细化",
)

PRINCIPLE_KEYWORDS = (
    "质量守恒",
    "能量守恒",
    "动量守恒",
    "组分守恒",
    "热平衡",
    "物料平衡",
    "时序因果",
    "传热",
    "传质",
    "反应",
    "压力",
    "温度",
    "流量",
    "液位",
    "浓度",
    "泄漏",
    "爆炸",
)


class ReflectionAgent:
    def __init__(self):
        self.vector_store = get_vector_store()

    def run(self, report: dict) -> dict:
        print("\n[智能体3] 物理反思校验...")
        all_issues = []

        fast_issues, fallacy_hits = self._fast_screening(report)
        all_issues.extend(fast_issues)
        if fast_issues:
            print(f"   [fast] 命中 {len(fast_issues)} 条负样本物理谬误")
        else:
            print("   [fast] 未发现负样本物理谬误")

        deep_result = self._deep_verification(report)
        deep_issues = deep_result["issues"]
        all_issues.extend(deep_issues)
        if deep_issues:
            print(f"   [deep] 发现 {len(deep_issues)} 个硬性物理问题")
        else:
            print("   [deep] 守恒与机理校验通过")

        unique_issues = self._deduplicate_issues(all_issues)
        hard_issues = [issue for issue in unique_issues if self._is_hard_issue(issue)]
        checks = deep_result.get("checks", {})
        all_checks_passed = self._checks_all_passed(checks)

        # 硬规则通过条件：
        # 1. 没有命中新的负样本物理谬误
        # 2. 守恒/因果/机理五项检查全部通过
        # 3. 没有硬性物理违规
        passed = len(fallacy_hits) == 0 and all_checks_passed and len(hard_issues) == 0

        if passed:
            print("   [result] 物理校验通过")
        else:
            print(
                f"   [result] 未通过: hard_issues={len(hard_issues)} "
                f"fallacy_hits={len(fallacy_hits)} all_checks_passed={all_checks_passed}"
            )

        return {
            "passed": passed,
            "issues": hard_issues,
            "fallacy_hits": fallacy_hits,
            "correction_guidance": self._build_correction_guidance(hard_issues),
            "checks": checks,
            "all_checks_passed": all_checks_passed,
        }

    def _fast_screening(self, report: dict) -> tuple[list, list]:
        issues = []
        fallacy_hits = []
        propositions = self._extract_propositions(report)

        for prop in propositions:
            try:
                similar_fallacies = self.vector_store.search_similar_fallacies(prop, top_k=3)
                for item in similar_fallacies:
                    similarity = item["similarity"]
                    fallacy = item["fallacy"]
                    fuzzy_score = fuzz.partial_ratio(prop, fallacy["false_claim"]) / 100.0
                    combined_score = max(similarity, fuzzy_score * 0.8)
                    principle_match = self._has_principle_overlap(
                        prop,
                        fallacy.get("physics_principle", []),
                    )

                    # 只在高相似且物理原则关键词也对得上的情况下才判为硬问题。
                    if combined_score >= config.SIMILARITY_THRESHOLD and principle_match:
                        issue = {
                            "issue_type": "物理谬误匹配",
                            "description": (
                                f"命题「{prop[:60]}」与已知物理谬误相似"
                                f"（相似度:{combined_score:.2f}）：{fallacy['false_claim']}"
                            ),
                            "location": "分析内容",
                            "correction_hint": fallacy["correct_understanding"],
                            "source": "fast_screening",
                            "similarity": combined_score,
                        }
                        issues.append(issue)
                        fallacy_hits.append(
                            {
                                "matched_proposition": prop[:80],
                                "fallacy_id": fallacy.get("fallacy_id", ""),
                                "false_claim": fallacy["false_claim"],
                                "correct_understanding": fallacy["correct_understanding"],
                                "similarity": round(combined_score, 3),
                                "category": fallacy.get("category", ""),
                            }
                        )
                        break
            except Exception:
                pass

        return issues, fallacy_hits

    def _deep_verification(self, report: dict) -> dict:
        analysis_text = self._report_to_text(report)
        prompt = DEEP_VERIFY_PROMPT.format(analysis_text=analysis_text)
        try:
            result = call_llm_json(prompt)
            checks = self._normalize_checks(result.get("checks"))
            issues = []
            for item in result.get("issues", []):
                issue = {
                    "issue_type": item.get("issue_type", "物理错误"),
                    "description": item.get("description", ""),
                    "location": item.get("location", ""),
                    "correction_hint": item.get("correction_hint", ""),
                    "source": "deep_verification",
                }
                if self._is_hard_issue(issue):
                    issues.append(issue)
            return {
                "issues": issues,
                "checks": checks,
            }
        except Exception as exc:
            print(f"   [deep] 调用失败: {exc}")
            return {
                "issues": [],
                "checks": {},
            }

    def _extract_propositions(self, report: dict) -> list:
        propositions = []
        for deviation in report.get("deviations", []):
            causes = deviation.get("causes", {})
            if isinstance(causes, list):
                for item in causes:
                    if item.get("description"):
                        propositions.append(item["description"])
            elif isinstance(causes, dict):
                for layer in ["primary", "secondary", "pending"]:
                    for item in causes.get(layer, []):
                        if item.get("description"):
                            propositions.append(item["description"])

            for item in deviation.get("consequences", []):
                if isinstance(item, dict):
                    propositions.append(item.get("description", ""))
                elif isinstance(item, str):
                    propositions.append(item)

        return [item for item in propositions if item]

    def _report_to_text(self, report: dict) -> str:
        lines = []
        node = report.get("node_info", {})
        lines.append(
            f"设备:{node.get('equipment')} 参数:{node.get('parameter')} 偏差:{node.get('deviation_direction')}"
        )
        for deviation in report.get("deviations", []):
            lines.append("\n[原因分析]")
            causes = deviation.get("causes", {})
            if isinstance(causes, list):
                for item in causes:
                    lines.append(f"  [{item.get('type', '')}] {item.get('description', '')}")
            elif isinstance(causes, dict):
                for layer in ["primary", "secondary", "pending"]:
                    for item in causes.get(layer, []):
                        lines.append(
                            f"  [{layer}/{item.get('type', '')}] {item.get('description', '')}"
                        )

            lines.append("\n[后果链]")
            for item in deviation.get("consequences", []):
                desc = item.get("description", item) if isinstance(item, dict) else item
                lines.append(f"  -> {desc}")

        return "\n".join(lines)

    def _deduplicate_issues(self, issues: list) -> list:
        seen = set()
        unique = []
        for issue in issues:
            key = issue.get("description", "")[:80]
            if key and key not in seen:
                seen.add(key)
                unique.append(issue)
        return unique

    def _build_correction_guidance(self, issues: list) -> str:
        if not issues:
            return ""
        lines = []
        for index, issue in enumerate(issues, 1):
            lines.append(f"{index}. [{issue['issue_type']}] {issue['description']}")
            lines.append(f"   修正建议：{issue['correction_hint']}")
        return "\n".join(lines)

    def _normalize_checks(self, checks: dict | None) -> dict:
        if not isinstance(checks, dict):
            return {}
        normalized = {}
        for key in [
            "mass_conservation",
            "energy_conservation",
            "causal_order",
            "numeric_reasonableness",
            "process_mechanism",
        ]:
            normalized[key] = bool(checks.get(key, True))
        return normalized

    def _checks_all_passed(self, checks: dict) -> bool:
        if not checks:
            return True
        return all(bool(value) for value in checks.values())

    def _is_hard_issue(self, issue: dict) -> bool:
        text = " ".join(
            str(issue.get(field, ""))
            for field in ["issue_type", "description", "correction_hint"]
        ).strip()
        if not text:
            return False
        if any(keyword in text for keyword in SOFT_SUGGESTION_KEYWORDS):
            return False
        return any(keyword in text for keyword in HARD_VIOLATION_KEYWORDS)

    def _has_principle_overlap(self, proposition: str, principles: list) -> bool:
        principle_text = " ".join(str(item) for item in principles if item)
        if not principle_text.strip():
            return True

        hits = 0
        for keyword in PRINCIPLE_KEYWORDS:
            if keyword in principle_text and keyword in proposition:
                hits += 1
        return hits > 0
