"""
智能体 3：物理反思智能体
三层校验：
  Layer 1 — 双源对比评分（正样本 vs 负样本相似度对比）
  Layer 2 — NLI 守恒定律深度验证（对 DEEP_VERIFY 命题）
  Layer 3 — LLM 综合物理推理（整体报告校验）
"""

from fuzzywuzzy import fuzz

from utils.llm import call_llm, call_llm_json
from utils.vector_store import get_vector_store
import config

# ── 守恒定律知识库（NLI 验证用） ────────────────────────────────────────────
CONSERVATION_LAWS = {
    "质量守恒": "在封闭系统中，输入物质总量等于输出物质总量加上系统内积累量，物质不会凭空产生或消失",
    "能量守恒": "系统能量输入等于能量输出加上系统内能量积累加上化学反应热，能量不会凭空产生或消失",
    "动量守恒": "在无外力作用的系统中，流体动量守恒，压力变化遵循伯努利方程",
    "组分守恒": "在无化学反应的混合过程中，各组分的质量分别守恒，浓度变化遵循稀释或浓缩规律",
    "热力学第二定律": "热量自发地从高温物体传向低温物体，不可能自发地从低温传向高温；熵在孤立系统中不减少",
    "相平衡": "在气液平衡条件下，温度升高使蒸汽压升高，压力升高使沸点升高，符合克劳修斯-克拉珀龙方程",
}

NLI_CHECK_PROMPT = """你是一名物理定律验证专家。请判断以下"假设"是否与"前提"（物理定律）矛盾。

前提（物理定律）：{premise}

假设（待验证命题）：{hypothesis}

请严格判断假设是否违反前提中的物理定律，输出以下三个标签之一：
- "entailment"：假设与前提一致，不矛盾
- "neutral"：假设与前提无直接关系，无法判断
- "contradiction"：假设明确违反前提中的物理定律

只输出一个JSON：
{{"label": "entailment/neutral/contradiction", "reason": "一句话解释"}}

重要：只有在假设明确违反物理定律时才判定为contradiction。模糊或间接关联判定为neutral。"""

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
2. "还可补充""建议更严谨""可以进一步说明"这类优化建议不算问题，不要写入 issues。
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

# ── 双源对比阈值 ─────────────────────────────────────────────────────────────
CONTRAST_PASS_THRESHOLD = 0.3     # contrast > 0.3 → PASS
CONTRAST_REJECT_THRESHOLD = -0.3  # contrast < -0.3 → REJECT
                                  # 中间地带 → DEEP_VERIFY


class ReflectionAgent:
    def __init__(self):
        self.vector_store = get_vector_store()

    def run(self, report: dict) -> dict:
        print("\n🔬 [智能体3] 物理反思校验（双源对比 + NLI）...")
        all_issues = []

        # ── Layer 1: 双源对比评分 ──
        contrast_results = self._dual_source_contrast(report)
        rejected_issues = [r for r in contrast_results if r["verdict"] == "REJECT"]
        deep_verify_items = [r for r in contrast_results if r["verdict"] == "DEEP_VERIFY"]
        passed_items = [r for r in contrast_results if r["verdict"] == "PASS"]

        print(f"   📊 [双源对比] PASS={len(passed_items)} "
              f"DEEP_VERIFY={len(deep_verify_items)} REJECT={len(rejected_issues)}")

        # 将 REJECT 的命题转化为 issues
        for item in rejected_issues:
            all_issues.append({
                "issue_type": "负样本高相似",
                "description": (
                    f"命题「{item['proposition'][:60]}」与负样本高度相似"
                    f"（对比度:{item['contrast']:.2f}，"
                    f"neg_sim:{item['sim_neg']:.2f}）"
                ),
                "location": "分析内容",
                "correction_hint": item.get("neg_fallacy_hint", "请核查该论断的物理合理性"),
                "source": "dual_source_contrast",
            })

        # 收集 fallacy_hits（用于外部兼容）
        fallacy_hits = []
        for item in rejected_issues:
            if item.get("neg_fallacy_id"):
                fallacy_hits.append({
                    "matched_proposition": item["proposition"][:80],
                    "fallacy_id": item["neg_fallacy_id"],
                    "false_claim": item.get("neg_false_claim", ""),
                    "correct_understanding": item.get("neg_fallacy_hint", ""),
                    "similarity": round(item["sim_neg"], 3),
                    "category": item.get("neg_category", ""),
                    "contrast": round(item["contrast"], 3),
                })

        # ── Layer 2: NLI 深度验证（仅对 DEEP_VERIFY 命题） ──
        nli_issues = self._nli_deep_verify(deep_verify_items)
        all_issues.extend(nli_issues)
        if nli_issues:
            print(f"   🧪 [NLI验证] 发现 {len(nli_issues)} 个守恒定律冲突")
        else:
            print(f"   🧪 [NLI验证] {len(deep_verify_items)} 条命题均通过守恒定律检查")

        # ── Layer 3: LLM 综合物理推理 ──
        deep_result = self._deep_verification(report)
        deep_issues = deep_result["issues"]
        all_issues.extend(deep_issues)
        if deep_issues:
            print(f"   🔬 [深度推理] 发现 {len(deep_issues)} 个硬性物理问题")
        else:
            print("   🔬 [深度推理] 守恒与机理校验通过")

        unique_issues = self._deduplicate_issues(all_issues)
        hard_issues = [issue for issue in unique_issues if self._is_hard_issue(issue)]
        checks = deep_result.get("checks", {})
        all_checks_passed = self._checks_all_passed(checks)

        passed = (
            len(fallacy_hits) == 0
            and len(nli_issues) == 0
            and all_checks_passed
            and len(hard_issues) == 0
        )

        if passed:
            print("   ✅ [result] 物理校验通过")
        else:
            print(
                f"   ❌ [result] 未通过: hard_issues={len(hard_issues)} "
                f"fallacy_hits={len(fallacy_hits)} nli_issues={len(nli_issues)} "
                f"all_checks_passed={all_checks_passed}"
            )

        return {
            "passed": passed,
            "issues": hard_issues,
            "fallacy_hits": fallacy_hits,
            "correction_guidance": self._build_correction_guidance(hard_issues),
            "checks": checks,
            "all_checks_passed": all_checks_passed,
            "contrast_results": contrast_results,
        }

    # ── Layer 1: 双源对比评分 ─────────────────────────────────────────────
    def _dual_source_contrast(self, report: dict) -> list:
        """
        对报告中每条命题做双源对比：
          sim_pos = 与正样本库的最大相似度
          sim_neg = 与负样本库的最大相似度
          contrast = sim_pos - sim_neg
          verdict: PASS / REJECT / DEEP_VERIFY
        """
        propositions = self._extract_propositions(report)
        results = []

        for prop in propositions:
            sim_pos = 0.0
            sim_neg = 0.0
            neg_fallacy_info = {}

            # 正样本相似度
            try:
                pos_scored = self.vector_store.search_similar_cases_with_scores(
                    prop, top_k=3
                )
                if pos_scored:
                    sim_pos = max(r["similarity"] for r in pos_scored)
            except Exception:
                pass

            # 负样本相似度
            try:
                neg_results = self.vector_store.search_similar_fallacies(prop, top_k=3)
                if neg_results:
                    best_neg = max(neg_results, key=lambda x: x["similarity"])
                    sim_neg = best_neg["similarity"]
                    fallacy = best_neg["fallacy"]
                    neg_fallacy_info = {
                        "neg_fallacy_id": fallacy.get("fallacy_id", ""),
                        "neg_false_claim": fallacy.get("false_claim", ""),
                        "neg_fallacy_hint": fallacy.get(
                            "correct_understanding",
                            fallacy.get("correct_explanation", ""),
                        ),
                        "neg_category": fallacy.get("category", ""),
                    }
            except Exception:
                pass

            contrast = sim_pos - sim_neg

            if contrast > CONTRAST_PASS_THRESHOLD:
                verdict = "PASS"
            elif contrast < CONTRAST_REJECT_THRESHOLD:
                verdict = "REJECT"
            else:
                verdict = "DEEP_VERIFY"

            results.append({
                "proposition": prop,
                "sim_pos": sim_pos,
                "sim_neg": sim_neg,
                "contrast": contrast,
                "verdict": verdict,
                **neg_fallacy_info,
            })

        return results

    # ── Layer 2: NLI 守恒定律深度验证 ────────────────────────────────────
    def _nli_deep_verify(self, deep_verify_items: list) -> list:
        """
        对 DEEP_VERIFY 命题逐一用守恒定律做 NLI 检查。
        发现 contradiction 则记录为 issue。
        """
        issues = []
        for item in deep_verify_items:
            prop = item["proposition"]
            for law_name, law_premise in CONSERVATION_LAWS.items():
                try:
                    nli_result = self._llm_nli_check(law_premise, prop)
                    if nli_result.get("label") == "contradiction":
                        issues.append({
                            "issue_type": f"NLI-{law_name}冲突",
                            "description": (
                                f"命题「{prop[:60]}」违反{law_name}："
                                f"{nli_result.get('reason', '无详细原因')}"
                            ),
                            "location": "分析内容",
                            "correction_hint": f"请依据{law_name}（{law_premise[:40]}…）修正该论断",
                            "source": "nli_verification",
                        })
                        # 一旦某条守恒定律判定矛盾，该命题不再检查其余定律
                        break
                except Exception as e:
                    print(f"   ⚠️ NLI 检查失败（{law_name}）: {e}")
                    continue
        return issues

    def _llm_nli_check(self, premise: str, hypothesis: str) -> dict:
        """调用 LLM 做 NLI 三分类判断"""
        prompt = NLI_CHECK_PROMPT.format(premise=premise, hypothesis=hypothesis)
        return call_llm_json(prompt)

    # ── Layer 3: LLM 综合物理推理 ────────────────────────────────────────
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
            print(f"   ⚠️ [深度推理] 调用失败: {exc}")
            return {
                "issues": [],
                "checks": {},
            }

    # ── 辅助方法 ──────────────────────────────────────────────────────────
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
