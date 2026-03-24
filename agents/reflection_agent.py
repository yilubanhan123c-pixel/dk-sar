"""
智能体 3：物理反思智能体
三层校验：
  Layer 1 — 双源对比评分（正样本 vs 负样本相似度对比）
  Layer 2 — 反事实物理验证（对 HAZOP 建议做反事实推演）
  Layer 3 — LLM 综合物理推理（整体报告校验）
"""

from fuzzywuzzy import fuzz

from utils.llm import call_llm, call_llm_json
from utils.vector_store import get_vector_store
import config

# ── 反事实物理验证 Prompt ─────────────────────────────────────────────────
COUNTERFACTUAL_SYSTEM_PROMPT = (
    "你是一位化工安全物理验证专家。你的任务是对HAZOP建议做反事实物理验证。"
    "不要只看建议是否逻辑通顺，而要推演：如果真的执行了这条建议，"
    "物理系统会发生什么变化？是否有被忽略的负面副作用？"
)

COUNTERFACTUAL_USER_PROMPT = """请对以下HAZOP建议做反事实物理验证：
建议内容：{suggestion}
场景上下文：{context}
请按以下三步分析：
Step 1 反事实假设：
如果真的执行了这条建议，物理系统的哪些参数会发生变化？
Step 2 效果推演：
- 正面效果：这条建议预期达到的效果是什么？量级多大？
- 负面副作用：执行后可能产生哪些被忽略的负面影响？量级多大？
- 涉及的物理定律：质量守恒/能量守恒/动量守恒/时序因果
Step 3 净效果判定：
综合正面效果和负面副作用，这条建议执行后：
- 净效果为正（有效）→ 输出 PASS
- 净效果为负（有害或无效）→ 输出 FAIL，并给出修正方向
输出JSON格式：
{{
  "suggestion": "原建议",
  "step1_hypothesis": "反事实假设",
  "step2_positive": "正面效果及量级",
  "step2_negative": "负面副作用及量级",
  "step2_physics_law": "涉及的物理定律",
  "step3_verdict": "PASS或FAIL",
  "step3_net_effect": "净效果分析",
  "correction": "如果FAIL，修正方向是什么"
}}"""

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
    "反事实验证-FAIL",
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
        print("\n🔬 [智能体3] 物理反思校验（双源对比 + 反事实验证）...")
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

        # ── Layer 2: 反事实物理验证（对 HAZOP 建议） ──
        suggestions = self._extract_suggestions(report)
        context_text = self._report_to_text(report)
        counterfactual_results = self._counterfactual_verify(suggestions, context_text)
        cf_issues = []
        for cf in counterfactual_results:
            if cf.get("step3_verdict", "").upper() == "FAIL":
                cf_issues.append({
                    "issue_type": "反事实验证-FAIL",
                    "description": (
                        f"建议「{cf.get('suggestion', '')[:60]}」"
                        f"反事实验证未通过：{cf.get('step3_net_effect', '')[:80]}"
                    ),
                    "location": "建议/措施",
                    "correction_hint": cf.get("correction", "请重新评估该建议的物理可行性"),
                    "source": "counterfactual_verification",
                })
        all_issues.extend(cf_issues)
        cf_fail_count = len(cf_issues)
        cf_pass_count = len(counterfactual_results) - cf_fail_count
        if cf_issues:
            print(f"   🧪 [反事实验证] {len(suggestions)} 条建议中 {cf_fail_count} 条FAIL, {cf_pass_count} 条PASS")
        else:
            print(f"   🧪 [反事实验证] {len(suggestions)} 条建议全部通过反事实物理验证")

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
            and len(cf_issues) == 0
            and all_checks_passed
            and len(hard_issues) == 0
        )

        if passed:
            print("   ✅ [result] 物理校验通过")
        else:
            print(
                f"   ❌ [result] 未通过: hard_issues={len(hard_issues)} "
                f"fallacy_hits={len(fallacy_hits)} cf_fail={len(cf_issues)} "
                f"all_checks_passed={all_checks_passed}"
            )

        return {
            "passed": passed,
            "issues": hard_issues,
            "fallacy_hits": fallacy_hits,
            "counterfactual_results": counterfactual_results,
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

    # ── Layer 2: 反事实物理验证 ──────────────────────────────────────────
    def _extract_suggestions(self, report: dict) -> list:
        """从报告中提取所有 HAZOP 建议（safeguards + recommendations）"""
        suggestions = []
        for deviation in report.get("deviations", []):
            for sg in deviation.get("safeguards", []):
                measure = sg.get("measure", "") if isinstance(sg, dict) else str(sg)
                if measure:
                    suggestions.append(measure)
            recs = deviation.get("recommendations", {})
            if isinstance(recs, dict):
                for level in ["immediate", "short_term", "long_term"]:
                    for rec in recs.get(level, []):
                        action = rec.get("action", "") if isinstance(rec, dict) else str(rec)
                        if action:
                            suggestions.append(action)
            elif isinstance(recs, list):
                for rec in recs:
                    action = rec.get("action", "") if isinstance(rec, dict) else str(rec)
                    if action:
                        suggestions.append(action)
        return suggestions

    def _counterfactual_verify(self, suggestions: list, context: str) -> list:
        """对每条 HAZOP 建议做反事实物理验证"""
        results = []
        for suggestion in suggestions:
            try:
                user_prompt = COUNTERFACTUAL_USER_PROMPT.format(
                    suggestion=suggestion, context=context
                )
                result = call_llm_json(
                    user_prompt,
                    system_prompt=COUNTERFACTUAL_SYSTEM_PROMPT,
                )
                # 确保 suggestion 字段与原始建议一致
                result["suggestion"] = suggestion
                results.append(result)
            except Exception as e:
                print(f"   ⚠️ 反事实验证失败: {e}")
                results.append({
                    "suggestion": suggestion,
                    "step1_hypothesis": "验证失败",
                    "step2_positive": "",
                    "step2_negative": "",
                    "step2_physics_law": "",
                    "step3_verdict": "PASS",
                    "step3_net_effect": "验证调用失败，默认通过",
                    "correction": "",
                })
        return results

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
