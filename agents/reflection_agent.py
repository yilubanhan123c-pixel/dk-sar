"""
智能体 3：物理反思智能体
双层校验：Layer1 负样本库匹配 + Layer2 LLM物理推理
修复：兼容新版分层causes格式（dict），并暴露负样本命中详情供界面展示
"""
import json
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

规则：只报告确实违反物理定律的错误；issues 为空时 has_issues 必须为 false。"""


class ReflectionAgent:
    def __init__(self):
        self.vector_store = get_vector_store()

    def run(self, report: dict) -> dict:
        print(f"\n🔬 [智能体3] 物理反思校验...")
        all_issues = []

        # Layer 1：负样本库快速筛查
        fast_issues, fallacy_hits = self._fast_screening(report)
        all_issues.extend(fast_issues)
        if fast_issues:
            print(f"   ⚠️ 快速筛查命中 {len(fast_issues)} 条物理谬误")
        else:
            print(f"   ✅ 快速筛查通过")

        # Layer 2：LLM深度验证
        deep_issues = self._deep_verification(report)
        all_issues.extend(deep_issues)
        if deep_issues:
            print(f"   ⚠️ 深度验证发现 {len(deep_issues)} 个物理问题")
        else:
            print(f"   ✅ 深度验证通过")

        unique_issues = self._deduplicate_issues(all_issues)
        passed = len(unique_issues) == 0
        correction_guidance = self._build_correction_guidance(unique_issues)

        if passed:
            print(f"   🎉 物理校验通过！")
        else:
            print(f"   📋 共发现 {len(unique_issues)} 个问题，需要修正")

        return {
            "passed": passed,
            "issues": unique_issues,
            "fallacy_hits": fallacy_hits,   # ← 新增：暴露负样本命中详情
            "correction_guidance": correction_guidance,
        }

    def _fast_screening(self, report: dict) -> tuple[list, list]:
        """
        Layer 1：负样本库向量匹配
        返回 (issues列表, fallacy_hits详情列表)
        fallacy_hits 用于界面展示"双知识库"的负样本命中情况
        """
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

                    if combined_score >= config.SIMILARITY_THRESHOLD:
                        issue = {
                            "issue_type": "物理谬误匹配",
                            "description": f"命题「{prop[:60]}」与已知物理谬误相似（相似度:{combined_score:.2f}）：{fallacy['false_claim']}",
                            "location": "分析内容",
                            "correction_hint": fallacy["correct_understanding"],
                            "source": "fast_screening",
                            "similarity": combined_score,
                        }
                        issues.append(issue)
                        # 记录命中详情，供界面展示
                        fallacy_hits.append({
                            "matched_proposition": prop[:80],
                            "fallacy_id": fallacy.get("fallacy_id", ""),
                            "false_claim": fallacy["false_claim"],
                            "correct_understanding": fallacy["correct_understanding"],
                            "similarity": round(combined_score, 3),
                            "category": fallacy.get("category", ""),
                        })
                        break
            except Exception:
                pass

        return issues, fallacy_hits

    def _deep_verification(self, report: dict) -> list:
        """Layer 2：LLM五项物理规则深度验证"""
        analysis_text = self._report_to_text(report)
        prompt = DEEP_VERIFY_PROMPT.format(analysis_text=analysis_text)
        try:
            result = call_llm_json(prompt)
            if result.get("has_issues") and result.get("issues"):
                return [{
                    "issue_type": i.get("issue_type", "物理错误"),
                    "description": i.get("description", ""),
                    "location": i.get("location", ""),
                    "correction_hint": i.get("correction_hint", ""),
                    "source": "deep_verification",
                } for i in result["issues"]]
        except Exception as e:
            print(f"   ⚠️ 深度验证调用失败: {e}")
        return []

    def _extract_propositions(self, report: dict) -> list:
        """提取报告关键命题，兼容新版分层causes格式"""
        propositions = []
        for deviation in report.get("deviations", []):
            causes = deviation.get("causes", {})
            # 兼容旧格式（list）和新格式（dict分层）
            if isinstance(causes, list):
                for c in causes:
                    if c.get("description"):
                        propositions.append(c["description"])
            elif isinstance(causes, dict):
                for layer in ["primary", "secondary", "pending"]:
                    for c in causes.get(layer, []):
                        if c.get("description"):
                            propositions.append(c["description"])
            # 后果链
            for c in deviation.get("consequences", []):
                if isinstance(c, dict):
                    propositions.append(c.get("description", ""))
                elif isinstance(c, str):
                    propositions.append(c)
        return [p for p in propositions if p]

    def _report_to_text(self, report: dict) -> str:
        """展平报告为可读文本，兼容新版格式"""
        lines = []
        node = report.get("node_info", {})
        lines.append(f"设备:{node.get('equipment')} 参数:{node.get('parameter')} 偏差:{node.get('deviation_direction')}")
        for deviation in report.get("deviations", []):
            lines.append("\n【原因分析】")
            causes = deviation.get("causes", {})
            if isinstance(causes, list):
                for c in causes:
                    lines.append(f"  [{c.get('type','')}] {c.get('description','')}")
            elif isinstance(causes, dict):
                for layer in ["primary", "secondary", "pending"]:
                    for c in causes.get(layer, []):
                        lines.append(f"  [{layer}/{c.get('type','')}] {c.get('description','')}")
            lines.append("\n【后果链】")
            for c in deviation.get("consequences", []):
                desc = c.get("description", c) if isinstance(c, dict) else c
                lines.append(f"  → {desc}")
        return "\n".join(lines)

    def _deduplicate_issues(self, issues: list) -> list:
        seen, unique = set(), []
        for issue in issues:
            key = issue["description"][:50]
            if key not in seen:
                seen.add(key)
                unique.append(issue)
        return unique

    def _build_correction_guidance(self, issues: list) -> str:
        if not issues:
            return ""
        lines = []
        for i, issue in enumerate(issues, 1):
            lines.append(f"{i}. [{issue['issue_type']}] {issue['description']}")
            lines.append(f"   修正建议：{issue['correction_hint']}")
        return "\n".join(lines)
