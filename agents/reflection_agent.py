"""
智能体 3：物理反思智能体
功能：对 RAG 生成的报告进行物理一致性校验，发现错误触发修正
对应论文 3.3 节：Physical Reflection Agent（双层级联架构）
"""
import json
from fuzzywuzzy import fuzz
from utils.llm import call_llm_json
from utils.vector_store import get_vector_store
import config


# ── 深度验证 Prompt ────────────────────────────────────────────────────────────
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
      "issue_type": "问题类型（质量守恒/能量守恒/时序因果/数值合理性/过程机理）",
      "description": "发现的具体问题描述",
      "location": "问题出现在哪个字段（如：causes[0].description）",
      "correction_hint": "如何修正的建议"
    }}
  ],
  "summary": "总体评价（一句话）"
}}

重要规则：
- 只报告确实违反物理定律的错误，不要挑剔措辞
- issues 为空列表时 has_issues 必须为 false
- 每个 issue 必须有具体的 correction_hint"""


class ReflectionAgent:
    """
    物理反思智能体
    职责：双层物理校验（快速筛查 + 深度验证）
    """
    
    def __init__(self):
        self.vector_store = get_vector_store()
    
    def run(self, report: dict) -> dict:
        """
        主执行函数
        
        输入: RAG 智能体生成的 HAZOP 报告
        输出: {
            "passed": True/False,  # 是否通过物理校验
            "issues": [...],       # 发现的问题列表
            "correction_guidance": "修正指导文本（传给 RAG 智能体）"
        }
        """
        print(f"\n🔬 [智能体3] 物理反思校验...")
        
        all_issues = []
        
        # ── Layer 1: 快速筛查（向量 + 模糊匹配）─────────────────
        fast_issues = self._fast_screening(report)
        all_issues.extend(fast_issues)
        
        if fast_issues:
            print(f"   ⚠️ 快速筛查发现 {len(fast_issues)} 个疑似谬误")
        else:
            print(f"   ✅ 快速筛查通过")
        
        # ── Layer 2: 深度验证（LLM 物理推理）────────────────────
        deep_issues = self._deep_verification(report)
        all_issues.extend(deep_issues)
        
        if deep_issues:
            print(f"   ⚠️ 深度验证发现 {len(deep_issues)} 个物理问题")
        else:
            print(f"   ✅ 深度验证通过")
        
        # 去重
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
            "correction_guidance": correction_guidance
        }
    
    def _fast_screening(self, report: dict) -> list:
        """
        Layer 1：快速筛查
        把报告中的关键命题与负样本库做相似度匹配
        """
        issues = []
        
        # 提取报告中的关键命题文本
        propositions = self._extract_propositions(report)
        
        for prop in propositions:
            try:
                # 向量相似度检索
                similar_fallacies = self.vector_store.search_similar_fallacies(prop, top_k=3)
                
                for item in similar_fallacies:
                    similarity = item["similarity"]
                    fallacy = item["fallacy"]
                    
                    # 额外做模糊字符串匹配（处理中文）
                    fuzzy_score = fuzz.partial_ratio(prop, fallacy["false_claim"]) / 100.0
                    combined_score = max(similarity, fuzzy_score * 0.8)
                    
                    if combined_score >= config.SIMILARITY_THRESHOLD:
                        issues.append({
                            "issue_type": "物理谬误匹配",
                            "description": f"命题「{prop[:60]}」与已知物理谬误高度相似（相似度:{combined_score:.2f}）：{fallacy['false_claim']}",
                            "location": "分析内容",
                            "correction_hint": fallacy["correct_understanding"],
                            "source": "fast_screening",
                            "similarity": combined_score
                        })
                        break  # 一个命题只报一个最高相似度的谬误
            except Exception as e:
                pass  # 检索失败时跳过，不中断流程
        
        return issues
    
    def _deep_verification(self, report: dict) -> list:
        """
        Layer 2：深度验证
        用 LLM 逐一检查五个物理检查点
        """
        analysis_text = self._report_to_text(report)
        
        prompt = DEEP_VERIFY_PROMPT.format(analysis_text=analysis_text)
        
        try:
            result = call_llm_json(prompt)
            
            if result.get("has_issues") and result.get("issues"):
                issues = []
                for issue in result["issues"]:
                    issues.append({
                        "issue_type": issue.get("issue_type", "物理错误"),
                        "description": issue.get("description", ""),
                        "location": issue.get("location", ""),
                        "correction_hint": issue.get("correction_hint", ""),
                        "source": "deep_verification"
                    })
                return issues
        except Exception as e:
            print(f"   ⚠️ 深度验证调用失败: {e}")
        
        return []
    
    def _extract_propositions(self, report: dict) -> list:
        """从报告中提取关键命题文本（用于向量匹配）"""
        propositions = []
        
        for deviation in report.get("deviations", []):
            # 原因描述
            for cause in deviation.get("causes", []):
                desc = cause.get("description", "")
                if desc:
                    propositions.append(desc)
            
            # 后果描述
            for consequence in deviation.get("consequences", []):
                if consequence:
                    propositions.append(consequence)
        
        return propositions
    
    def _report_to_text(self, report: dict) -> str:
        """将报告 JSON 转为可读文本（用于 LLM 审查）"""
        lines = []
        node_info = report.get("node_info", {})
        lines.append(f"设备: {node_info.get('equipment')}，参数: {node_info.get('parameter')}，偏差: {node_info.get('deviation_direction')}")
        
        for i, deviation in enumerate(report.get("deviations", []), 1):
            lines.append(f"\n【原因分析】")
            for j, cause in enumerate(deviation.get("causes", []), 1):
                lines.append(f"  {j}. [{cause.get('type')}] {cause.get('description')}")
            
            lines.append(f"\n【后果链】")
            for consequence in deviation.get("consequences", []):
                lines.append(f"  → {consequence}")
            
            lines.append(f"\n【保护措施】")
            for sg in deviation.get("safeguards", []):
                lines.append(f"  - {sg.get('measure')} ({sg.get('effectiveness')})")
            
            lines.append(f"\n【建议措施】")
            for rec in deviation.get("recommendations", []):
                lines.append(f"  [{rec.get('priority')}] {rec.get('action')}")
        
        return "\n".join(lines)
    
    def _deduplicate_issues(self, issues: list) -> list:
        """简单去重，避免快速筛查和深度验证报告同一个问题"""
        seen = set()
        unique = []
        for issue in issues:
            key = issue["description"][:50]
            if key not in seen:
                seen.add(key)
                unique.append(issue)
        return unique
    
    def _build_correction_guidance(self, issues: list) -> str:
        """将问题列表整理成修正指导文本"""
        if not issues:
            return ""
        
        lines = []
        for i, issue in enumerate(issues, 1):
            lines.append(f"{i}. [{issue['issue_type']}] {issue['description']}")
            lines.append(f"   修正建议：{issue['correction_hint']}")
        
        return "\n".join(lines)
