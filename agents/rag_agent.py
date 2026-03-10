"""
智能体 2：RAG 增强分析智能体
功能：检索相似历史案例，参考案例生成 HAZOP 分析报告初稿
对应论文 3.2 节：RAG-Enhanced Analysis Agent
"""
import json
from utils.llm import call_llm_json
from utils.vector_store import get_vector_store
import config


# ── Prompt 模板 ────────────────────────────────────────────────────────────────
RAG_PROMPT = """你是一名资深化工安全分析师，正在执行 HAZOP（危险与可操作性分析）。

## 当前场景
设备: {equipment}
参数: {parameter}
偏差方向: {deviation_direction}
偏差描述: {deviation_type}
正常值: {normal_value}
当前值: {current_value}
其他条件: {other_conditions}

## 参考历史案例（从事故库中检索到的相似案例）
{cases_text}

## 分析要求
参考上述历史案例，对当前场景进行全面的 HAZOP 分析。

{correction_guidance}

请严格按以下 JSON 格式输出（不要输出任何其他内容）：
{{
  "node_info": {{
    "equipment": "{equipment}",
    "parameter": "{parameter}",
    "deviation_type": "{deviation_type}",
    "deviation_direction": "{deviation_direction}",
    "normal_value": "{normal_value}",
    "current_value": "{current_value}"
  }},
  "deviations": [
    {{
      "causes": [
        {{"description": "原因1（详细描述）", "type": "设备故障"}},
        {{"description": "原因2（详细描述）", "type": "操作失误"}},
        {{"description": "原因3（详细描述）", "type": "工艺异常"}}
      ],
      "consequences": [
        "后果1（初始事件）",
        "后果2（扩展后果）",
        "后果3（最终后果）"
      ],
      "safeguards": [
        {{"measure": "保护措施1", "effectiveness": "有效"}},
        {{"measure": "保护措施2", "effectiveness": "部分有效"}}
      ],
      "recommendations": [
        {{"action": "建议措施1（具体可执行的行动）", "priority": "高"}},
        {{"action": "建议措施2", "priority": "中"}}
      ]
    }}
  ],
  "analysis_metadata": {{
    "referenced_cases": {referenced_case_ids},
    "reflection_rounds": {reflection_rounds},
    "physical_issues_found": [],
    "confidence_level": "高"
  }}
}}

要求：
- causes 至少 3 个，最多 5 个，每个都要具体详细
- consequences 至少 3 个，要形成完整因果链
- safeguards 至少 2 个
- recommendations 至少 2 个，优先级从高到低排列
- 所有分析必须基于物理化学原理，不能违反基本定律"""

CORRECTION_GUIDANCE_TEMPLATE = """
⚠️ 上一轮分析存在物理问题，必须在本次修正：
{issues}

修正要求：
- 删除或修正所有违反物理定律的描述
- 确保因果关系的时序正确性
- 数值描述必须在合理范围内"""


class RAGAgent:
    """
    RAG 增强分析智能体
    职责：检索历史案例 + 生成 HAZOP 分析报告
    """
    
    def __init__(self):
        self.vector_store = get_vector_store()
    
    def run(self, context: dict, correction_guidance: str = "", 
            reflection_rounds: int = 0) -> dict:
        """
        主执行函数
        
        输入:
            context: 上下文智能体输出的工艺上下文
            correction_guidance: 物理反思智能体发现的问题（首次为空）
            reflection_rounds: 当前是第几轮修正
        
        输出: HAZOP 分析报告 JSON
        """
        print(f"\n🔎 [智能体2] RAG 增强分析（第{reflection_rounds+1}轮）...")
        
        # ── Step 1: 检索相似案例 ───────────────────────────────
        query_text = context.get("query_text", "")
        
        try:
            similar_cases = self.vector_store.search_similar_cases(query_text, config.TOP_K_CASES)
            print(f"   📚 检索到 {len(similar_cases)} 个相似案例: {[c['case_id'] for c in similar_cases]}")
        except Exception as e:
            print(f"   ⚠️ 案例检索失败: {e}，使用空案例库")
            similar_cases = []
        
        # ── Step 2: 格式化案例文本 ───────────────────────────────
        cases_text = self._format_cases(similar_cases)
        referenced_ids = [c["case_id"] for c in similar_cases]
        
        # ── Step 3: 构建 Prompt ──────────────────────────────────
        if correction_guidance:
            guidance_text = CORRECTION_GUIDANCE_TEMPLATE.format(issues=correction_guidance)
        else:
            guidance_text = "请基于物理化学原理，生成准确的 HAZOP 分析报告。"
        
        prompt = RAG_PROMPT.format(
            equipment=context.get("equipment", "未知设备"),
            parameter=context.get("parameter", "未知参数"),
            deviation_type=context.get("deviation_type", "未知偏差"),
            deviation_direction=context.get("deviation_direction", "其他"),
            normal_value=context.get("normal_value", "未说明"),
            current_value=context.get("current_value", "未说明"),
            other_conditions=context.get("other_conditions", "无"),
            cases_text=cases_text,
            correction_guidance=guidance_text,
            referenced_case_ids=json.dumps(referenced_ids, ensure_ascii=False),
            reflection_rounds=reflection_rounds
        )
        
        # ── Step 4: 调用 LLM 生成报告 ────────────────────────────
        for attempt in range(3):
            try:
                report = call_llm_json(prompt)
                
                # 注入元数据
                if "analysis_metadata" not in report:
                    report["analysis_metadata"] = {}
                report["analysis_metadata"]["referenced_cases"] = referenced_ids
                report["analysis_metadata"]["reflection_rounds"] = reflection_rounds
                
                print(f"   ✅ 报告生成成功（参考案例: {referenced_ids}）")
                return report
                
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"RAG 智能体生成失败: {e}")
                print(f"   ⚠️ 第{attempt+1}次生成失败，重试: {e}")
    
    def _format_cases(self, cases: list) -> str:
        """格式化案例为 Prompt 可用的文本"""
        if not cases:
            return "（未找到相似历史案例，请基于化工安全通用知识分析）"
        
        texts = []
        for i, case in enumerate(cases, 1):
            text = f"""
案例 {i}：{case['name']} ({case.get('year', 'N/A')})
- 设备类型: {case.get('equipment', 'N/A')}
- 工艺类型: {case.get('process_type', 'N/A')}
- 直接原因: {case.get('causes', {}).get('direct', 'N/A')}
- 主要后果: {'; '.join(case.get('consequences', [])[:3])}
- 关键物理机理: {'; '.join(case.get('key_physics', [])[:2])}
- 核心教训: {case.get('causes', {}).get('root', 'N/A')}"""
            texts.append(text)
        
        return "\n".join(texts)
