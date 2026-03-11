"""
智能体 2：RAG 增强分析智能体
Prompt v5.0 — 基于真实创新的设计：
  - 分层原因（primary/secondary/pending）让诊断逻辑清晰
  - 证据链展示 RAG 推理过程
  - 建议三层分级（立即/短期/长期）
  - 来源标注（案例检索/规则推断/物理推导）
  - 置信度简洁表达，不做虚假量化
"""
import json
from utils.llm import call_llm_json
from utils.vector_store import get_vector_store
import config

RAG_PROMPT = """你是一名资深化工安全分析师，正在执行 HAZOP（危险与可操作性分析）。

## 当前场景
设备: {equipment} | 参数: {parameter}
偏差方向: {deviation_direction} | 描述: {deviation_type}
正常值: {normal_value} | 当前值: {current_value}
其他条件: {other_conditions}

## 参考历史案例（RAG检索结果）
{cases_text}

## 分析要求
{correction_guidance}

## 输出规范（严格遵守）
1. causes 分三层，每层最多2条，末尾加 [案例检索]/[规则推断]/[物理推导]
2. consequences 分直接/升级/极端三个阶段
3. recommendations 分 immediate立即/short_term短期/long_term长期
4. 禁止无来源的精确数字（用"显著升高"代替"升高62%"）
5. confidence 只写 高/中/低 + 一句理由，不需要拆分维度

请严格按以下 JSON 格式输出（不要输出其他内容）：
{{
  "summary": {{
    "core_deviation": "核心偏差一句话",
    "top_suspect": "首要怀疑原因",
    "highest_risk": "最高风险后果",
    "immediate_action": "首要处置动作"
  }},
  "node_info": {{
    "equipment": "{equipment}",
    "parameter": "{parameter}",
    "deviation_type": "{deviation_type}",
    "deviation_direction": "{deviation_direction}",
    "normal_value": "{normal_value}",
    "current_value": "{current_value}"
  }},
  "evidence_chain": [
    {{"step": 1, "type": "输入事实", "content": "用户描述中的关键事实"}},
    {{"step": 2, "type": "规则触发", "content": "触发了哪条 HAZOP 分析规则"}},
    {{"step": 3, "type": "案例支持", "content": "参考案例中的相似模式（注明案例名）"}},
    {{"step": 4, "type": "物理校验", "content": "适用的守恒定律判断"}},
    {{"step": 5, "type": "结论收敛", "content": "综合得出的核心判断"}}
  ],
  "deviations": [{{
    "causes": {{
      "primary": [
        {{"description": "首要原因 [来源标签]", "type": "设备故障/操作失误/工艺异常/管理缺陷"}}
      ],
      "secondary": [
        {{"description": "次要原因 [来源标签]", "type": "类型"}}
      ],
      "pending": [
        {{"description": "待验证原因，需通过XXX方式核实 [来源标签]", "type": "类型"}}
      ]
    }},
    "consequences": [
      {{"stage": "直接后果", "description": "紧接发生的事件"}},
      {{"stage": "升级后果", "description": "未及时处置的演变"}},
      {{"stage": "极端后果", "description": "最坏情况"}}
    ],
    "safeguards": [
      {{
        "measure": "保护措施名称",
        "effectiveness": "有效/部分有效/需改进",
        "effectiveness_reason": "判断理由（一句话）"
      }}
    ],
    "recommendations": {{
      "immediate": [{{"action": "立即处置（分钟级）", "priority": "高"}}],
      "short_term": [{{"action": "短期整改（天级）", "priority": "中"}}],
      "long_term": [{{"action": "长期改进（月级）", "priority": "低"}}]
    }}
  }}],
  "analysis_metadata": {{
    "referenced_cases": {referenced_case_ids},
    "reflection_rounds": {reflection_rounds},
    "physical_issues_found": [],
    "confidence_level": "高/中/低",
    "confidence_reason": "置信度判断依据（一句话）"
  }}
}}"""

CORRECTION_TEMPLATE = """
⚠️ 上一轮存在以下物理问题，本次必须修正：
{issues}
要求：删除或修正违反物理定律的描述，确保因果时序正确，不引入新的物理错误。"""


class RAGAgent:
    def __init__(self):
        self.vector_store = get_vector_store()

    def run(self, context: dict, correction_guidance: str = "",
            reflection_rounds: int = 0) -> dict:
        print(f"\n🔎 [智能体2] RAG 增强分析（第{reflection_rounds+1}轮）...")
        query_text = context.get("query_text", "")

        try:
            similar_cases = self.vector_store.search_similar_cases(query_text, config.TOP_K_CASES)
            print(f"   📚 检索到 {len(similar_cases)} 个相似案例: {[c['case_id'] for c in similar_cases]}")
        except Exception as e:
            print(f"   ⚠️ 案例检索失败: {e}")
            similar_cases = []

        cases_text = self._format_cases(similar_cases)
        referenced_ids = [c["case_id"] for c in similar_cases]
        # 同时保存案例名称，供界面展示
        referenced_names = {c["case_id"]: c.get("name", "") for c in similar_cases}

        guidance_text = (
            CORRECTION_TEMPLATE.format(issues=correction_guidance)
            if correction_guidance
            else "请基于物理化学原理，严格遵守输出规范，生成准确的 HAZOP 分析报告。"
        )

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
            reflection_rounds=reflection_rounds,
        )

        for attempt in range(3):
            try:
                report = call_llm_json(prompt)
                if "analysis_metadata" not in report:
                    report["analysis_metadata"] = {}
                report["analysis_metadata"]["referenced_cases"] = referenced_ids
                report["analysis_metadata"]["referenced_names"] = referenced_names
                report["analysis_metadata"]["reflection_rounds"] = reflection_rounds
                print(f"   ✅ 报告生成成功（参考案例: {referenced_ids}）")
                return report
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"RAG 智能体失败: {e}")
                print(f"   ⚠️ 第{attempt+1}次失败，重试: {e}")

    def _format_cases(self, cases: list) -> str:
        if not cases:
            return "（未找到相似案例，请基于化工安全通用知识分析）"
        texts = []
        for i, case in enumerate(cases, 1):
            texts.append(
                f"案例{i} [{case['case_id']}]：{case['name']} ({case.get('year','N/A')})\n"
                f"  设备: {case.get('equipment')} | 工艺: {case.get('process_type')}\n"
                f"  直接原因: {case.get('causes',{}).get('direct','N/A')}\n"
                f"  主要后果: {'; '.join(case.get('consequences',[])[:2])}\n"
                f"  关键物理机理: {'; '.join(case.get('key_physics',[])[:2])}"
            )
        return "\n\n".join(texts)
