"""
智能体 2：RAG 增强分析智能体
Prompt v6.0 — 质量分级检索 + 负样本警示：
  - 检索质量三级评估（高/中/低），中等时触发查询改写
  - 负样本库检索相关物理谬误，作为警示注入 Prompt
  - 分层原因（primary/secondary/pending）让诊断逻辑清晰
  - 证据链展示 RAG 推理过程
  - 建议三层分级（立即/短期/长期）
  - 来源标注（案例检索/规则推断/物理推导）
  - 置信度简洁表达，不做虚假量化
"""
import json
from utils.llm import call_llm, call_llm_json
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

## 注意避免以下物理谬误
{fallacies_warning}

## 分析要求
{correction_guidance}

## 输出规范（严格遵守）
1. causes 分三层，每层最多2条，末尾加 [案例检索]/[规则推断]/[物理推导]
2. consequences 分直接/升级/极端三个阶段
3. recommendations 分 immediate立即/short_term短期/long_term长期
4. 禁止无来源的精确数字（用"显著升高"代替"升高62%"）
5. confidence 只写 高/中/低 + 一句理由，不需要拆分维度
6. 分析中不得出现"注意避免"章节中列出的错误论断

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
    "retrieval_quality": "{retrieval_quality}",
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

QUERY_REWRITE_PROMPT = """你是化工安全领域的检索优化专家。
当前查询未能找到高质量匹配案例，请改写查询以提高检索召回率。

原始查询：{original_query}
场景上下文：设备={equipment}，参数={parameter}，偏差={deviation_direction}

请输出一个改写后的英文查询（8-15个词），要求：
- 使用更通用的化工安全术语
- 扩展同义词和上位概念
- 保留核心偏差语义
只输出改写后的查询文本，不要输出其他内容。"""

# ── 检索质量阈值 ──────────────────────────────────────────────────────────────
QUALITY_HIGH_THRESHOLD = 0.6    # 平均相似度 > 0.6 → 高质量，直接使用
QUALITY_MED_THRESHOLD = 0.4     # 平均相似度 > 0.4 → 中等，触发查询改写
                                # 平均相似度 ≤ 0.4 → 低质量，零样本模式


class RAGAgent:
    def __init__(self):
        self.vector_store = get_vector_store()

    # ── 检索质量评估 + 分级处理 ──────────────────────────────────────────
    def _retrieve_with_quality_assessment(self, query_text: str,
                                          context: dict) -> tuple:
        """
        三级质量评估检索：
          高质量 (avg_sim > 0.6): 直接使用 top-3
          中等 (avg_sim > 0.4):   LLM 改写查询后重新检索
          低质量 (avg_sim ≤ 0.4): 零样本，不注入案例
        返回: (selected_cases, quality_label)
        """
        scored_results = self.vector_store.search_similar_cases_with_scores(
            query_text, top_k=5
        )
        if not scored_results:
            return [], "low"

        avg_sim = sum(r["similarity"] for r in scored_results) / len(scored_results)
        print(f"   📊 检索质量评估: 平均相似度={avg_sim:.3f}")

        # ── 高质量匹配 ──
        if avg_sim > QUALITY_HIGH_THRESHOLD:
            selected = [r["case"] for r in scored_results[:3]]
            print(f"   ✅ 高质量匹配，直接使用 top-3 案例")
            return selected, "high"

        # ── 中等质量 → 查询改写 ──
        if avg_sim > QUALITY_MED_THRESHOLD:
            print(f"   🔄 中等质量，触发查询改写...")
            enhanced_query = self._rewrite_query(query_text, context)
            print(f"   🔄 改写后查询: {enhanced_query[:80]}")

            rewritten_results = self.vector_store.search_similar_cases_with_scores(
                enhanced_query, top_k=5
            )
            if rewritten_results:
                rewritten_avg = sum(r["similarity"] for r in rewritten_results) / len(rewritten_results)
                print(f"   📊 改写后相似度: {rewritten_avg:.3f}")
                # 如果改写后更好就用改写结果，否则用原始结果
                if rewritten_avg > avg_sim:
                    selected = [r["case"] for r in rewritten_results[:3]]
                else:
                    selected = [r["case"] for r in scored_results[:3]]
            else:
                selected = [r["case"] for r in scored_results[:3]]
            return selected, "medium"

        # ── 低质量 → 零样本 ──
        print(f"   ⚠️ 低质量匹配（avg={avg_sim:.3f}），进入零样本模式")
        return [], "low"

    def _rewrite_query(self, original_query: str, context: dict) -> str:
        """调用 LLM 改写检索查询以提高召回率"""
        prompt = QUERY_REWRITE_PROMPT.format(
            original_query=original_query,
            equipment=context.get("equipment", "未知"),
            parameter=context.get("parameter", "未知"),
            deviation_direction=context.get("deviation_direction", "未知"),
        )
        return call_llm(prompt).strip()

    # ── 负样本警示检索 ────────────────────────────────────────────────────
    def _retrieve_fallacy_warnings(self, query_text: str) -> str:
        """从负样本库检索 2-3 条最相关物理谬误，格式化为警示文本"""
        try:
            fallacies = self.vector_store.search_similar_fallacies(query_text, top_k=3)
            if not fallacies:
                return "（无特定物理谬误警示）"

            # 取相似度最高的 2-3 条
            relevant = [f for f in fallacies if f["similarity"] > 0.3][:3]
            if not relevant:
                return "（无特定物理谬误警示）"

            lines = []
            for i, item in enumerate(relevant, 1):
                f = item["fallacy"]
                lines.append(
                    f"⚠️ 谬误{i} [{f.get('category', '未知类别')}]：\n"
                    f"  错误论断: {f.get('false_claim', 'N/A')}\n"
                    f"  正确解释: {f.get('correct_explanation', 'N/A')}\n"
                    f"  适用场景: {f.get('applicable_scenario', 'N/A')}"
                )
            print(f"   🚫 检索到 {len(relevant)} 条相关物理谬误作为警示")
            return "\n\n".join(lines)

        except Exception as e:
            print(f"   ⚠️ 负样本检索失败: {e}")
            return "（负样本检索不可用）"

    # ── 主执行入口 ────────────────────────────────────────────────────────
    def run(self, context: dict, correction_guidance: str = "",
            reflection_rounds: int = 0) -> dict:
        print(f"\n🔎 [智能体2] RAG 增强分析（第{reflection_rounds+1}轮）...")
        query_text = context.get("query_text", "")

        # ── 质量分级检索正样本 ──
        try:
            similar_cases, retrieval_quality = \
                self._retrieve_with_quality_assessment(query_text, context)
            print(f"   📚 最终选用 {len(similar_cases)} 个案例"
                  f"（质量: {retrieval_quality}）: "
                  f"{[c['case_id'] for c in similar_cases]}")
        except Exception as e:
            print(f"   ⚠️ 案例检索失败: {e}")
            similar_cases = []
            retrieval_quality = "low"

        # ── 检索负样本警示 ──
        fallacies_warning = self._retrieve_fallacy_warnings(query_text)

        cases_text = self._format_cases(similar_cases)
        referenced_ids = [c["case_id"] for c in similar_cases]
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
            fallacies_warning=fallacies_warning,
            correction_guidance=guidance_text,
            referenced_case_ids=json.dumps(referenced_ids, ensure_ascii=False),
            retrieval_quality=retrieval_quality,
            reflection_rounds=reflection_rounds,
        )

        for attempt in range(3):
            try:
                report = call_llm_json(prompt)
                if "analysis_metadata" not in report:
                    report["analysis_metadata"] = {}
                report["analysis_metadata"]["referenced_cases"] = referenced_ids
                report["analysis_metadata"]["referenced_names"] = referenced_names
                report["analysis_metadata"]["retrieval_quality"] = retrieval_quality
                report["analysis_metadata"]["reflection_rounds"] = reflection_rounds
                print(f"   ✅ 报告生成成功（参考案例: {referenced_ids}，"
                      f"检索质量: {retrieval_quality}）")
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
