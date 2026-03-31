"""
智能体 1：HA-FCoT（HAZOP-Aware Focused Chain-of-Thought）智能体
功能：把用户的自然语言输入，通过两阶段推理解析成结构化的 HAZOP 上下文 JSON
对应论文 3.1 节：Context and Schema Agent（改造为 HA-FCoT 版本）

两阶段流程：
  阶段 0（可选）：P&ID 图多模态解析 —— 调用 qwen-vl-max 解析管道仪表流程图
  阶段 1：五维完备性校验抽取 —— 从输入中提取 HAZOP 五个维度（节点/参数/引导词/偏差/上下文）
  阶段 2：IEC 61882 标准链式推理 —— 基于五维信息做标准 HAZOP 逻辑链推理

与旧版的区别：
  - 旧版：XML 中间格式 → 引导词匹配（通用 F-CoT）
  - 新版：五维抽取+自动补全 → IEC 61882 链式推理（HA-FCoT）
"""
from utils.llm import call_llm, call_llm_json, call_llm_vl


# ── HAZOP 五维定义 ────────────────────────────────────────────────────────────
REQUIRED_DIMS = ["node", "parameter", "guideword", "deviation", "context"]

# ── P&ID 图解析 Prompt ───────────────────────────────────────────────────────
PID_PARSE_PROMPT = (
    "你是一个化工P&ID图解析专家。请从图中识别并输出：\n"
    "1. 所有设备节点及编号（如：反应釜R-101、换热器E-201）\n"
    "2. 管道连接关系（从哪到哪，管道编号）\n"
    "3. 仪表配置（温度TI/TC、压力PI/PC、流量FI/FC、液位LI/LC）\n"
    "4. 控制回路（控制器-执行器-被控变量）\n"
    "5. 阀门类型及位置（截止阀、调节阀、安全阀等）\n\n"
    "请用结构化的文字描述输出，尽量包含所有编号和数值信息。"
)

# ── 阶段 1 Prompt：五维完备性抽取 ─────────────────────────────────────────────
STAGE1_EXTRACT_PROMPT = """你是一个HAZOP（危险与可操作性分析）信息提取专家。
请从以下用户描述中提取HAZOP分析所需的五个维度信息。

用户描述：{user_input}

请严格按以下JSON格式输出（不要输出任何其他内容）：
{{
    "node": "节点/设备名称及编号（如：反应釜R-101、换热器E-201）",
    "parameter": "工艺参数（从以下选择：温度/压力/流量/液位/成分/反应速率，或其他具体参数）",
    "guideword": "HAZOP引导词（从以下选择：无/过多/过少/反向/部分/其他）",
    "deviation": "偏差方向描述（如：温度偏高、压力偏低、流量中断）",
    "context": "操作上下文（如：正常运行、开车阶段、停车阶段、检修期间等）"
}}

重要规则：
- 仅提取描述中明确提到或可直接推断的信息
- 如果某项信息在描述中完全没有提及，对应字段填空字符串 ""
- node 应尽量包含设备编号
- guideword 必须从以下选择：无、过多、过少、反向、部分、其他
- 不要添加任何分析或推理内容，只做信息提取"""

# ── 阶段 1 补全 Prompt：自动推断缺失维度 ──────────────────────────────────────
STAGE1_COMPLETE_PROMPT = """你是一个HAZOP分析专家。以下是从用户描述中已提取的信息，但部分维度缺失。
请根据已有信息和化工领域知识，推断补全缺失的维度。

用户原始描述：{user_input}

已提取的信息：
{extracted_json}

缺失的维度：{missing_dims}

请补全所有缺失维度，并输出完整的五维JSON（不要输出任何其他内容）：
{{
    "node": "节点/设备名称及编号",
    "parameter": "工艺参数",
    "guideword": "引导词（无/过多/过少/反向/部分/其他）",
    "deviation": "偏差方向描述",
    "context": "操作上下文"
}}

推断规则：
- 如果缺少node，根据参数和偏差推断最可能涉及的设备
- 如果缺少parameter，根据偏差描述推断工艺参数
- 如果缺少guideword，根据偏差方向匹配最合适的引导词
- 如果缺少deviation，根据参数和引导词组合推断偏差
- 如果缺少context，默认为"正常运行"
- 推断要合理，符合化工实际"""

# ── 阶段 2 Prompt：IEC 61882 标准链式推理 ─────────────────────────────────────
STAGE2_REASONING_PROMPT = """你是一名资深化工安全分析师，请严格按照 IEC 61882 标准执行HAZOP链式推理。

## 当前 HAZOP 五维信息
- 节点/设备：{node}
- 工艺参数：{parameter}
- 引导词：{guideword}
- 偏差：{deviation}
- 操作上下文：{context}

## 推理要求
请按照 IEC 61882 标准的 HAZOP 逻辑链，依次推理以下内容：
1. **原因分析**：该偏差可能由哪些原因导致？（至少列出2个，最多5个）
2. **后果分析**：该偏差如果不处理，会导致什么后果？（从直接后果到最严重后果的升级链）
3. **现有保护措施**：针对该偏差，通常应有哪些安全防护措施？
4. **改进建议**：还需要增加哪些措施来降低风险？

请严格按以下JSON格式输出（不要输出任何其他内容）：
{{
    "deviation": "{deviation}",
    "causes": ["原因1", "原因2", "原因3"],
    "consequences": ["直接后果", "升级后果", "最严重后果"],
    "safeguards": ["现有保护措施1", "现有保护措施2"],
    "recommendations": ["改进建议1", "改进建议2"],
    "risk_level": "高/中/低",
    "reasoning_chain": "简述推理逻辑链（一段话）"
}}

重要规则：
- causes 每个原因要具体，不要泛泛而谈
- consequences 要体现后果升级链（直接→升级→极端）
- safeguards 只列实际工程中常见的保护措施
- recommendations 要有可操作性
- risk_level 根据后果严重程度和发生可能性综合判断"""

# ── 引导词映射：中文 → IEC 61882 英文 ─────────────────────────────────────────
GUIDEWORD_MAP = {
    "无": "NO",
    "过多": "MORE",
    "过少": "LESS",
    "反向": "REVERSE",
    "部分": "PART OF",
    "其他": "OTHER THAN",
}

# ── 引导词 → 偏差方向映射（兼容下游 rag_agent） ─────────────────────────────
GUIDEWORD_TO_DIRECTION = {
    "无": "无/停止",
    "过多": "过高",
    "过少": "过低",
    "反向": "反向",
    "部分": "其他",
    "其他": "其他",
}


class ContextAgent:
    """
    HA-FCoT（HAZOP-Aware Focused Chain-of-Thought）智能体
    职责：
      阶段 0（可选）：P&ID 图 → 结构化文字描述（通义千问VL）
      阶段 1：五维完备性校验抽取（节点/参数/引导词/偏差/上下文）
      阶段 2：IEC 61882 标准链式推理（原因/后果/保护/建议）
    """

    # ── 阶段 0：P&ID 图多模态解析 ────────────────────────────────────────────
    def _parse_pid_image(self, image_url: str) -> str:
        """调用 qwen-vl-max 解析 P&ID 管道仪表流程图，返回结构化文本描述"""
        print("   🖼️ [阶段0] P&ID图解析：图片 → 结构化描述...")
        try:
            description = call_llm_vl(image_url, PID_PARSE_PROMPT)
            print(f"   ✅ P&ID解析完成（{len(description)}字）")
            return description
        except Exception as e:
            print(f"   ⚠️ P&ID解析失败，回退到通用视觉转写: {e}")
            # 回退到通用视觉描述
            fallback_prompt = (
                "请仔细观察这张化工/工艺相关的图片，用中文详细描述图中的内容，"
                "包括：设备名称与类型、物料名称、工艺参数的数值、报警信息、异常状态。"
            )
            description = call_llm_vl(image_url, fallback_prompt)
            print(f"   ✅ 通用转写完成（{len(description)}字）")
            return description

    # ── 阶段 1：五维完备性抽取 ────────────────────────────────────────────────
    def _stage1_extract(self, user_input: str) -> dict:
        """从用户输入中提取 HAZOP 五维信息"""
        print("   📋 [阶段1] 五维抽取：文字 → HAZOP五维JSON...")
        prompt = STAGE1_EXTRACT_PROMPT.format(user_input=user_input)
        extracted = call_llm_json(prompt)
        print(f"   ✅ 五维抽取完成: 节点={extracted.get('node', '?')} "
              f"参数={extracted.get('parameter', '?')}")
        return extracted

    # ── 阶段 1 补全：自动推断缺失维度 ────────────────────────────────────────
    def _auto_complete_dimensions(self, extracted: dict, missing: list,
                                  user_input: str) -> dict:
        """调用LLM补全缺失的HAZOP维度"""
        import json
        print(f"   🔧 [补全] 缺失维度: {missing}，正在自动推断...")
        prompt = STAGE1_COMPLETE_PROMPT.format(
            user_input=user_input,
            extracted_json=json.dumps(extracted, ensure_ascii=False, indent=2),
            missing_dims=", ".join(missing),
        )
        completed = call_llm_json(prompt)

        # 合并：已有值优先，缺失值用补全结果填充
        for dim in REQUIRED_DIMS:
            if not extracted.get(dim) and completed.get(dim):
                extracted[dim] = completed[dim]

        completed_info = ", ".join(f"{d}={extracted.get(d, '?')}" for d in missing)
        print(f"   ✅ 补全完成: {completed_info}")
        return extracted

    # ── 五维完备性校验 ────────────────────────────────────────────────────────
    def _validate_and_complete(self, extracted: dict, user_input: str) -> dict:
        """校验五维完备性，缺失时自动补全"""
        missing = [d for d in REQUIRED_DIMS if not extracted.get(d)]
        if missing:
            extracted = self._auto_complete_dimensions(
                extracted, missing, user_input
            )
            # 二次校验
            still_missing = [d for d in REQUIRED_DIMS if not extracted.get(d)]
            if still_missing:
                print(f"   ⚠️ 补全后仍缺失: {still_missing}，使用默认值填充")
                defaults = {
                    "node": "未识别设备",
                    "parameter": "未识别参数",
                    "guideword": "其他",
                    "deviation": "异常偏差",
                    "context": "正常运行",
                }
                for dim in still_missing:
                    extracted[dim] = defaults[dim]
        return extracted

    # ── 阶段 2：IEC 61882 标准链式推理 ───────────────────────────────────────
    def _stage2_reasoning(self, dims: dict) -> dict:
        """基于五维信息执行 IEC 61882 标准链式推理"""
        print("   🧠 [阶段2] IEC 61882链式推理：五维 → HAZOP分析...")
        prompt = STAGE2_REASONING_PROMPT.format(
            node=dims["node"],
            parameter=dims["parameter"],
            guideword=dims["guideword"],
            deviation=dims["deviation"],
            context=dims["context"],
        )
        result = call_llm_json(prompt)
        print(f"   ✅ 链式推理完成: 原因数={len(result.get('causes', []))} "
              f"后果数={len(result.get('consequences', []))} "
              f"风险={result.get('risk_level', '?')}")
        return result

    # ── 构建兼容下游的 context 字典 ──────────────────────────────────────────
    def _build_compatible_context(self, dims: dict, reasoning: dict) -> dict:
        """
        将 HA-FCoT 结果转换为兼容 rag_agent.py 的 context 格式。
        rag_agent 需要的字段：equipment, parameter, deviation_type,
        deviation_direction, normal_value, current_value, other_conditions, query_text
        """
        guideword_cn = dims.get("guideword", "其他")
        guideword_en = GUIDEWORD_MAP.get(guideword_cn, "OTHER THAN")
        deviation_direction = GUIDEWORD_TO_DIRECTION.get(guideword_cn, "其他")

        # 构建用于 ChromaDB 检索的 embedding_text
        embedding_text = (
            f"{dims['node']} {dims['parameter']} {dims['deviation']} "
            f"{guideword_en} "
            f"{' '.join(reasoning.get('causes', [])[:2])} "
            f"{' '.join(reasoning.get('consequences', [])[:2])}"
        )

        # 构建英文 query_text 用于向量检索
        query_prompt = (
            f"请将以下HAZOP场景翻译为英文关键词（8-15个词）："
            f"设备={dims['node']}，参数={dims['parameter']}，"
            f"偏差={dims['deviation']}，引导词={guideword_en}"
        )
        try:
            query_text = call_llm(query_prompt)
            # 清理可能的引号和多余空格
            query_text = query_text.strip().strip('"').strip("'")
        except Exception:
            # 回退：用基本信息拼接
            query_text = f"{dims['node']} {dims['parameter']} {dims['deviation']} {guideword_en}"

        return {
            # ── 兼容 rag_agent 的字段 ──
            "equipment": dims["node"],
            "parameter": dims["parameter"],
            "deviation_type": dims["deviation"],
            "deviation_direction": deviation_direction,
            "guide_word": guideword_en,
            "normal_value": "未说明",
            "current_value": "未说明",
            "other_conditions": f"操作上下文: {dims['context']}",
            "query_text": query_text,
            # ── HA-FCoT 新增字段 ──
            "hazop_dims": dims,                             # 原始五维信息
            "causes": reasoning.get("causes", []),          # Stage 2 推理的原因
            "consequences": reasoning.get("consequences", []),  # Stage 2 推理的后果
            "safeguards": reasoning.get("safeguards", []),  # Stage 2 推理的保护措施
            "recommendations": reasoning.get("recommendations", []),  # Stage 2 推理的建议
            "risk_level": reasoning.get("risk_level", "中"),
            "reasoning_chain": reasoning.get("reasoning_chain", ""),
            "embedding_text": embedding_text,               # 用于 ChromaDB 检索
        }

    # ── 主执行入口 ────────────────────────────────────────────────────────────
    def run(self, user_input: str, image_url: str = None) -> dict:
        """
        主执行函数（HA-FCoT 两阶段推理）

        参数:
            user_input: 用户自然语言描述
            image_url:  可选，图片URL（P&ID图或工艺截图）
        输出: {
            "context": {HAZOP结构化上下文，兼容rag_agent},
            "query_text": "用于向量检索的文本",
            "scenario_xml": ""  (保留字段，兼容旧接口)
        }
        """
        print(f"\n🔍 [智能体1] HA-FCoT上下文解析中...")
        print(f"   输入: {user_input[:80]}...")

        # 最多重试 3 次
        for attempt in range(3):
            try:
                # ── 阶段 0：P&ID图多模态解析（仅图片输入时）──
                text_input = user_input
                if image_url:
                    pid_desc = self._parse_pid_image(image_url)
                    text_input = f"{user_input}\n\n[P&ID图解析结果] {pid_desc}"

                # ── 阶段 1：五维完备性抽取 ──
                extracted = self._stage1_extract(text_input)

                # 五维完备性校验 + 自动补全
                dims = self._validate_and_complete(extracted, text_input)

                print(f"   📊 五维结果: 节点={dims['node']} | 参数={dims['parameter']} | "
                      f"引导词={dims['guideword']} | 偏差={dims['deviation']} | "
                      f"上下文={dims['context']}")

                # ── 阶段 2：IEC 61882 标准链式推理 ──
                reasoning = self._stage2_reasoning(dims)

                # ── 构建兼容下游的输出 ──
                context = self._build_compatible_context(dims, reasoning)

                print(f"   ✅ HA-FCoT解析成功: {context['equipment']} / "
                      f"{context['parameter']} / {context['guide_word']}")

                return {
                    "context": context,
                    "query_text": context.get("query_text", user_input),
                    "scenario_xml": "",  # 保留字段，兼容旧接口
                }

            except Exception as e:
                if attempt == 2:
                    print(f"   ⚠️ HA-FCoT解析失败，使用默认值: {e}")
                    return {
                        "context": {
                            "equipment": "未识别设备",
                            "parameter": "未识别参数",
                            "deviation_type": "异常偏差",
                            "deviation_direction": "其他",
                            "guide_word": "OTHER THAN",
                            "normal_value": "未说明",
                            "current_value": "未说明",
                            "other_conditions": "无",
                            "query_text": user_input[:100],
                            "hazop_dims": {
                                "node": "未识别设备",
                                "parameter": "未识别参数",
                                "guideword": "其他",
                                "deviation": "异常偏差",
                                "context": "未知",
                            },
                            "causes": [],
                            "consequences": [],
                            "safeguards": [],
                            "recommendations": [],
                            "risk_level": "中",
                            "reasoning_chain": "",
                            "embedding_text": user_input[:200],
                        },
                        "query_text": user_input,
                        "scenario_xml": "",
                    }
                print(f"   ⚠️ 第{attempt + 1}次解析失败，重试中: {e}")
