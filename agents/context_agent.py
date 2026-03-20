"""
智能体 1：上下文与模式智能体（两阶段抽取）
功能：把用户的自然语言输入，解析成结构化的工艺上下文 JSON
对应论文 3.1 节：Context and Schema Agent

两阶段流程：
  阶段 0（可选）：视觉转写 —— 图片输入时调用通义千问VL转为文字
  阶段 1：聚焦抽取 —— 从文字描述中提取关键要素，输出 XML
  阶段 2：引导词推理 —— 基于 XML 按 IEC 61882 匹配引导词，输出 JSON
"""
from utils.llm import call_llm, call_llm_json, call_llm_vl


# ── 视觉转写 Prompt ──────────────────────────────────────────────────────────
VISION_TRANSCRIBE_PROMPT = (
    "请仔细观察这张化工/工艺相关的图片，用中文详细描述图中的内容，"
    "包括：设备名称与类型、物料名称、工艺参数（温度、压力、流量、液位等）的数值、"
    "报警信息、异常状态。只描述看到的事实，不做分析。"
)

# ── 阶段 1 Prompt（聚焦抽取）────────────────────────────────────────────────
STAGE1_EXTRACT_PROMPT = """你是一个化工场景信息提取器。请从以下描述中仅提取关键要素，\
输出为严格的XML格式，不做任何分析或推理：

用户描述：{user_input}

请严格按以下XML格式输出（不要输出任何其他内容）：
<scenario>
  <equipment>设备名</equipment>
  <material>物料名</material>
  <parameter name="参数名" value="当前值" normal="正常值"/>
  <deviation direction="偏差方向"/>
</scenario>

重要规则：
- 仅提取描述中明确提到的信息
- 如果某项未提及，填写"未说明"
- 偏差方向从以下选择：过高、过低、无/停止、反向、其他
- 可以有多个 parameter 标签
- 不要添加任何分析或推理内容"""

# ── 阶段 2 Prompt（引导词推理）────────────────────────────────────────────────
STAGE2_GUIDEWORD_PROMPT = """基于以下结构化场景信息（不要参考其他信息），\
按IEC 61882标准匹配最合适的引导词和偏差类型：

{scenario_xml}

请严格按以下JSON格式输出HAZOP节点定义（不要输出任何其他内容）：
{{
  "equipment": "设备名称",
  "material": "物料名称",
  "parameter": "分析参数",
  "deviation_type": "偏差类型（如：高温、高压、低流量等）",
  "deviation_direction": "偏差方向",
  "guide_word": "IEC 61882引导词（如：MORE, LESS, NO, REVERSE, OTHER THAN等）",
  "normal_value": "正常操作值",
  "current_value": "当前异常值",
  "other_conditions": "其他工艺条件",
  "query_text": "用于向量检索的英文关键词（包含设备类型、偏差类型、引导词，8-15个词）"
}}

重要规则：
- 引导词必须严格来自 IEC 61882 标准（NO, MORE, LESS, AS WELL AS, PART OF, REVERSE, OTHER THAN 等）
- deviation_type 应结合引导词与参数给出（例如 MORE + 温度 → 高温）
- query_text 必须用英文"""


class ContextAgent:
    """
    上下文与模式智能体（两阶段抽取）
    职责：
      阶段 0（可选）：图片 → 文字转写（通义千问VL）
      阶段 1：文字 → XML 结构化要素
      阶段 2：XML → HAZOP JSON 节点定义
    """

    # ── 阶段 0：视觉转写 ──────────────────────────────────────────────────
    def _vision_transcribe(self, image_url: str) -> str:
        """调用通义千问VL将图片转写为文字描述"""
        print("   🖼️ [阶段0] 视觉转写：图片 → 文字描述...")
        description = call_llm_vl(image_url, VISION_TRANSCRIBE_PROMPT)
        print(f"   ✅ 转写完成（{len(description)}字）")
        return description

    # ── 阶段 1：聚焦抽取 ──────────────────────────────────────────────────
    def _stage1_extract(self, user_input: str) -> str:
        """从文字描述中提取关键要素，返回 XML 字符串"""
        print("   📋 [阶段1] 聚焦抽取：文字 → XML...")
        prompt = STAGE1_EXTRACT_PROMPT.format(user_input=user_input)
        xml_text = call_llm(prompt)

        # 清理可能的 Markdown 代码块包裹
        xml_text = xml_text.strip()
        if xml_text.startswith("```"):
            lines = xml_text.split("\n")
            xml_text = "\n".join(lines[1:-1])

        print(f"   ✅ XML 抽取完成")
        return xml_text

    # ── 阶段 2：引导词推理 ────────────────────────────────────────────────
    def _stage2_guideword(self, scenario_xml: str) -> dict:
        """基于 XML 推理 IEC 61882 引导词，返回 HAZOP JSON"""
        print("   🧠 [阶段2] 引导词推理：XML → HAZOP JSON...")
        prompt = STAGE2_GUIDEWORD_PROMPT.format(scenario_xml=scenario_xml)
        result = call_llm_json(prompt)
        print(f"   ✅ 引导词: {result.get('guide_word', '?')} / "
              f"偏差: {result.get('deviation_type', '?')}")
        return result

    # ── 主执行入口 ────────────────────────────────────────────────────────
    def run(self, user_input: str, image_url: str = None) -> dict:
        """
        主执行函数（两阶段抽取）

        参数:
            user_input: 用户自然语言描述
            image_url:  可选，图片URL（提供时先做视觉转写）
        输出: {
            "context": {HAZOP结构化上下文},
            "query_text": "用于向量检索的文本",
            "scenario_xml": "阶段1的XML中间结果"
        }
        """
        print(f"\n🔍 [智能体1] 上下文解析中（两阶段抽取）...")
        print(f"   输入: {user_input[:80]}...")

        # 最多重试 3 次
        for attempt in range(3):
            try:
                # 阶段 0：视觉转写（仅图片输入时）
                text_input = user_input
                if image_url:
                    vision_desc = self._vision_transcribe(image_url)
                    text_input = f"{user_input}\n\n[图片描述] {vision_desc}"

                # 阶段 1：聚焦抽取 → XML
                scenario_xml = self._stage1_extract(text_input)

                # 阶段 2：引导词推理 → JSON
                context = self._stage2_guideword(scenario_xml)

                # 验证必填字段
                required_fields = [
                    "equipment", "parameter", "deviation_direction",
                    "guide_word", "query_text",
                ]
                missing = [f for f in required_fields if not context.get(f)]
                if missing:
                    raise ValueError(f"缺少必填字段: {missing}")

                print(f"   ✅ 解析成功: {context['equipment']} / "
                      f"{context['parameter']} / {context['guide_word']}")

                return {
                    "context": context,
                    "query_text": context.get("query_text", user_input),
                    "scenario_xml": scenario_xml,
                }

            except Exception as e:
                if attempt == 2:
                    print(f"   ⚠️ 解析失败，使用默认值: {e}")
                    return {
                        "context": {
                            "equipment": "未识别设备",
                            "material": "未识别物料",
                            "parameter": "未识别参数",
                            "deviation_type": "异常偏差",
                            "deviation_direction": "其他",
                            "guide_word": "OTHER THAN",
                            "normal_value": "未说明",
                            "current_value": "未说明",
                            "other_conditions": "无",
                            "query_text": user_input[:100],
                        },
                        "query_text": user_input,
                        "scenario_xml": "",
                    }
                print(f"   ⚠️ 第{attempt + 1}次解析失败，重试中...")
