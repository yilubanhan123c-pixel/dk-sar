"""
DK-SAR Prompt 管理模块

将所有 Prompt 集中管理，支持版本控制和 A/B 测试。
对应 AI 产品经理核心技能：Prompt 工程与 AI 行为规则设计。

Prompt 设计原则（记录在此，方便迭代）：
  1. 角色设定：明确告知模型扮演的专家身份
  2. 输入格式：结构化输入减少歧义
  3. 输出约束：严格 JSON Schema 保证格式稳定
  4. 少样本示例：关键步骤给出参考示例
  5. 负面约束：明确告知模型不能做什么
"""

from dataclasses import dataclass, field
from typing import Optional
import json


# ════════════════════════════════════════════════════════════════
#  Prompt 版本定义
#  修改 Prompt 时，version + 1，并在 changelog 记录原因
#  这是产品经理管理 AI 行为规则的标准做法
# ════════════════════════════════════════════════════════════════

@dataclass
class PromptTemplate:
    name: str           # Prompt 名称
    version: str        # 版本号
    template: str       # Prompt 模板文本
    changelog: str      # 修改记录
    variables: list     # 需要填入的变量名


# ════════════════════════════════════════════════════════════════
#  智能体 1：上下文解析 Prompt
# ════════════════════════════════════════════════════════════════

CONTEXT_PROMPT_V1 = PromptTemplate(
    name="context_extraction",
    version="1.2",
    changelog="""
    v1.0: 初始版本，基础字段提取
    v1.1: 增加 other_conditions 字段，捕获搅拌状态等附加信息
    v1.2: 增加 query_text 英文关键词字段，提升向量检索精度
    """,
    variables=["user_input"],
    template="""你是一个化工工艺分析助手。请从用户的描述中提取结构化工艺信息。

用户描述：{user_input}

请严格按以下 JSON 格式输出（不要输出任何其他内容）：
{{
  "equipment": "设备名称（如：反应釜、换热器、储罐、管道、分馏塔）",
  "parameter": "分析参数（如：温度、压力、流量、液位、浓度）",
  "deviation_type": "偏差类型（简洁描述）",
  "deviation_direction": "偏差方向（从以下选一个：过高、过低、无/停止、反向、其他）",
  "normal_value": "正常操作值（如有提及，否则填未说明）",
  "current_value": "当前异常值（如有提及，否则填未说明）",
  "other_conditions": "其他工艺条件（如搅拌状态、其他设备状态，没有则填无）",
  "query_text": "用于向量检索的英文关键词（包含设备类型、偏差类型、关键参数，8-15个词）"
}}

规则：所有字段必须填写；如描述模糊，根据化工常识合理推断；query_text 必须英文。"""
)


# ════════════════════════════════════════════════════════════════
#  智能体 2：RAG 分析生成 Prompt
# ════════════════════════════════════════════════════════════════

RAG_PROMPT_V1 = PromptTemplate(
    name="rag_analysis",
    version="2.1",
    changelog="""
    v1.0: 初始版本，基础 HAZOP 生成
    v2.0: 加入参考案例结构化输入，显著提升原因分析质量
    v2.1: 加入 correction_guidance 修正指令槽位，支持物理反思迭代
    """,
    variables=[
        "equipment", "parameter", "deviation_type", "deviation_direction",
        "normal_value", "current_value", "other_conditions",
        "cases_text", "correction_guidance",
        "referenced_case_ids", "reflection_rounds"
    ],
    template="""你是一名资深化工安全分析师，正在执行 HAZOP（危险与可操作性分析）。

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
  "deviations": [{{
    "causes": [
      {{"description": "原因1（详细，含物理机理）", "type": "设备故障"}},
      {{"description": "原因2", "type": "操作失误"}},
      {{"description": "原因3", "type": "工艺异常"}}
    ],
    "consequences": ["初始事件", "扩展后果", "最终后果"],
    "safeguards": [
      {{"measure": "保护措施1", "effectiveness": "有效"}}
    ],
    "recommendations": [
      {{"action": "建议措施1（具体可执行）", "priority": "高"}}
    ]
  }}],
  "analysis_metadata": {{
    "referenced_cases": {referenced_case_ids},
    "reflection_rounds": {reflection_rounds},
    "physical_issues_found": [],
    "confidence_level": "高"
  }}
}}

要求：causes ≥3个；consequences ≥3条形成完整因果链；所有分析必须符合物理化学定律。"""
)


# ════════════════════════════════════════════════════════════════
#  智能体 3：物理反思验证 Prompt
# ════════════════════════════════════════════════════════════════

REFLECTION_PROMPT_V1 = PromptTemplate(
    name="physical_reflection",
    version="1.3",
    changelog="""
    v1.0: 初始版本，笼统检查
    v1.1: 拆分为五个明确检查点，提升检查精度
    v1.2: 增加 location 字段定位问题位置
    v1.3: 增加 correction_hint 字段，让修正建议更具体
    """,
    variables=["analysis_text"],
    template="""你是一名严格的物理审查专家。请检查以下 HAZOP 分析中是否存在物理错误。

## 待检查的分析内容
{analysis_text}

## 检查五个维度（逐一核查）
1. 质量守恒：物料平衡是否正确？
2. 能量守恒：热量产生和散失因果关系是否正确？
3. 动量守恒：流体行为描述是否合理？
4. 时序因果：原因是否发生在结果之前，有无因果颠倒？
5. 数值合理性：温度、压力、浓度等数值是否在合理范围内？

## 输出格式（JSON）
{{
  "has_issues": true 或 false,
  "issues": [
    {{
      "issue_type": "问题类型（质量守恒/能量守恒/动量守恒/时序因果/数值合理性）",
      "description": "发现的具体问题描述",
      "location": "问题出现在哪个字段",
      "correction_hint": "如何修正的具体建议"
    }}
  ],
  "summary": "总体评价（一句话）"
}}

规则：只报告确实违反物理定律的错误；issues 为空时 has_issues 必须为 false。"""
)


# ════════════════════════════════════════════════════════════════
#  Prompt 管理器：对外接口
# ════════════════════════════════════════════════════════════════

class PromptManager:
    """
    统一管理所有 Prompt 模板
    
    设计思路（AI 产品视角）：
    - 集中管理便于版本控制和 A/B 测试
    - 修改 AI 行为只需改这一个文件
    - changelog 记录每次迭代的原因，方便复盘
    """

    _registry = {
        "context":    CONTEXT_PROMPT_V1,
        "rag":        RAG_PROMPT_V1,
        "reflection": REFLECTION_PROMPT_V1,
    }

    @classmethod
    def get(cls, name: str) -> PromptTemplate:
        """获取 Prompt 模板"""
        if name not in cls._registry:
            raise KeyError(f"未知 Prompt: {name}，可用: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def render(cls, name: str, **kwargs) -> str:
        """渲染 Prompt，填入变量"""
        tmpl = cls.get(name)
        try:
            return tmpl.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Prompt '{name}' 缺少变量: {e}")

    @classmethod
    def list_all(cls) -> dict:
        """列出所有 Prompt 及其版本"""
        return {
            name: {"version": t.version, "variables": t.variables}
            for name, t in cls._registry.items()
        }

    @classmethod
    def export_for_review(cls) -> str:
        """
        导出所有 Prompt 为可读文本
        用于与算法/研发团队对齐 AI 行为规则（对应 JD 第2条）
        """
        lines = ["=" * 60, "DK-SAR Prompt 规则文档", "=" * 60]
        for name, tmpl in cls._registry.items():
            lines.append(f"\n## {name.upper()} (v{tmpl.version})")
            lines.append(f"变更记录: {tmpl.changelog.strip()}")
            lines.append(f"输入变量: {tmpl.variables}")
            lines.append(f"\n{tmpl.template[:200]}...")
        return "\n".join(lines)


# ── 命令行查看所有 Prompt ──────────────────────────────────────
if __name__ == "__main__":
    print(PromptManager.export_for_review())
    print("\n所有 Prompt 版本:")
    print(json.dumps(PromptManager.list_all(), ensure_ascii=False, indent=2))
