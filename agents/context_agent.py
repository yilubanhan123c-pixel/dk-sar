"""
智能体 1：上下文与模式智能体
功能：把用户的自然语言输入，解析成结构化的工艺上下文 JSON
对应论文 3.1 节：Context and Schema Agent
"""
import json
from utils.llm import call_llm_json


# ── Prompt 模板 ────────────────────────────────────────────────────────────────
CONTEXT_PROMPT = """你是一个化工工艺分析助手。请从用户的描述中提取结构化工艺信息。

用户描述：{user_input}

请严格按以下 JSON 格式输出（不要输出任何其他内容）：
{{
  "equipment": "设备名称（如：反应釜、换热器、储罐、管道、分馏塔）",
  "parameter": "分析参数（如：温度、压力、流量、液位、浓度）",
  "deviation_type": "偏差类型（简洁描述）",
  "deviation_direction": "偏差方向（从以下选一个：过高、过低、无/停止、反向、其他）",
  "normal_value": "正常操作值（如有提及，否则填"未说明"）",
  "current_value": "当前异常值（如有提及，否则填"未说明"）",
  "other_conditions": "其他工艺条件（如搅拌状态、其他设备状态，没有则填"无"）",
  "query_text": "用于向量检索的英文关键词（包含设备类型、偏差类型、关键参数，8-15个词）"
}}

重要规则：
- 所有字段必须填写，不能为空
- query_text 必须用英文，包含足够的专业术语用于检索
- 如果用户描述模糊，根据化工常识给出合理推断"""


class ContextAgent:
    """
    上下文与模式智能体
    职责：解析用户自然语言 → 标准化工艺上下文 JSON
    """
    
    def run(self, user_input: str) -> dict:
        """
        主执行函数
        
        输入: 用户自然语言描述
        输出: {
            "context": {结构化工艺上下文},
            "query_text": "用于向量检索的文本"
        }
        """
        print(f"\n🔍 [智能体1] 上下文解析中...")
        print(f"   输入: {user_input[:80]}...")
        
        prompt = CONTEXT_PROMPT.format(user_input=user_input)
        
        # 最多重试 3 次
        for attempt in range(3):
            try:
                context = call_llm_json(prompt)
                
                # 验证必填字段
                required_fields = ["equipment", "parameter", "deviation_direction", "query_text"]
                missing = [f for f in required_fields if not context.get(f)]
                if missing:
                    raise ValueError(f"缺少必填字段: {missing}")
                
                print(f"   ✅ 解析成功: {context['equipment']} / {context['parameter']} / {context['deviation_direction']}")
                
                return {
                    "context": context,
                    "query_text": context.get("query_text", user_input)
                }
                
            except Exception as e:
                if attempt == 2:
                    # 最后一次尝试失败，返回基础默认值
                    print(f"   ⚠️ 解析失败，使用默认值: {e}")
                    return {
                        "context": {
                            "equipment": "未识别设备",
                            "parameter": "未识别参数",
                            "deviation_type": "异常偏差",
                            "deviation_direction": "其他",
                            "normal_value": "未说明",
                            "current_value": "未说明",
                            "other_conditions": "无",
                            "query_text": user_input[:100]
                        },
                        "query_text": user_input
                    }
                print(f"   ⚠️ 第{attempt+1}次解析失败，重试中...")
