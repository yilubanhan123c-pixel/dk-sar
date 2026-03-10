"""
LLM 调用封装
支持通义千问和 OpenAI，通过 config.py 切换
"""
import json
from openai import OpenAI
import config


# 初始化客户端（只初始化一次）
_client = None

def get_client():
    global _client
    if _client is None:
        if not config.LLM_API_KEY:
            raise ValueError(
                "\n❌ 没有找到 API Key！\n"
                "请按以下步骤操作：\n"
                "1. 复制 .env.example 文件，重命名为 .env\n"
                "2. 在 .env 文件中填入你的 API Key\n"
                "3. 通义千问免费申请：https://bailian.aliyun.com\n"
            )
        _client = OpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL
        )
    return _client


def call_llm(prompt: str, system_prompt: str = None, json_mode: bool = False) -> str:
    """
    调用大模型
    
    参数:
        prompt: 用户提示词
        system_prompt: 系统提示词（可选）
        json_mode: 是否要求返回 JSON 格式
    
    返回:
        模型返回的文本
    """
    client = get_client()
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    kwargs = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.3,  # 低温度，让输出更稳定
    }
    
    # 通义千问的 JSON 模式
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    
    try:
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            raise ValueError(f"API Key 无效或已过期，请检查 .env 文件中的配置\n原始错误: {e}")
        raise RuntimeError(f"LLM 调用失败: {e}")


def call_llm_json(prompt: str, system_prompt: str = None) -> dict:
    """
    调用大模型并返回解析后的 JSON 字典
    自动处理模型返回的 Markdown 代码块
    """
    response_text = call_llm(prompt, system_prompt, json_mode=True)
    
    # 清理可能的 Markdown 代码块
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])  # 去掉第一行(```json)和最后一行(```)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"模型返回的不是有效 JSON：\n{response_text}\n\n解析错误：{e}")
