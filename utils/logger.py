"""
DK-SAR 日志模块
使用 loguru 替代 print，输出带时间戳和级别的结构化日志
日志同时写入文件和控制台
"""
import sys
import os
from loguru import logger

# ── 日志目录 ──────────────────────────────────────────────────
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ── 移除默认 handler，重新配置 ────────────────────────────────
logger.remove()

# 控制台输出：彩色，简洁格式
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[agent]: <12}</cyan> | {message}",
    level="DEBUG",
)

# 文件输出：完整格式，自动按天切割，保留7天
logger.add(
    os.path.join(LOG_DIR, "dk_sar_{time:YYYY-MM-DD}.log"),
    rotation="00:00",       # 每天午夜切割
    retention="7 days",     # 保留7天
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[agent]: <12} | {message}",
    level="DEBUG",
)


def get_logger(agent_name: str = "SYSTEM"):
    """
    获取带 agent 标签的日志器
    
    用法：
        log = get_logger("RAGAgent")
        log.info("开始检索案例")
        log.warning("检索结果为空")
        log.error("API调用失败")
    """
    return logger.bind(agent=agent_name)


# 预置各模块日志器
system_log  = get_logger("SYSTEM")
context_log = get_logger("ContextAgent")
rag_log     = get_logger("RAGAgent")
reflect_log = get_logger("ReflectAgent")
eval_log    = get_logger("Evaluator")
