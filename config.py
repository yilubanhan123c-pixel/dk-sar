"""
DK-SAR 项目配置文件
修改这里来切换 API Key 和模型
"""
import os
from dotenv import load_dotenv

load_dotenv()  # 自动读取 .env 文件

# ── LLM 配置 ──────────────────────────────────────────
# 通义千问（推荐，国内免费）
LLM_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen-plus"
VL_MODEL = "qwen-vl-plus"          # 通义千问视觉模型，用于图片转写



# ── 智能体参数 ─────────────────────────────────────────
MAX_REFLECTION_ROUNDS = 5      # 最多反思修正几轮
TOP_K_CASES = 3                # 检索几个相似案例
SIMILARITY_THRESHOLD = 0.85    # 负样本拦截阈值（0~1，越低防幻觉越严，越高越容易通过）

# ── 数据路径 ───────────────────────────────────────────
DATA_DIR = "data"
POSITIVE_CASES_FILE = "data/positive/cases.json"
NEGATIVE_FALLACIES_FILE = "data/negative/fallacies.json"
SCHEMA_FILE = "data/schema/hazop_schema.json"
CHROMA_DB_DIR = "data/chroma_db"  # 向量数据库存储位置

# ── Embedding 模型 ─────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 轻量英文模型，会自动下载
