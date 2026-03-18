"""
配置中心 —— 环境变量加载、模型参数、自动检测机制

学习要点：
- 了解如何通过 .env 文件管理敏感配置（API Key）
- 了解 RAG 系统中的关键超参数及其作用
- 理解 LLM 后端的自动检测与回退机制
"""

import os
import logging

logger = logging.getLogger(__name__)

import requests
from pathlib import Path
from dotenv import load_dotenv

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第一步：加载环境变量
# 优先加载 .env（用户配置），不存在则回退到 example.env（示例配置）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
dotenv_path = Path(__file__).parent / ".env"
if not dotenv_path.exists():
    dotenv_path = Path(__file__).parent / "example.env"
    logger.warning("⚠️ 未找到 .env 文件，已回退加载 example.env。建议：cp example.env .env 并填入真实 API Key")
load_dotenv(dotenv_path)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第二步：API 密钥配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SEARCH_ENGINE = "google"

# SiliconFlow
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_API_URL = os.getenv(
    "SILICONFLOW_API_URL",
    "https://api.siliconflow.cn/v1/chat/completions"
)
# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv(
    "DEEPSEEK_API_URL",
    "https://api.deepseek.com/v1/chat/completions"
)
# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv(
    "OPENAI_API_URL",
    "https://api.openai.com/v1/chat/completions"
)

# 云端 API 提供商配置表
CLOUD_PROVIDERS = {
    "siliconflow": {"api_key": SILICONFLOW_API_KEY, "api_url": SILICONFLOW_API_URL},
    "deepseek":    {"api_key": DEEPSEEK_API_KEY,    "api_url": DEEPSEEK_API_URL},
    "openai":      {"api_key": OPENAI_API_KEY,      "api_url": OPENAI_API_URL},
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第三步：模型名称配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "deepseek-r1:8b")
SILICONFLOW_MODEL_NAME = os.getenv("SILICONFLOW_MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
RERANK_METHOD = os.getenv("RERANK_METHOD", "cross_encoder")

# 模型名称映射
CLOUD_MODEL_NAMES = {
    "siliconflow": SILICONFLOW_MODEL_NAME,
    "deepseek":    DEEPSEEK_MODEL_NAME,
    "openai":      OPENAI_MODEL_NAME,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第四步：RAG 超参数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHUNK_SIZE = 400          # 文本分块大小（字符数）
CHUNK_OVERLAP = 40        # 相邻分块的重叠字符数
HYBRID_ALPHA = 0.7        # 混合检索中语义检索的权重（0-1）
RETRIEVAL_TOP_K = 10      # 检索返回的候选文档数量
RERANK_TOP_K = 5          # 重排序后保留的文档数量
MAX_RETRIEVAL_ITERATIONS = 3  # 递归检索的最大迭代轮数
VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "data", "vector_store")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第五步：运行时环境配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_DATASETS_OFFLINE'] = '1'  # 禁止 datasets 联网，走本地缓存
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
os.environ['FLAGS_use_onednn'] = '0'  # 禁用 ONEDNN，避免 PaddleOCR PIR 属性转换报错
requests.adapters.DEFAULT_RETRIES = 3

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第六步：LLM 后端自动检测
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def detect_default_model():
    """
    自动检测可用的 LLM 后端，返回默认模型选择

    检测优先级：
    1. DeepSeek API Key → deepseek
    2. OpenAI API Key → openai
    3. SiliconFlow API Key → siliconflow
    4. 本地 Ollama → ollama
    """
    for provider, name in [("deepseek", "DeepSeek"), ("openai", "OpenAI"), ("siliconflow", "SiliconFlow")]:
        cfg = CLOUD_PROVIDERS[provider]
        if cfg["api_key"] and cfg["api_key"].strip() and not cfg["api_key"].startswith("Your"):
            logger.info(f"检测到 {name} API Key，默认使用云端模型")
            return provider

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            logger.info("检测到本地 Ollama 服务，默认使用本地模型")
            return "ollama"
    except Exception:
        pass

    logger.warning("未检测到可用 LLM 后端，请配置 API Key 或启动 Ollama")
    return "deepseek"

DEFAULT_MODEL_CHOICE = detect_default_model()
