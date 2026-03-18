"""
轻量 Query 分类器（规则驱动，零依赖）

根据 query 特征自动判定检索策略：
- keyword: 专有名词/错误码/版本号 → BM25 优先
- semantic: 解释型自然语言问句 → Dense 优先
- default: 两者兼顾
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── 检测模式 ──
_KEYWORD_PATTERNS = [
    # 错误码: ERR_CONNECTION_RESET, 0xDEAD, HTTP404
    r'[A-Z]{2,}_[A-Z]{2,}(?:_[A-Z]+)?',   # UPPER_SNAKE
    r'0x[0-9A-Fa-f]+',                       # 0x hex
    r'HTTP\s*\d{3}',                          # HTTP 404
    # 版本号: CUDA12.4, v3.2.1, RTX4090
    r'[A-Z]+\d+[A-Za-z]*',                    # CUDA12, RTX4090
    r'v?\d+\.\d+(?:\.\d+)?',                  # 12.4, v3.2.1
    # 专有名词: 连续大写缩写
    r'[A-Z]{2,5}',                            # GPU, CPU, RAM
    # 特殊符号: 含 - 的技术名词
    r'\w+-\w+(?:-\w+)?',                      # DeepSeek-V3
]

_SEMANTIC_PATTERNS = [
    r'为什么', r'如何', r'怎么', r'怎样',
    r'什么(?:是|叫|意思|区别)',
    r'区别', r'对比', r'比较', r'哪个好',
    r'原理', r'机制', r'流程', r'步骤',
    r'介绍', r'说明', r'解释', r'概述',
    r'应用', r'用途', r'作用',
    r'优缺点', r'优势', r'不足',
]

# ── 权重配置 ──
WEIGHTS = {
    "keyword":   {"dense": 0.2, "bm25": 0.8},
    "semantic":  {"dense": 0.8, "bm25": 0.2},
    "default":   {"dense": 0.7, "bm25": 0.3},
}


def classify(query):
    """
    返回 (query_type, alpha) 其中 alpha = dense 权重

    检测优先级：keyword > semantic > default
    """
    kw_score = sum(1 for p in _KEYWORD_PATTERNS if re.search(p, query))
    sem_score = sum(1 for p in _SEMANTIC_PATTERNS if p in query)

    if kw_score > 0 and sem_score == 0:
        qtype = "keyword"
    elif sem_score > 0 and kw_score == 0:
        qtype = "semantic"
    elif kw_score > 0 and sem_score > 0:
        # 混合：看哪边更多
        qtype = "keyword" if kw_score > sem_score else "semantic"
    else:
        qtype = "default"

    w = WEIGHTS[qtype]
    logger.info(
        f"Query 分类: {qtype} (kw={kw_score}, sem={sem_score}) "
        f"→ Dense={w['dense']} BM25={w['bm25']}")
    return qtype, w["dense"]
