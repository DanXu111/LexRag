"""
文本分块器 —— 将长文本切分为检索友好的片段

两种模式：
- 普通模式（≤5万字）：直接按 CHUNK_SIZE 切分
- Parent-Child 模式（>5万字）：大块(parent)保留上下文，小块(child)做 embedding 检索
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
import logging

logger = logging.getLogger(__name__)

LONG_TEXT_THRESHOLD = 50_000   # 超过此字符数启用 parent-child
PARENT_CHUNK_SIZE = 2000       # parent 块大小
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400         # child 块大小（用于 embedding）
CHILD_CHUNK_OVERLAP = 40


def split_text(text, chunk_size=None, chunk_overlap=None):
    """
    普通切分（向后兼容）

    Returns:
        切分后的文本片段列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or CHUNK_SIZE,
        chunk_overlap=chunk_overlap or CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", "；", "：", " ", ""]
    )
    return text_splitter.split_text(text)


def split_text_router(text):
    """
    根据文本长度自动选择切分策略。

    ≤5万字: 普通切分 → (chunks, None)
    >5万字: parent-child 切分 → (children, parent_map)
            parent_map = {child_index_in_all_children: parent_content}
    """
    if len(text) <= LONG_TEXT_THRESHOLD:
        return split_text(text), None

    logger.info(f"长文本 ({len(text)} 字)，启用 Parent-Child 切分")

    # 创建更大的 parent 切分器
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", "；", "：", " ", ""]
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", "；", "：", " ", ""]
    )

    parents = parent_splitter.split_text(text)
    children = []
    parent_map = {}   # child_global_index → parent_content

    for pi, parent_text in enumerate(parents):
        p_id = f"P{pi}"
        sub_children = child_splitter.split_text(parent_text)
        for ci, child_text in enumerate(sub_children):
            global_idx = len(children)
            children.append(child_text)
            parent_map[str(global_idx)] = {  # 用 str idx 作 key（metadata 只支持 str）
                "parent_id": p_id,
                "parent_content": parent_text,
            }

    logger.info(f"Parent-Child 切分完成: {len(parents)} parents → {len(children)} children")
    return children, parent_map
