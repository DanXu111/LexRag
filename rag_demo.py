"""
🧠 本地化智能问答系统（FAISS版）—— 主入口

本文件职责：
- Gradio Web UI 的布局与事件绑定
- 文档处理的编排（调用 core/ 模块完成各步骤）
- 系统监控面板
- 应用启动

核心 RAG 逻辑已拆分到 core/ 和 features/ 模块中，
请按照 core/__init__.py 中的学习路线逐模块阅读。
"""

import os
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import webbrowser
import gradio as gr
from typing import List, Tuple, Optional
from datetime import datetime

# 导入配置
from config import (
    DEFAULT_MODEL_CHOICE, OLLAMA_MODEL_NAME, VECTOR_STORE_DIR
)

# 导入核心模块
from core.document_loader import extract_text, is_layout_aware_result, blocks_to_plain_text


def _overlap(chunk_text, block_text):
    """计算 chunk 与 block 的字符重叠度，用于 layout 映射"""
    if not block_text:
        return 0
    # 简单的子串匹配
    count = 0
    pos = 0
    while True:
        pos = block_text.find(chunk_text[:20], pos)
        if pos == -1:
            break
        count += 1
        pos += 1
    return count
from core.text_splitter import split_text_router
from core.embeddings import encode_texts
from core.vector_store import vector_store
from core.bm25_index import bm25_manager
from core.generator import stream_answer

# 导入工具
from utils.network import is_port_available

print("Gradio version:", gr.__version__)
_t0 = time.time()  # 启动计时

# ── DEBUG: callback 计时 ──
def _timed(name):
    """装饰器：记录 callback 耗时和返回数据大小"""
    def deco(fn):
        def inner(*args, **kwargs):
            t0 = time.time()
            result = fn(*args, **kwargs)
            elapsed = time.time() - t0
            size_mb = 0
            try:
                if isinstance(result, tuple):
                    for i, r in enumerate(result):
                        s = len(str(r))
                        if s > 100000:
                            size_mb += s / 1_000_000
                elif result is not None:
                    size_mb = len(str(result)) / 1_000_000
            except Exception:
                pass
            mark = " [BIG]" if size_mb > 1 else ""
            print(f"[TIMING] {name}: {elapsed:.3f}s, data={size_mb:.2f}MB{mark}")
            return result
        return inner
    return deco


# ── 启动时预加载模型（避免首次请求等待）──
from core.embeddings import preload_embed_model
from core.reranker import preload_reranker
from core.document_loader import preload_ocr
preload_embed_model()
preload_reranker()
preload_ocr()

# ── 启动时自动加载已有索引 ──
if vector_store.load(VECTOR_STORE_DIR):
    bm25_manager.rebuild_from_id_order(vector_store.id_order, vector_store.contents_map)
    print(f"[STARTUP] 已加载本地索引: {vector_store.total_chunks} 个文本块, "
          f"{len(vector_store.doc_registry)} 个文档")
    # 打印数据量统计
    total_chars = sum(len(v) for v in vector_store.contents_map.values())
    print(f"[STARTUP] 总字符数: {total_chars:,}, 总chunks: {vector_store.total_chunks}")
    print(f"[STARTUP] 启动耗时: {time.time() - _t0:.2f}s")
else:
    print("本地索引不存在，初始化为空")

vector_store.set_auto_save(VECTOR_STORE_DIR)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 文档处理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@_timed("process_multiple_files")
def process_multiple_files(files, progress=gr.Progress()):
    """增量处理文件：提取文本 → 分块 → 向量化 → 添加到索引（不删除已有数据）"""
    if not files:
        return "请选择要上传的文件(支持PDF, Word, Excel, PPT, TXT, Markdown等)"

    try:
        total_files = len(files)
        processed_results = []
        has_new = False

        for idx, file in enumerate(files, 1):
            try:
                file_name = os.path.basename(file.name)
                progress((idx - 1) / total_files, desc=f"检查 {idx}/{total_files}: {file_name}")

                # 去重：检查同名文档是否已存在
                existing_id = None
                for did, info in vector_store.doc_registry.items():
                    if info.get("source") == file_name:
                        existing_id = did
                        break
                if existing_id is not None:
                    existing = vector_store.doc_registry[existing_id]
                    processed_results.append(
                        f"⏭️ {file_name}: 已跳过 (已存在, {existing['chunk_ids']}块 v{existing['version']})")
                    continue

                progress(0.3, desc=f"提取文本: {file_name}")
                raw = extract_text(file.name)
                if not raw:
                    raise ValueError("文档内容为空或无法提取文本")

                # 判断是否 layout-aware（block 列表 vs 纯文本字符串）
                if is_layout_aware_result(raw):
                    layout_blocks = raw
                    text = blocks_to_plain_text(layout_blocks)
                else:
                    layout_blocks = None
                    text = raw

                chunks, parent_map = split_text_router(text)
                doc_id = f"doc_{int(time.time())}_{idx}"
                metadatas = [{"source": file_name, "doc_id": doc_id} for _ in chunks]
                chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

                # Parent-Child 模式：标记 parent 关联
                if parent_map is not None:
                    for i in range(len(chunks)):
                        pinfo = parent_map.get(str(i), {})
                        if pinfo:
                            metadatas[i]["parent_id"] = pinfo["parent_id"]
                            metadatas[i]["parent_content"] = pinfo["parent_content"]

                # Layout-aware：映射 block 元数据到 chunk
                if layout_blocks and len(layout_blocks) > 0:
                    block_texts = [(bi, b) for bi, b in enumerate(layout_blocks)]
                    for i, chunk in enumerate(chunks):
                        best = max(block_texts, key=lambda x: _overlap(chunk, x[1]["text"]))
                        if best:
                            bi, block = best
                            metadatas[i]["page_number"] = block.get("page", 1)
                            if block.get("bbox"):
                                metadatas[i]["bbox"] = list(block["bbox"])
                            metadatas[i]["element_type"] = block.get("element_type", "text")
                            metadatas[i]["source_type"] = block.get("source_type", "pdf")

                progress(0.7, desc=f"向量化 {file_name}...")
                embeddings = encode_texts(chunks, show_progress=False)

                progress(0.85, desc=f"写入索引: {file_name}...")
                vector_store.add_documents(chunks, chunk_ids, metadatas, embeddings,
                                           doc_id=doc_id, source=file_name)
                has_new = True

                mode = "Parent-Child" if parent_map else "普通"
                processed_results.append(f"✅ {file_name}: {len(chunks)} 个文本块 ({mode}模式)")

            except Exception as e:
                logger.error(f"处理文件 {file_name} 时出错: {str(e)}")
                processed_results.append(f"❌ {file_name}: 处理失败 - {str(e)}")

        if has_new:
            progress(0.95, desc="同步BM25索引...")
            bm25_manager.rebuild_from_id_order(vector_store.id_order,
                                               vector_store.contents_map)

        processed_results.append(
            f"\n知识库: {vector_store.total_chunks} 块 · {len(vector_store.doc_registry)} 文档")
        return "\n".join(processed_results)

    except Exception as e:
        logger.error(f"处理过程出错: {str(e)}")
        return f"处理过程出错: {str(e)}"


@_timed("clear_all_indexes")
def clear_all_indexes():
    """清空全部索引（FAISS + BM25 + 元数据）"""
    vector_store.clear()
    bm25_manager.clear()
    return "✅ 全部索引已清空"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gradio UI（Gradio 6.x 兼容）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CSS = """
.gradio-container { max-width:100%!important; width:100%!important; }
.left-panel { padding:16px; border-radius:12px; }
.index-panel { padding:16px; border-radius:12px; border:1px solid var(--border-color); }
.file-list { margin-top:10px; }
.footer-note { opacity:0.7; font-size:13px; margin-top:12px; }
.theme-toggle-btn { min-width:40px!important; font-size:20px!important; padding:4px 8px!important; }
.doc-card { padding:12px 16px; margin:8px 0; border-radius:8px;
    background:var(--panel-bg); border:1px solid var(--border-color); }
.doc-card-name { font-weight:600; font-size:15px; }
.doc-card-meta { font-size:12px; opacity:0.7; margin-top:4px; }
.index-stat { font-size:28px; font-weight:700; }
.index-stat-label { font-size:13px; opacity:0.7; }
"""

# 主题切换 JS（Gradio 6 通过 body.classList.toggle('dark') 切换暗色模式）
THEME_JS = """
function() {
    // 读取上次保存的主题偏好，默认白色
    const saved = localStorage.getItem('rag-theme');
    if (saved === 'dark') {
        document.querySelector('body').classList.add('dark');
    }
}
"""

def toggle_theme():
    """返回切换主题的 JS 代码（通过 Gradio 的 js 参数执行）"""
    return gr.update()

# ━━━ Index Panel 渲染 ━━━
def _render_index_panel():
    """生成右侧 Index Panel 的完整 HTML"""
    if not vector_store.is_ready:
        return """
        <div style="padding:20px;text-align:center;opacity:0.7">
            <p style="font-size:48px;margin:0">📭</p>
            <p>索引为空<br>请上传文档构建知识库</p>
        </div>"""

    docs = vector_store.list_documents()
    cards = ""
    for d in sorted(docs, key=lambda x: x['source']):
        cards += f"""
        <div class="doc-card">
            <div class="doc-card-name">📄 {d['source']}</div>
            <div class="doc-card-meta">
                {d['chunks']} 个文本块 · v{d['version']} · {d['added_at']}
            </div>
        </div>"""

    return f"""
    <div style="margin-bottom:16px">
        <div style="display:flex;gap:16px;margin-bottom:16px">
            <div style="flex:1;text-align:center">
                <div class="index-stat">{vector_store.total_chunks}</div>
                <div class="index-stat-label">文本块</div>
            </div>
            <div style="flex:1;text-align:center">
                <div class="index-stat">{len(docs)}</div>
                <div class="index-stat-label">文档</div>
            </div>
        </div>
    </div>
    <div style="font-size:14px;font-weight:600;margin-bottom:8px">📚 已索引文档</div>
    {cards}
    """


def _get_doc_choices():
    """返回下拉框选项: [(label, doc_id), ...]"""
    docs = sorted(vector_store.list_documents(), key=lambda x: x['source'])
    return [(f"{d['source']} ({d['chunks']}块 v{d['version']})", d['doc_id']) for d in docs]


with gr.Blocks(title="本地RAG问答系统") as demo:
    with gr.Row():
        with gr.Column(scale=9):
            gr.Markdown("# 🧠 智能文档问答系统")
        with gr.Column(scale=1, min_width=60):
            theme_btn = gr.Button("🌓", min_width=40, elem_classes="theme-toggle-btn")

    with gr.Row(equal_height=False):
        # ━━━ 左侧：纯问答区 ━━━
        with gr.Column(scale=6, elem_classes="left-panel"):
            chatbot = gr.Chatbot(
                label="", height=560, show_label=False,
                layout="bubble", placeholder="上传文档后开始提问",
            )

            with gr.Row():
                question_input = gr.Textbox(
                    label="", lines=2, placeholder="输入消息...",
                    scale=8, container=False,
                )
                with gr.Column(scale=1, min_width=50):
                    ask_btn = gr.Button("➤", variant="primary", size="lg")
                    stop_btn = gr.Button("■", variant="stop", size="lg", visible=False)

            with gr.Row():
                model_choice = gr.Dropdown(
                    choices=["deepseek", "openai", "siliconflow", "ollama"],
                    value=DEFAULT_MODEL_CHOICE, label="模型", scale=2,
                )
                web_search_checkbox = gr.Checkbox(
                    label="联网搜索", value=False, scale=1,
                )
                clear_btn = gr.Button("清空对话", variant="secondary", size="sm", scale=1)

        # ━━━ 右侧：文档管理 + Index Panel ━━━
        with gr.Column(scale=4, elem_classes="index-panel"):
            with gr.Accordion("📂 文档管理", open=True):
                file_input = gr.File(
                    label="上传文档",
                    file_types=[".pdf", ".txt", ".docx", ".xlsx", ".xls", ".pptx", ".md",
                                ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"],
                    file_count="multiple"
                )
                with gr.Row():
                    upload_btn = gr.Button("🚀 上传并处理", variant="primary", scale=2)
                    clear_all_btn = gr.Button("🗑️ 清空全部索引", variant="stop", scale=1)
                upload_status = gr.Textbox(label="处理状态", interactive=False, lines=2)

            gr.Markdown("---")
            gr.Markdown("### 📊 索引状态")
            index_panel_html = gr.HTML(_render_index_panel)

            gr.Markdown("### 🗑️ 删除文档")
            doc_selector = gr.Dropdown(
                label="选择要删除的文档",
                choices=_get_doc_choices(),
                interactive=True,
            )
            delete_doc_btn = gr.Button("❌ 删除选中文档", variant="stop")
            delete_status = gr.Textbox(label="", interactive=False, visible=False)

            # ━━━ Benchmark 折叠区 ━━━
            with gr.Accordion("📊 RAG Benchmark", open=False):
                benchmark_dataset = gr.Dropdown(
                    label="数据集",
                    choices=["eval/dataset_example.json", "covidqa", "cuad", "delucionqa",
                             "emanual", "expertqa", "finqa", "hagrid", "hotpotqa",
                             "msmarco", "pubmedqa", "tatqa", "techqa"],
                    value="covidqa",
                )
                with gr.Row():
                    benchmark_vector_db = gr.Dropdown(
                        choices=["faiss", "qdrant", "milvus"],
                        value="faiss", label="Vector DB")
                    benchmark_embed_model = gr.Dropdown(
                        choices=["all-MiniLM-L6-v2", "bge-small-zh", "bge-base-zh", "e5-small"],
                        value="all-MiniLM-L6-v2", label="Embedding")
                with gr.Row():
                    benchmark_chunk_mode = gr.Dropdown(
                        choices=["fixed", "parent-child", "sliding"],
                        value="fixed", label="Chunk Mode")
                    benchmark_chunk_size = gr.Slider(
                        200, 1200, value=400, step=100, label="Chunk Size")
                with gr.Row():
                    benchmark_hybrid_weight = gr.Slider(
                        0.0, 1.0, value=0.7, step=0.1,
                        label="Dense 权重 (0=纯BM25, 1=纯Dense)")
                    benchmark_rerank = gr.Dropdown(
                        choices=["cross-encoder", "none"],
                        value="cross-encoder", label="Rerank")
                with gr.Row():
                    benchmark_top_k = gr.Slider(
                        3, 20, value=5, step=1, label="Top-K", scale=2)
                    benchmark_btn = gr.Button(
                        " 开始 Benchmark", variant="primary", scale=1)

                benchmark_status = gr.Textbox(
                    label="状态", interactive=False, lines=6,
                    placeholder="使用当前索引对数据集执行检索评估...")
                benchmark_result_path = gr.Textbox(
                    label="结果文件", interactive=False, visible=False)


    # ━━━ 流式聊天停止标志 ━━━
    _stop_stream = False

    # ━━━ 事件处理函数 ━━━
    def _refresh_panel_and_dropdown():
        return _render_index_panel(), gr.Dropdown(choices=_get_doc_choices(), value=None)

    @_timed("clear_chat_history")
    def clear_chat_history():
        return []

    def stream_chat(question, history, enable_web_search, model_choice_val):
        """流式聊天：用户消息立显 → 逐 token 输出 AI 回复"""
        global _stop_stream
        _stop_stream = False

        if history is None:
            history = []
        if not question or not question.strip():
            yield history, question, gr.update(visible=False), gr.update(visible=True)
            return

        from core.generator import stream_answer
        from core.retriever import start_request_trace, collect_request_metrics
        from core.metrics import insert_log

        # 初始化请求追踪
        import uuid
        session_id = str(uuid.uuid4())[:8]
        t0 = time.time()
        start_request_trace(session_id, question, model_choice_val, enable_web_search)

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": ""})
        yield history, "", gr.update(visible=True), gr.update(visible=False)

        full = ""
        error = None
        try:
            for token in stream_answer(question, enable_web_search, model_choice_val):
                if _stop_stream:
                    break
                full += token
                history[-1]["content"] = full
                yield history, "", gr.update(visible=True), gr.update(visible=False)
        except Exception as e:
            logger.error(f"流式聊天异常: {e}")
            error = str(e)
            if not full:
                history[-1]["content"] = f"系统错误: {e}"
                yield history, "", gr.update(visible=True), gr.update(visible=False)

        # 写入统一日志
        try:
            m = collect_request_metrics()
            total_ms = round((time.time() - t0) * 1000, 1)
            insert_log(
                session_id=session_id,
                original_query=question,
                rewritten_query=("; ".join(
                    f"{r['from']}->{r['to']}" for r in m["query_rewrites"])
                    if m["query_rewrites"] else None),
                retrieval_latency_ms=round(m["retrieval_latency_ms"], 1),
                rerank_latency_ms=round(m["rerank_latency_ms"], 1),
                llm_latency_ms=round(total_ms - m["retrieval_latency_ms"] - m["rerank_latency_ms"], 1),
                total_latency_ms=total_ms,
                retrieved_chunks=m["retrieval_chunks"],
                prompt_tokens=m["token_usage"].get("prompt", 0),
                completion_tokens=m["token_usage"].get("completion", 0),
                total_tokens=m["token_usage"].get("total", 0),
                cache_hit=m["cache_hit"],
                query_type=m.get("query_type", "default"),
                hybrid_alpha=m.get("hybrid_alpha", 0.7),
                answer=full,
                model=model_choice_val,
                web_search=enable_web_search,
                error=error,
            )
        except Exception:
            pass

        yield history, "", gr.update(visible=False), gr.update(visible=True)

    def on_stop():
        global _stop_stream
        _stop_stream = True
        return gr.update(visible=False), gr.update(visible=True)

    @_timed("delete_document")
    def on_delete_document(doc_id):
        if not doc_id:
            return _render_index_panel(), gr.Dropdown(choices=_get_doc_choices(), value=None)
        if vector_store.is_document_exists(doc_id):
            vector_store.delete_document(doc_id)
            bm25_manager.rebuild_from_id_order(vector_store.id_order, vector_store.contents_map)
        return _render_index_panel(), gr.Dropdown(choices=_get_doc_choices(), value=None)

    # ━━━ 绑定事件 ━━━
    demo.load(fn=_refresh_panel_and_dropdown, outputs=[index_panel_html, doc_selector])
    upload_btn.click(process_multiple_files, inputs=[file_input], outputs=[upload_status], show_progress=True) \
             .then(fn=_refresh_panel_and_dropdown, outputs=[index_panel_html, doc_selector])
    clear_all_btn.click(clear_all_indexes, outputs=[upload_status]) \
                .then(fn=_refresh_panel_and_dropdown, outputs=[index_panel_html, doc_selector])
    delete_doc_btn.click(on_delete_document, inputs=[doc_selector],
                         outputs=[index_panel_html, doc_selector])

    ask_btn.click(
        fn=stream_chat,
        inputs=[question_input, chatbot, web_search_checkbox, model_choice],
        outputs=[chatbot, question_input, stop_btn, ask_btn],
    )
    question_input.submit(
        fn=stream_chat,
        inputs=[question_input, chatbot, web_search_checkbox, model_choice],
        outputs=[chatbot, question_input, stop_btn, ask_btn],
    )
    stop_btn.click(fn=on_stop, outputs=[stop_btn, ask_btn])
    clear_btn.click(fn=clear_chat_history, outputs=[chatbot])

    # ━━ Benchmark ━━
    def run_gui_benchmark(dataset_spec, vector_db, embed_model, chunk_mode,
                          chunk_size, hybrid_weight, rerank, top_k):
        from eval.gui_benchmark import run_benchmark

        yield "正在初始化...", "", gr.update(visible=True)

        log_lines = []
        def log_cb(msg):
            log_lines.append(msg)

        try:
            results, summary, path = run_benchmark(
                dataset_spec=dataset_spec,
                embedding_model=embed_model,
                chunk_size=int(chunk_size),
                chunk_mode=chunk_mode,
                vector_db=vector_db,
                top_k=int(top_k),
                hybrid_dense_weight=float(hybrid_weight),
                rerank_enabled=(rerank != "none"),
                progress_callback=log_cb,
            )
            log_lines.append("")
            log_lines.append("--- 完成 ---")
            log_lines.append(f"配置: {vector_db} | {embed_model} | {chunk_mode} | topK={top_k}")
            log_lines.append(f"问题数: {len(results)}")
            log_lines.append(f"平均检索耗时: {summary['avg_retrieval_ms']}ms")
            log_lines.append(f"平均召回 chunk: {summary['avg_chunks_retrieved']}")
            log_lines.append(f"索引状态: {'已加载' if summary['has_index'] else '空'}")
            log_lines.append(f"结果: {path}")

            yield "\n".join(log_lines), path, gr.update(visible=True)
        except Exception as e:
            log_lines.append(f"ERROR: {e}")
            yield "\n".join(log_lines), "", gr.update(visible=True)

    benchmark_btn.click(
        fn=run_gui_benchmark,
        inputs=[benchmark_dataset, benchmark_vector_db, benchmark_embed_model,
                benchmark_chunk_mode, benchmark_chunk_size,
                benchmark_hybrid_weight, benchmark_rerank, benchmark_top_k],
        outputs=[benchmark_status, benchmark_result_path, benchmark_btn],
    )

    theme_btn.click(fn=toggle_theme, inputs=[], outputs=[], js="""
        () => {
            document.querySelector('body').classList.toggle('dark');
            const isDark = document.querySelector('body').classList.contains('dark');
            localStorage.setItem('rag-theme', isDark ? 'dark' : 'light');
        }
    """)


def check_environment():
    """环境依赖检查"""
    from config import CLOUD_PROVIDERS
    from core.generator import _call_cloud_api

    # 优先检测云端 API
    for provider in ("deepseek", "openai", "siliconflow"):
        cfg = CLOUD_PROVIDERS[provider]
        if cfg["api_key"] and not cfg["api_key"].startswith("Your"):
            print(f"检测到 {provider} API Key，测试连接...")
            try:
                result = _call_cloud_api(provider, "你好，请回复'连接成功'", temperature=0.1, max_tokens=50)
                if isinstance(result, str) and ("连接成功" in result or "你好" in result):
                    print(f"{provider} 连接测试成功")
                else:
                    print(f"{provider} 响应异常，但继续运行")
                return True
            except Exception as e:
                print(f"{provider} 测试失败: {e}")
                return True

    # 回退到本地 Ollama
    print("未检测到云端 API Key，尝试本地 Ollama...")
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            print("本地 Ollama 服务可用")
            models = resp.json().get("models", [])
            model_names = [m["name"] for m in models]
            if model_names:
                print(f"   已安装模型: {', '.join(model_names)}")
            if OLLAMA_MODEL_NAME not in model_names:
                print(f"当前配置的模型 '{OLLAMA_MODEL_NAME}' 未安装!")
                print(f"   已安装: {', '.join(model_names) if model_names else '(无)'}")
                print(f"   请执行: ollama pull {OLLAMA_MODEL_NAME}")
                if model_names:
                    print(f"   或修改 .env 中 OLLAMA_MODEL_NAME 为已安装的模型")
            else:
                print(f"模型 '{OLLAMA_MODEL_NAME}' 已就绪")
            return True
        else:
            print(f"Ollama 返回异常状态码: {resp.status_code}")
            return True
    except Exception:
        pass

    print("未找到任何可用的 LLM 后端")
    print("   请在 .env 中配置 DEEPSEEK_API_KEY / OPENAI_API_KEY / SILICONFLOW_API_KEY 或启动 Ollama")
    return False


if __name__ == "__main__":
    if not check_environment():
        exit(1)

    ports = [17995, 17996, 17997, 17998, 17999]
    selected_port = next((p for p in ports if is_port_available(p)), None)

    if not selected_port:
        print("所有端口都被占用，请手动释放端口")
        exit(1)

    try:
        webbrowser.open(f"http://127.0.0.1:{selected_port}")
        demo.launch(
            server_port=selected_port, server_name="0.0.0.0",
            show_error=True, ssl_verify=False, height=900,
            css=CSS, js=THEME_JS
        )
    except Exception as e:
        print(f"启动失败: {str(e)}")
