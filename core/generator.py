"""
LLM 调用 —— 大模型回答生成（Ollama + SiliconFlow）

学习要点：
- Prompt Engineering：如何构建高质量的提示词模板
- 流式输出 vs 非流式输出的区别
- 多模型适配：本地 Ollama 和云端 SiliconFlow API 的对接
"""

import json
import logging

logger = logging.getLogger(__name__)

import requests
from functools import lru_cache
from config import (
    OLLAMA_MODEL_NAME, CLOUD_PROVIDERS, CLOUD_MODEL_NAMES
)
from utils.network import get_session
from core.retriever import recursive_retrieval
from core.vector_store import vector_store
from features.conflict_detector import detect_conflicts, evaluate_source_credibility
from features.thinking_chain import process_thinking_content


@lru_cache(maxsize=1)
def check_ollama_model(model_name=None):
    """检查 Ollama 模型是否已拉取，返回 (可用, 已安装模型列表)"""
    if model_name is None:
        model_name = OLLAMA_MODEL_NAME
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return (model_name in models, models)
    except Exception:
        pass
    return (False, [])


def _call_cloud_api(provider, prompt, temperature=0.7, max_tokens=1024):
    """通用云端 API 调用（OpenAI 兼容格式），支持 siliconflow/deepseek/openai"""
    cfg = CLOUD_PROVIDERS.get(provider)
    if not cfg or not cfg.get("api_key"):
        logger.error(f"未设置 {provider} API Key")
        return f"错误：未配置 {provider} API 密钥。"

    model = CLOUD_MODEL_NAMES.get(provider, "gpt-4o")

    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": 0.7,
            "frequency_penalty": 0.5, "n": 1,
        }
        headers = {
            "Authorization": f"Bearer {cfg['api_key'].strip()}",
            "Content-Type": "application/json; charset=utf-8"
        }
        json_payload = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        response = requests.post(cfg["api_url"], data=json_payload, headers=headers, timeout=180)
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0]["message"]
            content = message.get("content", "")
            reasoning = message.get("reasoning_content", "")
            if reasoning:
                return f"{content}<think>{reasoning}</think>"
            return content
        return "API返回结果格式异常"

    except requests.exceptions.RequestException as e:
        logger.error(f"调用 {provider} API 时出错: {str(e)}")
        return f"调用API时出错: {str(e)}"
    except Exception as e:
        logger.error(f"{provider} API 未知错误: {str(e)}")
        return f"发生未知错误: {str(e)}"


# 向后兼容别名
def call_siliconflow_api(prompt, temperature=0.7, max_tokens=1024):
    return _call_cloud_api("siliconflow", prompt, temperature, max_tokens)


def _call_cloud_api_stream(provider, prompt, temperature=0.7, max_tokens=1536):
    """流式调用云端 API（SSE），yield content 增量"""
    cfg = CLOUD_PROVIDERS.get(provider)
    if not cfg or not cfg.get("api_key"):
        yield f"错误：未配置 {provider} API 密钥。"
        return

    model = CLOUD_MODEL_NAMES.get(provider, "gpt-4o")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True, "max_tokens": max_tokens,
        "temperature": temperature, "top_p": 0.7,
        "frequency_penalty": 0.5, "n": 1,
    }
    headers = {
        "Authorization": f"Bearer {cfg['api_key'].strip()}",
        "Content-Type": "application/json; charset=utf-8"
    }
    json_payload = json.dumps(payload, ensure_ascii=False).encode('utf-8')

    try:
        response = requests.post(
            cfg["api_url"], data=json_payload, headers=headers,
            timeout=180, stream=True
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
                # 记录 token 用量（最后一个 chunk 可能包含 usage）
                if "usage" in chunk:
                    try:
                        from core.retriever import _get_metrics
                        u = chunk["usage"]
                        _get_metrics()["token_usage"] = {
                            "prompt": u.get("prompt_tokens", 0),
                            "completion": u.get("completion_tokens", 0),
                            "total": u.get("total_tokens", 0),
                        }
                    except Exception:
                        pass
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    except requests.exceptions.RequestException as e:
        logger.error(f"调用 {provider} API 流式出错: {str(e)}")
        yield f"\n调用API时出错: {str(e)}"
    except Exception as e:
        logger.error(f"{provider} API 流式未知错误: {str(e)}")
        yield f"\n发生未知错误: {str(e)}"


def call_llm_simple(prompt, model_choice="deepseek"):
    """简单的 LLM 调用（用于递归检索中的查询改写判断）"""
    if model_choice in CLOUD_PROVIDERS:
        result = _call_cloud_api(model_choice, prompt)
        result = result.strip() if isinstance(result, str) else result[0].strip()
        if "<think>" in result:
            result = result.split("<think>")[0].strip()
        return result
    else:
        model_ok, available = check_ollama_model(OLLAMA_MODEL_NAME)
        if not model_ok:
            msg = f"模型 '{OLLAMA_MODEL_NAME}' 未安装"
            if available:
                msg += f"，可用: {', '.join(available)}"
            logger.error(msg)
            return "不需要进一步查询"
        try:
            response = get_session().post(
                "http://localhost:11434/api/generate",
                json={"model": OLLAMA_MODEL_NAME, "prompt": prompt, "stream": False},
                timeout=180
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"call_llm_simple Ollama 调用失败: {e}")
            return "不需要进一步查询"


def _build_prompt(question, context, enable_web_search, knowledge_base_exists,
                  time_sensitive, conflict_detected):
    """构建提示词"""
    prompt_template = """作为一个专业的问答助手，你需要基于以下{context_type}回答用户问题。

提供的参考内容：
{context}

用户问题：{question}

请遵循以下回答原则：
1. 仅基于提供的参考内容回答问题，不要使用你自己的知识
2. 如果参考内容中没有足够信息，请坦诚告知你无法回答
3. 回答应该全面、准确、有条理，并使用适当的段落和结构
4. 请用中文回答
5. 在回答末尾标注信息来源{time_instruction}{conflict_instruction}

请现在开始回答："""

    return prompt_template.format(
        context_type="本地文档和网络搜索结果" if enable_web_search and knowledge_base_exists else (
            "网络搜索结果" if enable_web_search else "本地文档"),
        context=context if context else (
            "网络搜索结果将用于回答。" if enable_web_search and not knowledge_base_exists else "知识库为空或未找到相关内容。"),
        question=question,
        time_instruction="，优先使用最新的信息" if time_sensitive and enable_web_search else "",
        conflict_instruction="，并明确指出不同来源的差异" if conflict_detected else ""
    )


def _build_context(all_contexts, all_doc_ids, all_metadata, enable_web_search):
    """构建上下文和来源信息"""
    context_parts = []
    sources_for_conflict = []

    for doc, doc_id, metadata in zip(all_contexts, all_doc_ids, all_metadata):
        source_type = metadata.get('source', '本地文档')
        source_item = {'text': doc, 'type': source_type}

        if source_type == 'web':
            url = metadata.get('url', '未知URL')
            title = metadata.get('title', '未知标题')
            context_parts.append(f"[网络来源: {title}] (URL: {url})\n{doc}")
            source_item['url'] = url
            source_item['title'] = title
        else:
            source = metadata.get('source', '未知来源')
            context_parts.append(f"[本地文档: {source}]\n{doc}")
            source_item['source'] = source

        sources_for_conflict.append(source_item)

    return "\n\n".join(context_parts), sources_for_conflict


def query_answer(question, enable_web_search=False, model_choice="deepseek", progress=None):
    """
    问答处理主流程（非流式）

    完整流程：递归检索 → 构建上下文 → 矛盾检测 → 构建Prompt → LLM生成
    """
    try:
        knowledge_base_exists = vector_store.is_ready
        if not knowledge_base_exists and not enable_web_search:
            return "⚠️ 知识库为空，请先上传文档。"

        if progress:
            progress(0.3, desc="执行递归检索...")

        all_contexts, all_doc_ids, all_metadata = recursive_retrieval(
            initial_query=question, enable_web_search=enable_web_search, model_choice=model_choice
        )

        context, sources = _build_context(all_contexts, all_doc_ids, all_metadata, enable_web_search)
        conflict_detected = detect_conflicts(sources)
        time_sensitive = any(w in question for w in ["最新", "今年", "当前", "最近", "刚刚"])

        prompt = _build_prompt(question, context, enable_web_search,
                               knowledge_base_exists, time_sensitive, conflict_detected)

        if progress:
            progress(0.8, desc="生成回答...")

        if model_choice in CLOUD_PROVIDERS:
            result = _call_cloud_api(model_choice, prompt, temperature=0.7, max_tokens=1536)
        else:
            model_ok, available = check_ollama_model(OLLAMA_MODEL_NAME)
            if not model_ok:
                msg = f"❌ Ollama 模型 '{OLLAMA_MODEL_NAME}' 未安装。"
                if available:
                    msg += f" 当前可用: {', '.join(available)}"
                msg += f"\n请先执行: ollama pull {OLLAMA_MODEL_NAME}"
                return msg
            try:
                response = get_session().post(
                    "http://localhost:11434/api/generate",
                    json={"model": OLLAMA_MODEL_NAME, "prompt": prompt, "stream": False},
                    timeout=180, headers={'Connection': 'close'}
                )
                response.raise_for_status()
                result = str(response.json().get("response", "未获取到有效回答"))
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else "未知"
                logger.error(f"Ollama 调用失败 (HTTP {status}): {e}")
                return f"❌ Ollama 服务返回错误 (HTTP {status})。请检查:\n1. 模型是否已拉取: ollama pull {OLLAMA_MODEL_NAME}\n2. Ollama 服务是否正常运行\n3. 系统内存是否充足"
            except requests.exceptions.ConnectionError:
                return "❌ 无法连接到 Ollama 服务 (localhost:11434)，请确认 Ollama 已启动。"
            except requests.exceptions.Timeout:
                return "❌ Ollama 请求超时，请稍后重试。"

        return process_thinking_content(result)

    except json.JSONDecodeError:
        return "响应解析失败，请重试"
    except Exception as e:
        return f"系统错误: {str(e)}"


def stream_answer(question, enable_web_search=False, model_choice="deepseek"):
    """流式问答：先检索构建 prompt，再流式输出 LLM 生成的每个 token"""
    all_contexts, all_doc_ids, all_metadata = recursive_retrieval(
        initial_query=question, enable_web_search=enable_web_search, model_choice=model_choice
    )

    context, sources = _build_context(all_contexts, all_doc_ids, all_metadata, enable_web_search)
    conflict_detected = detect_conflicts(sources)
    knowledge_base_exists = vector_store.is_ready
    time_sensitive = any(w in question for w in ["最新", "今年", "当前", "最近", "刚刚"])

    prompt = _build_prompt(question, context, enable_web_search,
                           knowledge_base_exists, time_sensitive, conflict_detected)

    if model_choice in CLOUD_PROVIDERS:
        for token in _call_cloud_api_stream(model_choice, prompt, temperature=0.7, max_tokens=1536):
            yield token
    else:
        response = get_session().post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL_NAME, "prompt": prompt, "stream": True},
            timeout=120, stream=True
        )
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode()).get("response", "")
                if chunk:
                    yield chunk
