"""
基于 ragas_dataset.json 运行 RAGAS 评估
LLM 用 DeepSeek，Embedding 用本地 bge-small-zh-v1.5

使用:
  uv run python -m eval.run_ragas_eval
"""

import os
import sys
import json
import logging
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "ragas_dataset.json")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", os.getenv("api_key", ""))
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")


def run():
    # Monkey-patch: ragas + langchain-community 0.4.x 兼容
    import langchain_community.chat_models
    _dummy = types.ModuleType("langchain_community.chat_models.vertexai")
    class _FakeChatVertexAI:
        pass
    _dummy.ChatVertexAI = _FakeChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _dummy
    langchain_community.chat_models.vertexai = _dummy

    from openai import OpenAI
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        context_precision,
        context_recall,
        answer_relevancy,
    )
    from ragas.llms import llm_factory
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

    # 加载数据集
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    raw = raw[:3]  # 临时：只测前 3 条，改大或删掉这行测全部
    logger.info(f"加载 {len(raw)} 条评估样本")

    # LLM: DeepSeek (OpenAI 兼容)
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_URL)
    eval_llm = llm_factory(DEEPSEEK_MODEL, client=client)
    logger.info(f"LLM 后端: DeepSeek ({DEEPSEEK_MODEL})")

    # Embedding: 本地 bge-small-zh-v1.5
    eval_embeddings = HuggingFaceEmbeddings(model=r"E:\model\bge-small-zh-v1.5")
    eval_embeddings.embed_query = lambda text: eval_embeddings.embed_text(text)
    eval_embeddings.embed_documents = lambda texts: eval_embeddings.embed_texts(texts)
    logger.info("Embedding 后端: 本地 bge-small-zh-v1.5")

    # 构建 EvaluationDataset
    samples = []
    for item in raw:
        samples.append(SingleTurnSample(
            user_input=item["user_input"],
            retrieved_contexts=item["retrieved_contexts"],
            response=item["response"],
            reference=item["reference"],
        ))
    dataset = EvaluationDataset(samples=samples)

    # 旧版指标 (兼容 evaluate)
    metrics = [faithfulness, context_precision, context_recall, answer_relevancy]
    logger.info(f"开始评估 ...")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

    logger.info("=" * 60)
    logger.info("RAGAS 评估结果:")
    df = result.to_pandas()
    for _, row in df.iterrows():
        logger.info(f"  {row.to_dict()}")
    avg = df.mean(numeric_only=True)
    logger.info(f"  平均分: {avg.to_dict()}")
    logger.info("=" * 60)

    # 导出
    out = avg.to_dict()
    out["num_samples"] = len(raw)
    out_path = os.path.join(os.path.dirname(__file__), "ragas_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存: {out_path}")

    return result


if __name__ == "__main__":
    run()
