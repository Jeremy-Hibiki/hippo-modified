from llama_index.core import BaseCallbackHandler, ChatPromptTemplate, PromptHelper, StorageContext, VectorStoreIndex
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.callbacks import CallbackManager
from llama_index_exts_fmsh.postprocessor.text_embeddings_inference_rerank import TextEmbeddingsInferenceRerank
from environs import env
from llama_index_exts_fmsh.response_synthesizers.rate_limited import RateLimitedTreeSummarize
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index_exts_fmsh.indices.raptor_tree import RaptorNodeArranger
import os
from llama_index.llms.openai_like import OpenAILike
from .llm import _get_llm_class, BaseLLM


def get_reranker(callback_manager: CallbackManager | None = None):
    provider = env("RERANK_PROVIDER", "tei")
    if provider == "tei":
        from llama_index_exts_fmsh.postprocessor.text_embeddings_inference_rerank.base import DEFAULT_URL
        from llama_index_exts_fmsh.postprocessor.text_embeddings_inference_rerank.base import (
            TextEmbeddingsInferenceRerank as TEIRerank,
        )

        rerank = TEIRerank(
            base_url=env("RERANK_API_BASE", DEFAULT_URL),
            batch_size=env.int("RERANK_BATCH_SIZE", 32),
            top_n=env.int("RERANK_TOP_K", 10),
        )
    elif provider == "jina":
        pass
        # from llama_index.postprocessor.jinaai_rerank.base import DEFAULT_JINA_AI_API_URL, JinaRerank

        # rerank = JinaRerank(
        #     model=env("RERANK_MODEL_NAME", "jina-reranker-v2-base-multilingual"),
        #     api_key=env("RERANK_API_KEY", "sk-fake-key"),
        #     base_url=env("RERANK_API_BASE", DEFAULT_JINA_AI_API_URL),
        #     top_n=env.int("RERANK_TOP_K", 10),
        # )
    else:
        raise ValueError(f"Invalid rerank provider: {provider}")

    if callback_manager is not None:
        rerank.callback_manager = callback_manager
    return rerank




def dense_passage_retrieval(query:str, callback_manager: CallbackManager, global_config,**kwargs):

    enable_rerank = True
    reranker = get_reranker(None)
    query_post_processors: list[BaseNodePostprocessor] = []

    if enable_rerank:
        query_post_processors.append(reranker)
        query_post_processors.append(SimilarityPostprocessor(similarity_cutoff=0.1))

    query_post_processors.append(
    RaptorNodeArranger(
        storage_context=kwargs.get("storage_context"),
        callback_manager=callback_manager,
    )
)
    llm_model: BaseLLM = _get_llm_class(global_config)


    response_synthesizer = RateLimitedTreeSummarize(
    callback_manager=callback_manager,
    llm=llm_model,
    num_workers=4,
    use_async=True,
    summary_template=ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名专业的 FAE (现场应用工程师)，负责解答客户提出的问题，包括但不限于关于 icraft 编译器、机器学习和神经网络基础知识、以及软件开发相关的问题。"
                "回答需要有一定的专业性，内容要详细、丰富，不要偷懒，**绝对不允许拒绝回答或者说出“我不知道”**。"
                "如果某个工具报错、或不能解答问题，请尝试修改你的调用参数、或者使用其他工具。",
            ),
            (
                "user",
                """从多个来源得到的上下文信息如下：
---------------------
{context_str}
---------------------
优先根据以上的多来源信息，适当结合先验知识，回答问题。
问题: {query_str}
---------------------
除非客户指定了回答的详细程度和格式，否则逐条目叙述、详实完整。
答案: """,
            ),
        ]
    ),
    prompt_helper=PromptHelper.from_llm_metadata(
        llm_model.metadata,
        tokenizer=llm_model.tokenizer.encode,  # type: ignore
    ),
    verbose=True,
)


