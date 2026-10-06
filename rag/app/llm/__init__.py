"""llm 子包：向量化与生成后端（可插拔）。"""

from app.llm.embeddings import (
    BaseEmbeddings,
    HashEmbeddings,
    LocalSTEmbeddings,
    OpenAICompatEmbeddings,
    embedding_info,
    get_embeddings,
)
from app.llm.generator import (
    ExtractiveAnswerer,
    LLMError,
    OpenAICompatChat,
    build_context,
    build_messages,
    get_chat_client,
    llm_info,
)

__all__ = [
    "BaseEmbeddings",
    "HashEmbeddings",
    "LocalSTEmbeddings",
    "OpenAICompatEmbeddings",
    "embedding_info",
    "get_embeddings",
    "ExtractiveAnswerer",
    "LLMError",
    "OpenAICompatChat",
    "build_context",
    "build_messages",
    "get_chat_client",
    "llm_info",
]
