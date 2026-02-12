from typing import List
from loguru import logger

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from libs.utils.envs import (
    EMBEDDING_PROVIDER,
    EmbeddingProvider,
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


_embeddings: Embeddings | None = None
_vector_store: Chroma | None = None
_text_splitter: RecursiveCharacterTextSplitter | None = None


def _get_embeddings() -> Embeddings:
    global _embeddings
    if _embeddings is None:
        if EMBEDDING_PROVIDER == EmbeddingProvider.OLLAMA:
            logger.info(f"Usando Ollama embeddings: {OLLAMA_EMBEDDING_MODEL}")
            _embeddings = OllamaEmbeddings(
                model=OLLAMA_EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL,
            )
        else:  
            logger.info(f"Usando OpenAI embeddings: {OPENAI_EMBEDDING_MODEL}")
            _embeddings = OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY,
            )
    return _embeddings


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    global _text_splitter
    if _text_splitter is None:
        _text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    return _text_splitter


def _get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=_get_embeddings(),
            persist_directory=CHROMA_PERSIST_DIR,
        )
        logger.info(f"Vector store LangChain Chroma inicializado: {COLLECTION_NAME}")
    return _vector_store


def add_documents(documents: List[Document]) -> int:
    if not documents:
        logger.warning("Nenhum documento para adicionar")
        return 0
    store = _get_vector_store()
    ids = store.add_documents(documents)
    n = len(ids) if ids else len(documents)
    logger.info(f"✓ {n} chunks armazenados no Chroma (LangChain)")
    return n


def get_retriever(top_k: int = 10):
    store = _get_vector_store()
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


def search_similar(query: str, top_k: int = 10) -> List[Document]:
    retriever = get_retriever(top_k=top_k)
    return retriever.invoke(query)


def get_collection_stats() -> dict:
    store = _get_vector_store()
    try:
        count = store._collection.count()
    except Exception as e:
        logger.warning(f"Erro ao obter count da collection: {e}")
        count = 0
    return {
        "total_chunks": count,
        "collection_name": COLLECTION_NAME,
    }