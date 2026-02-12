import os
from enum import Enum

class LLMProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"

class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:3b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

if EMBEDDING_PROVIDER == EmbeddingProvider.OLLAMA:
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_ollama")
    COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "documents")
else:  
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_openai")
    COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "documents")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

TOP_K = int(os.getenv("TOP_K", "10"))