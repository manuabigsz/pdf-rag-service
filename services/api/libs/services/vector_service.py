from uuid import UUID
from typing import List
import os

import chromadb
from chromadb.config import Settings


CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR", "./data/chroma_db"
)

COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION", "documents"
)

_chroma_client = None
_collection = None


def _get_collection():
    global _chroma_client, _collection

    if _collection is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(
                anonymized_telemetry=False,
            )
        )

        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    return _collection


def store_chunks(chunks: List[str], document_id: UUID) -> int:
    from libs.providers.embeddings_provider import create_embeddings

    collection = _get_collection()
    embeddings = create_embeddings(chunks)

    ids = []
    metadatas = []

    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{document_id}_{idx}"
        ids.append(chunk_id)

        metadatas.append({
            "document_id": str(document_id),
            "chunk_index": idx
        })

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    return len(chunks)


def search_similar(query_embedding, top_k: int = 5) -> List[str]:
    collection = _get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results.get("documents", [[]])[0]