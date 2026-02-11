from uuid import UUID
from typing import List
import os
import shutil
from loguru import logger

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

        try:
            existing_collections = _chroma_client.list_collections()
            collection_exists = any(col.name == COLLECTION_NAME for col in existing_collections)
            
            if collection_exists:
                temp_collection = _chroma_client.get_collection(name=COLLECTION_NAME)
                
                try:
                    count = temp_collection.count()
                    logger.info(f"Collection '{COLLECTION_NAME}' já existe com {count} itens")
                    
                    if count > 0:
                        sample = temp_collection.peek(1)
                        if sample and 'embeddings' in sample and sample['embeddings']:
                            current_dim = len(sample['embeddings'][0])
                            logger.info(f"Dimensão atual da collection: {current_dim}")
                            
                            if current_dim != 1536:
                                logger.warning(f"Dimensão incompatível! Esperado 1536, atual {current_dim}")
                                logger.warning("Deletando collection antiga...")
                                _chroma_client.delete_collection(name=COLLECTION_NAME)
                                collection_exists = False
                    
                except Exception as e:
                    logger.warning(f"Erro ao verificar collection: {e}")
                    logger.warning("Deletando collection antiga...")
                    _chroma_client.delete_collection(name=COLLECTION_NAME)
                    collection_exists = False
            
            if collection_exists:
                _collection = _chroma_client.get_collection(name=COLLECTION_NAME)
                logger.info(f"Usando collection existente: {COLLECTION_NAME}")
            else:
                _collection = _chroma_client.create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"} 
                )
                logger.info(f"Collection '{COLLECTION_NAME}' criada com sucesso")
                
        except Exception as e:
            logger.error(f"Erro ao gerenciar collection: {e}")
            raise

    return _collection


def store_chunks(chunks: List[str], document_id: UUID) -> int:
    from libs.providers.embeddings_provider import create_embeddings

    if not chunks:
        logger.warning(f"Nenhum chunk para armazenar do documento {document_id}")
        return 0

    collection = _get_collection()
    
    logger.info(f"Criando embeddings para {len(chunks)} chunks do documento {document_id}")
    embeddings = create_embeddings(chunks)
    
    if len(embeddings) != len(chunks):
        logger.error(f"Número de embeddings ({len(embeddings)}) diferente de chunks ({len(chunks)})")
        raise ValueError("Mismatch entre número de chunks e embeddings")

    ids = []
    metadatas = []

    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{document_id}_{idx}"
        ids.append(chunk_id)

        metadatas.append({
            "document_id": str(document_id),
            "chunk_index": idx,
            "chunk_length": len(chunk),
        })

    logger.info(f"Adicionando {len(chunks)} chunks à collection")
    
    try:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        logger.info(f"✓ {len(chunks)} chunks armazenados com sucesso para documento {document_id}")
    except Exception as e:
        logger.error(f"Erro ao adicionar chunks ao ChromaDB: {e}")
        raise

    return len(chunks)


def search_similar(query_embedding: List[float], top_k: int = 5) -> List[str]:
    collection = _get_collection()
    
    total_items = collection.count()
    logger.info(f"Buscando em {total_items} chunks armazenados")
    
    if total_items == 0:
        logger.warning("Nenhum documento indexado ainda!")
        return []
    
    actual_top_k = min(top_k, total_items)
    
    logger.info(f"Buscando top {actual_top_k} chunks mais relevantes")

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        logger.info(f"Encontrados {len(documents)} chunks")
        
        for i, (doc, dist, meta) in enumerate(zip(documents, distances, metadatas)):
            logger.debug(f"Chunk {i+1} (distância={dist:.4f}, doc_id={meta.get('document_id')}): {doc[:100]}...")
        
        return documents
        
    except Exception as e:
        logger.error(f"Erro na busca: {e}")
        raise


def get_collection_stats() -> dict:
    collection = _get_collection()
    count = collection.count()
    
    return {
        "total_chunks": count,
        "collection_name": COLLECTION_NAME,
    }