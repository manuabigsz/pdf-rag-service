import os
from typing import List
from openai import OpenAI
from loguru import logger
import tiktoken

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"

MAX_TOKENS_PER_REQUEST = 8000  
MAX_TEXTS_PER_BATCH = 2048 

_openai_client = None
_encoding = None


def _get_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _get_encoding():
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base") 
    return _encoding


def _count_tokens(text: str) -> int:
    encoding = _get_encoding()
    return len(encoding.encode(text))


def create_embedding(text: str) -> List[float]:
    client = _get_client()
    
    text = text.replace("\n", " ").strip()
    
    if not text:
        logger.warning("Texto vazio recebido para embedding")
        return []
    
    token_count = _count_tokens(text)
    if token_count > MAX_TOKENS_PER_REQUEST:
        logger.warning(f"Texto muito longo ({token_count} tokens). Truncando para {MAX_TOKENS_PER_REQUEST} tokens.")

        encoding = _get_encoding()
        tokens = encoding.encode(text)[:MAX_TOKENS_PER_REQUEST]
        text = encoding.decode(tokens)
    
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Erro ao criar embedding: {e}")
        raise


def create_embeddings(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    
    client = _get_client()
    
    cleaned_texts = [text.replace("\n", " ").strip() for text in texts]
    non_empty_texts = [t for t in cleaned_texts if t]
    
    if not non_empty_texts:
        logger.warning("Nenhum texto válido para criar embeddings")
        return []
    
    logger.info(f"Processando {len(non_empty_texts)} textos para embeddings")
    
    all_embeddings = []
    
    try:
        current_batch = []
        current_batch_tokens = 0
        
        for i, text in enumerate(non_empty_texts):
            text_tokens = _count_tokens(text)
            
            if text_tokens > MAX_TOKENS_PER_REQUEST:
                logger.warning(f"Texto {i+1} tem {text_tokens} tokens. Truncando...")
                encoding = _get_encoding()
                tokens = encoding.encode(text)[:MAX_TOKENS_PER_REQUEST]
                text = encoding.decode(tokens)
                text_tokens = MAX_TOKENS_PER_REQUEST
            
            if (current_batch_tokens + text_tokens > MAX_TOKENS_PER_REQUEST or 
                len(current_batch) >= MAX_TEXTS_PER_BATCH):
                
                if current_batch:
                    logger.info(f"Processando batch de {len(current_batch)} textos (~{current_batch_tokens} tokens)")
                    batch_embeddings = _create_batch_embeddings(client, current_batch)
                    all_embeddings.extend(batch_embeddings)
                    
                    current_batch = []
                    current_batch_tokens = 0
            
            current_batch.append(text)
            current_batch_tokens += text_tokens
        
        if current_batch:
            logger.info(f"Processando último batch de {len(current_batch)} textos (~{current_batch_tokens} tokens)")
            batch_embeddings = _create_batch_embeddings(client, current_batch)
            all_embeddings.extend(batch_embeddings)
        
        logger.info(f"✓ Total de {len(all_embeddings)} embeddings criados com sucesso")
        return all_embeddings
        
    except Exception as e:
        logger.error(f"Erro ao criar embeddings em batch: {e}")
        raise


def _create_batch_embeddings(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """Cria embeddings para um batch de textos."""
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logger.error(f"Erro ao processar batch: {e}")
        raise