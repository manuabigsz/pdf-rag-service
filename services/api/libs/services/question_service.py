from typing import List, Tuple
from loguru import logger

from libs.providers.embeddings_provider import create_embedding
from libs.services.vector_service import search_similar
from libs.providers.llm_provider import generate_answer


def process_question(question: str) -> Tuple[str, List[str]]:
    logger.info(f"Processando pergunta: {question}")
    
    question_embedding = create_embedding(question)
    
    if not question_embedding:
        logger.error("Falha ao criar embedding da pergunta")
        return "Error processing question", []
    
    top_k = 10 
    logger.info(f"Buscando top {top_k} chunks mais relevantes")
    
    relevant_chunks = search_similar(question_embedding, top_k=top_k)
    
    if not relevant_chunks:
        logger.warning("Nenhum chunk relevante encontrado")
        return "No relevant information found in the documents.", []
    
    logger.info(f"Encontrados {len(relevant_chunks)} chunks relevantes")
    
    answer = generate_answer(question, relevant_chunks)
    
    return answer, relevant_chunks