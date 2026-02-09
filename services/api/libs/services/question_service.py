from typing import List, Tuple
from libs.providers.embeddings_provider import create_embeddings
from libs.services.vector_service import search_similar
from libs.providers.llm_provider import generate_answer


def process_question(question: str) -> Tuple[str, List[str]]:
    question_embeddings = create_embeddings([question])
    question_embedding = question_embeddings[0]
    
    top_k = 5  
    relevant_chunks = search_similar(question_embedding, top_k=top_k)
    
    answer = generate_answer(question, relevant_chunks)
    
    return answer, relevant_chunks