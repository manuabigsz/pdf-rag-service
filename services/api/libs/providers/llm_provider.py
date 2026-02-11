import os
from typing import List
from openai import OpenAI
from loguru import logger

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def generate_answer(question: str, context_chunks: List[str]) -> str:
    logger.info("=== INÍCIO: generate_answer ===")
    logger.info(f"Pergunta: {question}")
    logger.info(f"Chunks recebidos: {len(context_chunks) if context_chunks else 0}")

    if not context_chunks or len(context_chunks) == 0:
        logger.warning("Nenhum chunk recuperado")
        return "I don't have enough information to answer this question based on the provided documents."

    for i, chunk in enumerate(context_chunks[:3]): 
        logger.debug(f"\n--- CHUNK {i+1} (tamanho={len(chunk)}) ---\n{chunk[:300]}...")

    context = "\n\n".join(
        [f"[Document {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )

    prompt = f"""You are a helpful assistant that answers questions based on the provided documents.
        DOCUMENTS:
        {context}

        QUESTION: {question}

        INSTRUCTIONS:
        1. Answer the question based PRIMARILY on the information in the documents above
        2. If the documents contain relevant information, use it to answer
        3. If the documents do NOT contain enough information to fully answer the question, say so clearly
        4. Be direct and concise in your answer
        5. When possible, reference which document(s) your answer comes from

        ANSWER:"""

    logger.info("=== CHAMANDO LLM ===")
    
    answer = _generate_with_openai(prompt)

    logger.info("=== RESPOSTA DO LLM ===")
    logger.info(answer)
    logger.info("=== FIM: generate_answer ===")

    return answer


def _generate_with_openai(prompt: str) -> str:
    """Chama a API da OpenAI."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    logger.info("Modelo: gpt-4o-mini")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise assistant that answers questions based on provided documents. Be helpful but honest about the limits of the available information."
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            max_tokens=1000,
            temperature=0.3,  
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Erro ao chamar OpenAI: {e}")
        raise