import os
from typing import List
from openai import OpenAI
from loguru import logger

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def generate_answer(question: str, context_chunks: List[str]) -> str:
    logger.info("=== INÍCIO: generate_answer ===")
    logger.info(f"Pergunta do usuário: {question}")
    logger.info(f"Número de chunks recebidos: {len(context_chunks) if context_chunks else 0}")

    if not context_chunks or len(context_chunks) == 0:
        logger.warning("Nenhum chunk foi recuperado do Chroma.")
        return "I don't have enough information to answer this question based on the provided documents."

    for i, chunk in enumerate(context_chunks):
        logger.debug(f"\n--- CHUNK {i+1} (tamanho={len(chunk)}) ---\n{chunk[:500]}")

    context = "\n\n".join(
        [f"--- DOCUMENTO {i+1} ---\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )

    prompt = f"""
You are an assistant that answers questions STRICTLY based on the provided documents.

========================
DOCUMENTOS DISPONÍVEIS:
========================
{context}
========================

PERGUNTA:
{question}

REGRAS OBRIGATÓRIAS:
1. Você DEVE responder com base nos documentos acima.
2. Se os documentos NÃO contiverem informação suficiente, responda exatamente:
"I don't have enough information to answer this question based on the provided documents."
3. Se responder, cite trechos dos documentos relevantes.
4. Seja direto e objetivo.

RESPOSTA:
"""

    logger.info("=== PROMPT ENVIADO AO LLM ===")
    logger.debug(prompt)

    answer = _generate_with_openai(prompt)

    logger.info("=== RESPOSTA DO LLM ===")
    logger.info(answer)

    logger.info("=== FIM: generate_answer ===")

    return answer



def _generate_with_openai(prompt: str) -> str:
    logger.info("Chamando OpenAI")
    logger.info("Modelo: gpt-4.1")
    logger.info("temperature=0.1, max_tokens=800")

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "You are a precise assistant that answers based only on provided documents."
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=800,
        temperature=0.1,
    )

    logger.debug(f"Resposta bruta da OpenAI: {response}")

    return response.choices[0].message.content.strip()
