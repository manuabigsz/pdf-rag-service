import os
from typing import List, Tuple
from loguru import logger

from langchain_openai import ChatOpenAI

from libs.services.vector_service import get_retriever


TOP_K = 10

SYSTEM = (
    "You are a precise assistant that answers questions based on provided documents. "
    "Be helpful but honest about the limits of the available information. "
    "When possible, reference which document (source) your answer comes from."
)

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=1000,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    return _llm


def process_question(question: str) -> Tuple[str, List[str]]:
    logger.info(f"Processando pergunta: {question}")

    retriever = get_retriever(top_k=TOP_K)
    docs = retriever.invoke(question)

    if not docs:
        logger.warning("Nenhum chunk relevante encontrado")
        return "No relevant information found in the documents.", []

    context_parts = []
    for doc in docs:
        content = getattr(doc, "page_content", "") or ""
        meta = getattr(doc, "metadata", None) or {}
        source = meta.get("source") or meta.get("document_id") or "documento"
        context_parts.append(f"[Fonte: {source}]\n{content}")
    context = "\n\n".join(context_parts)

    prompt = f"""Use os documentos abaixo para responder à pergunta. Se os documentos não tiverem informação suficiente, diga isso claramente.
        Documentos:
        {context}
        Pergunta: {question}
        Responda com base nos documentos acima:"""

    llm = _get_llm()
    try:
        response = llm.invoke([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ])
        answer = (response.content or "").strip()
    except Exception as e:
        logger.error(f"Erro ao chamar LLM: {e}")
        raise

    if not answer:
        answer = "No relevant information found in the documents."

    references = [
        getattr(doc, "page_content", "") or ""
        for doc in docs
        if getattr(doc, "page_content", None)
    ]

    logger.info(f"Encontrados {len(references)} chunks relevantes")
    return answer, references
