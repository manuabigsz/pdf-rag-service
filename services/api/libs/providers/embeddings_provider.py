from typing import List
import numpy as np

SPACY_MODEL = "pt_core_news_lg"

_spacy_nlp = None


def _load_spacy_model():
    import spacy
    return spacy.load(SPACY_MODEL)


def _get_nlp():
    global _spacy_nlp

    if _spacy_nlp is None:
        _spacy_nlp = _load_spacy_model()

    return _spacy_nlp


def _normalize_vector(vec: List[float]) -> List[float]:
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return vec
    return (arr / norm).tolist()


def create_embedding(text: str) -> List[float]:
    nlp = _get_nlp()

    doc = nlp(text)

    return _normalize_vector(doc.vector.tolist())


def create_embeddings(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    nlp = _get_nlp()

    docs = list(nlp.pipe(texts))

    embeddings = [doc.vector.tolist() for doc in docs]

    return [_normalize_vector(vec) for vec in embeddings]
