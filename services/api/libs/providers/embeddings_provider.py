from typing import List

SPACY_MODEL = "pt_core_news_lg"

_spacy_nlp = None

def _load_spacy_model():
    import spacy

    return spacy.load(SPACY_MODEL)

def create_embedding(text: str) -> List[float]:
    global _spacy_nlp

    if _spacy_nlp is None:
        _spacy_nlp = _load_spacy_model()

    doc = _spacy_nlp(text)
    return doc.vector.tolist()


def create_embeddings(texts: List[str]) -> List[List[float]]:
    global _spacy_nlp

    if _spacy_nlp is None:
        _spacy_nlp = _load_spacy_model()

    docs = list(_spacy_nlp.pipe(texts))
    return [doc.vector.tolist() for doc in docs]
