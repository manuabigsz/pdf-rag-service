from typing import List
import re
import nltk

try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")



def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def split_into_sentences(text: str) -> List[str]:
    """Quebra o texto em frases (usando nltk)."""
    from nltk.tokenize import sent_tokenize

    return sent_tokenize(text)


def chunk_text(
    text: str,
    max_chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[str]:

    if not text:
        return []

    text = clean_text(text)

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: List[str] = []

    for para in paragraphs:
        if len(para) <= max_chunk_size:
            chunks.append(para)
            continue

        sentences = split_into_sentences(para)
        current_chunk = ""

        for sent in sentences:
            if len(current_chunk) + len(sent) <= max_chunk_size:
                current_chunk += " " + sent if current_chunk else sent
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sent

        if current_chunk:
            chunks.append(current_chunk.strip())

    final_chunks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            final_chunks.append(chunk)
        else:
            prev_words = final_chunks[-1].split()
            overlap_words = prev_words[-chunk_overlap:] if len(prev_words) > chunk_overlap else prev_words

            new_chunk = " ".join(overlap_words) + " " + chunk
            final_chunks.append(new_chunk.strip())

    return final_chunks
