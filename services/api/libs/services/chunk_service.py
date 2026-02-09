from typing import List
import re

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text) 
    text = text.strip()
    return text


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[str]:
    if not text:
        return []

    text = clean_text(text)
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

        start += chunk_size - chunk_overlap

        if start < 0:
            start = 0

    return chunks
