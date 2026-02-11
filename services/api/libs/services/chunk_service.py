from typing import List
import re
import nltk
from loguru import logger

try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    from nltk.tokenize import sent_tokenize
    
    try:
        sentences = sent_tokenize(text)
        return [s for s in sentences if len(s.strip()) > 10]
    except Exception as e:
        logger.warning(f"Erro ao dividir em sentenças: {e}")
        return [s.strip() + '.' for s in text.split('.') if len(s.strip()) > 10]


def chunk_text(
    text: str,
    max_chunk_size: int = 1000,  
    chunk_overlap: int = 100,
) -> List[str]:
    if not text or len(text.strip()) < 50:
        logger.warning("Texto vazio ou muito curto para chunking")
        return []

    text = clean_text(text)
    
    if len(text) <= max_chunk_size:
        return [text]

    chunks: List[str] = []
    
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
    
    if not paragraphs:
        paragraphs = [text]
    
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chunk_size:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            if len(para) > max_chunk_size:
                sentences = split_into_sentences(para)
                temp_chunk = ""
                
                for sent in sentences:
                    if len(temp_chunk) + len(sent) + 1 <= max_chunk_size:
                        if temp_chunk:
                            temp_chunk += " " + sent
                        else:
                            temp_chunk = sent
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        
                        if len(sent) > max_chunk_size:
                            chunks.append(sent[:max_chunk_size].strip())
                            temp_chunk = ""
                        else:
                            temp_chunk = sent
                
                current_chunk = temp_chunk
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    chunks = [c for c in chunks if len(c.strip()) > 50]
    
    if chunk_overlap > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, chunk_overlap)

    logger.info(f"Texto dividido em {len(chunks)} chunks (tamanho médio: {sum(len(c) for c in chunks) // len(chunks)} chars)")
    
    return chunks


def _apply_overlap(chunks: List[str], overlap_chars: int) -> List[str]:
    if len(chunks) <= 1:
        return chunks
    
    overlapped_chunks = [chunks[0]]
    
    for i in range(1, len(chunks)):
        prev_chunk = overlapped_chunks[-1]
        overlap = prev_chunk[-overlap_chars:] if len(prev_chunk) > overlap_chars else prev_chunk
        
        overlap = overlap.lstrip()
        
        space_idx = overlap.find(' ')
        if space_idx > 0:
            overlap = overlap[space_idx+1:]
        
        new_chunk = overlap + " " + chunks[i]
        overlapped_chunks.append(new_chunk.strip())
    
    return overlapped_chunks