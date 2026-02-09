from typing import List
from fastapi import UploadFile

from libs.services.pdf_service import extract_text_from_pdf
from libs.services.chunk_service import chunk_text
from libs.services.vector_service import store_chunks

from libs.structures.documents import Document, DocumentMetadata

def process_documents(files: List[UploadFile]) -> dict:
    documents = []
    total_chunks = 0

    for file in files:
        metadata = DocumentMetadata(
            filename=file.filename,
            content_type=file.content_type,
            size_bytes=None 
        )

        document = Document(metadata=metadata)

        text = extract_text_from_pdf(file)
        document.text = text

        chunks = chunk_text(text)
        document.total_chunks = len(chunks)

        stored = store_chunks(chunks, document.id)

        total_chunks += stored
        documents.append(document)

    return {
        "message": "Documents processed successfully",
        "documents_indexed": len(documents),
        "total_chunks": total_chunks
    }
