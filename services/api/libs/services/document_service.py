from typing import List
from fastapi import UploadFile
from loguru import logger

from langchain_core.documents import Document

from libs.services.pdf_service import extract_text_from_pdf
from libs.services.vector_service import get_text_splitter, add_documents
from libs.structures.documents import Document as DocModel, DocumentMetadata


def process_documents(files: List[UploadFile]) -> dict:
    documents = []
    total_chunks = 0
    splitter = get_text_splitter()

    for file in files:
        metadata = DocumentMetadata(
            filename=file.filename,
            content_type=file.content_type,
            size_bytes=None,
        )
        doc_model = DocModel(metadata=metadata)

        text = extract_text_from_pdf(file)
        if not text or len(text.strip()) < 50:
            logger.warning(f"Documento {file.filename} vazio ou muito curto, ignorando")
            continue

        lc_doc = Document(
            page_content=text,
            metadata={
                "document_id": str(doc_model.id),
                "source": file.filename or "unknown",
            },
        )
        chunks = splitter.split_documents([lc_doc])
        doc_model.total_chunks = len(chunks)

        stored = add_documents(chunks)
        total_chunks += stored
        documents.append(doc_model)

    return {
        "message": "Documents processed successfully",
        "documents_indexed": len(documents),
        "total_chunks": total_chunks,
    }
