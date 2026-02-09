from fastapi import APIRouter, UploadFile, File
from typing import List

from libs.services.document_service import process_documents
from libs.structures.documents import DocumentUploadResponse

router = APIRouter(tags=["documents"])

@router.post("/", response_model=DocumentUploadResponse)
def store(files: List[UploadFile] = File(...)):
    return process_documents(files)
