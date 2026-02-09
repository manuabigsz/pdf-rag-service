from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Optional
from datetime import datetime

class DocumentUploadResponse(BaseModel):
    message: str
    documents_indexed: int
    total_chunks: int


class DocumentMetadata(BaseModel):
    filename: str
    content_type: str
    size_bytes: Optional[int] = None
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)


class Document(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    metadata: DocumentMetadata
    text: Optional[str] = None  
    total_chunks: Optional[int] = None


class DocumentChunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    text: str
    chunk_index: int
