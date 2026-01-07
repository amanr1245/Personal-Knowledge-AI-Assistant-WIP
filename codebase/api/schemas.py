"""Request and response schemas for the API."""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to ask the knowledge base")
    use_reranking: bool = Field(default=True, description="Whether to use LLM reranking")
    expand_context: bool = Field(default=True, description="Whether to expand with adjacent chunks")


class ChunkSource(BaseModel):
    file: str
    chunk_index: int
    similarity: float
    header: Optional[str] = None
    preview: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[ChunkSource]
    processing_time_ms: int
    chunks_retrieved: int
    chunks_used: int


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=50)


class SearchResult(BaseModel):
    file: str
    similarity: float
    preview: str
    chunk_count: int


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_files: int


class DocumentInfo(BaseModel):
    file: str
    chunks: int
    file_type: str
    indexed_at: Optional[datetime] = None


class StatsResponse(BaseModel):
    file_count: int
    chunk_count: int
    file_types: dict
    database_size_mb: Optional[float] = None


class UploadResponse(BaseModel):
    success: bool
    file: str
    chunks_created: int
    message: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
