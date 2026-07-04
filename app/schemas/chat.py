"""Pydantic 请求/响应模型。"""
from pydantic import BaseModel, Field
from typing import List, Optional


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None


class SourceDoc(BaseModel):
    text: str
    source: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]
    session_id: str


class IngestResponse(BaseModel):
    status: str
    chunks: int
    filename: str
