# backend/app/models/schemas.py
from pydantic import BaseModel
from typing import List, Optional, Dict

class Metadata(BaseModel):
    theme: Optional[str] = None
    type: Optional[str] = None

class Source(BaseModel):
    article_number: str
    content: str
    metadata: Metadata
    score: float

class ChatRequest(BaseModel):
    query: str
    mode: str = "advanced"  # naive, advanced, or compare


class ChatResponseResult(BaseModel):
    answer: str
    sources: List[Source]
    processing_time: float


class ChatResponse(BaseModel):
    answer: Optional[str] = None
    sources: Optional[List[Source]] = None
    processing_time: Optional[float] = None
    comparison: Optional[Dict[str, ChatResponseResult]] = None