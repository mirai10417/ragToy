from pydantic import BaseModel
from typing import List, Optional


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


class SourceChunk(BaseModel):
    rank: int
    source: Optional[str] = None
    page: Optional[int] = None
    text: str
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    matched_count: int
    sources: List[SourceChunk]