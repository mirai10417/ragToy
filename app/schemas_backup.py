from pydantic import BaseModel
from typing import List

class AskRequest(BaseModel):
    question: str
    top_k: int = 4

class SourceChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]