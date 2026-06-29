from pydantic import BaseModel
from src.rag.models import SourceDetails

class AskRequest(BaseModel):
    question: str
    top_k: int = 10

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceDetails]

class HealthResponse(BaseModel):
    status: str