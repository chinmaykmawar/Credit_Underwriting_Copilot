from fastapi import APIRouter, Request
from src.api.schemas import HealthResponse, AskRequest, AskResponse


router = APIRouter()

@router.get("/health", response_model=HealthResponse,)
def health() -> HealthResponse:
    return HealthResponse(status="healthy")

@router.post("/ask", response_model=AskResponse)
def ask(request: Request, ask_params: AskRequest,) -> AskResponse:
    credit_app = request.app.state.credit_app
    response=credit_app.ask(ask_params.question, ask_params.top_k)
    return AskResponse(answer=response.answer, sources=response.sources)

