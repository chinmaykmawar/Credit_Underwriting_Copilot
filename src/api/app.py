from fastapi import FastAPI
from src.api.routes import router
from src.CreditUnderwritingApp import CreditUnderwritingApp
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.credit_app = CreditUnderwritingApp()
    yield

app = FastAPI(
    title="AI Credit Underwriting API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)