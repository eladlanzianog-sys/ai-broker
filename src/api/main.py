"""FastAPI application factory."""
from fastapi import FastAPI

from src.api.routes import router


def create_app() -> FastAPI:
    application = FastAPI(
        title="Stock Analysis Agents",
        description="Multi-Agent AI System for Autonomous Stock Market Analysis",
        version="0.1.0",
    )
    application.include_router(router)
    return application


app = create_app()
