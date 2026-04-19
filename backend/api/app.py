"""FastAPI application entry point."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.tuning_routes import router as tuning_router
from api.data_routes import router as data_router
from api.config_routes import router as config_router
from api.consultant_routes import router as consultant_router
from api.sessions_routes import router as sessions_router
from config import settings

app = FastAPI(
    title="PID V2 - Intelligent Tuning System",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tuning_router, prefix="/api")
app.include_router(data_router, prefix="/api")
app.include_router(config_router, prefix="/api")
app.include_router(consultant_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")
