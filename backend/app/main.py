import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import health, ingest, query
from app.core.logging import setup_logging

# Initialize logging once at startup
setup_logging()

app = FastAPI(
    title="Mini RAG System",
    description="A minimal, production-minded RAG application",
    version="1.0.0"
)

# CORS for local frontend development.
# In production, restrict this to your deployed frontend origin(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Mini RAG backend is running.",
        "docs": "/docs",
    }


@app.get("/version")
def version():
    return {
        "service": "mini-rag-backend",
        "version": app.version,
        "git_commit": os.getenv("RENDER_GIT_COMMIT"),
    }


# Register routers
app.include_router(health.router)
app.include_router(ingest.router, prefix="/api")
app.include_router(query.router, prefix="/api")
