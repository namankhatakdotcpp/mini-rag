import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from app.core.logging import setup_logging

# Initialize logging once at startup
setup_logging()

app = FastAPI(
    title="Mini RAG System",
    description="A minimal, production-minded RAG application",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mini-b1qoueqff-namans-projects-dfbad539.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.options("/{path:path}")
async def preflight_handler(path: str, request: Request):
    return Response(status_code=200)


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


from app.api import health, ingest, query

app.include_router(health.router)
app.include_router(ingest.router, prefix="/api")
app.include_router(query.router, prefix="/api")
