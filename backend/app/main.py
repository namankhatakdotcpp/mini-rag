import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=200,
        content={
            "error": str(exc),
            "answer": "Backend error",
            "sources": []
        },
        headers={
            "Access-Control-Allow-Origin": "https://mini-b1qoueqff-namans-projects-dfbad539.vercel.app",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )


@app.options("/{path:path}")
async def options_handler(path: str):
    return JSONResponse(
        status_code=200,
        content={},
        headers={
            "Access-Control-Allow-Origin": "https://mini-b1qoueqff-namans-projects-dfbad539.vercel.app",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
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


from app.api import health, ingest, query

app.include_router(health.router)
app.include_router(ingest.router, prefix="/api")
app.include_router(query.router, prefix="/api")
