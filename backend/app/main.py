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
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router)
app.include_router(ingest.router, prefix="/api")
app.include_router(query.router, prefix="/api")
