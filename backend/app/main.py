# import os

# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse, Response
# from app.core.logging import setup_logging

# # Initialize logging once at startup
# setup_logging()

# app = FastAPI(
#     title="Mini RAG System",
#     description="A minimal, production-minded RAG application",
#     version="1.0.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "https://mini-b1qoueqff-namans-projects-dfbad539.vercel.app"
#     ],
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "OPTIONS"],
#     allow_headers=["*"],
# )


# @app.exception_handler(Exception)
# async def global_exception_handler(request: Request, exc: Exception):
#     return JSONResponse(
#         status_code=200,
#         content={
#             "error": str(exc),
#             "answer": "Backend error",
#             "sources": []
#         },
#         headers={
#             "Access-Control-Allow-Origin": "https://mini-b1qoueqff-namans-projects-dfbad539.vercel.app",
#             "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
#             "Access-Control-Allow-Headers": "*",
#         },
#     )


# @app.options("/{path:path}")
# async def options_handler(path: str):
#     return JSONResponse(
#         status_code=200,
#         content={},
#         headers={
#             "Access-Control-Allow-Origin": "https://mini-b1qoueqff-namans-projects-dfbad539.vercel.app",
#             "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
#             "Access-Control-Allow-Headers": "*",
#         },
#     )


# @app.get("/")
# def root():
#     return {
#         "status": "ok",
#         "message": "Mini RAG backend is running.",
#         "docs": "/docs",
#     }


# @app.post("/api/query")
# async def query(request: QueryRequest):
#     question = request.question

#     # Check if documents table is empty
#     top_chunks = retrieve_top_chunks("", 1)  # Check if any documents exist
#     if not top_chunks:
#         return {"answer": "Please ingest documents first.", "sources": []}

#     from app.api import health, ingest
#     from app.api.query import router as query_router
#     try:
#         question_embedding = generate_embedding(question)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to embed question: {str(e)}")

#     # Retrieve top similar chunks
#     try:
#         retrieved_chunks = retrieve_top_chunks(question_embedding, top_k=5)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to retrieve top chunks: {str(e)}")

#     # Generate answer
#     try:
#         context_chunks = [chunk["text"] for chunk in retrieved_chunks]
#         answer = generate_answer(question, context_chunks)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")

#     return {"answer": answer, "sources": context_chunks}

# from fastapi.middleware.cors import CORSMiddleware

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],   # later restrict to vercel domain
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# from app.api import health, ingest, query

# app.include_router(health.router)
# app.include_router(ingest.router, prefix="/api")
# app.include_router(query.router, prefix="/api")


import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.logging import setup_logging

# Initialize logging once at startup
setup_logging()

app = FastAPI(
    title="Mini RAG System",
    description="A minimal, production-minded RAG application",
    version="1.0.0"
)

# ✅ SINGLE CORS MIDDLEWARE (DO NOT ADD AGAIN BELOW)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mini-b1qoueqff-namans-projects-dfbad539.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ✅ Proper global exception handler (returns real error)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "answer": "Backend error",
            "sources": []
        }
    )


# ✅ Handle OPTIONS preflight requests (for browser)
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


# ✅ Root health route
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Mini RAG backend is running.",
        "docs": "/docs",
    }


# ✅ Import routers ONLY (NO manual /api/query here)
from app.api import health, ingest, query

app.include_router(health.router)
app.include_router(ingest.router, prefix="/api")
app.include_router(query.router, prefix="/api")
