from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from rag_engine import process_and_upload, get_answer
import os, shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # This tells the backend: "I'll talk to any frontend"
    allow_methods=["*"], # "I'll allow any action (GET, POST, etc.)"
    allow_headers=["*"], # "I'll accept any extra info in the request"
)

# Ensure directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse('static/index.html')

@app.post("/upload")
async def handle_upload(text: str = Form(None), file: UploadFile = File(None)):
    content = text
    name = "manual_input"
    if file:
        name = file.filename
        with open(f"uploads/{name}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        with open(f"uploads/{name}", "r") as f: # Simplified: assumes text/plain
            content = f.read()
    
    count = process_and_upload(content, name)
    return {"status": "success", "chunks": count}

@app.post("/query")
async def handle_query(question: str = Form(...)):
    answer, context, meta = get_answer(question)
    return {"answer": answer, "sources": context, "metadata": meta}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)