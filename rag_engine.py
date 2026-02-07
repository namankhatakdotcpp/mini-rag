import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec
import cohere
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- INITIALIZATION ---
genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Use gemini-2.5-flash for better 2026 quota availability
MODEL_NAME = "gemini-2.5-flash"
INDEX_NAME = os.getenv("PINECONE_INDEX", "mini-rag")

# Pinecone Index Setup
if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=768, 
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

index = pc.Index(INDEX_NAME)

def generate_with_retry(prompt, retries=3):
    """Simple wrapper to handle 429 Rate Limit errors."""
    for i in range(retries):
        try:
            return genai_client.models.generate_content(
                model=MODEL_NAME, 
                contents=prompt
            )
        except Exception as e:
            if "429" in str(e) and i < retries - 1:
                wait = (i + 1) * 5
                print(f"⚠️ Rate limited. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise e

def process_and_upload(text, source_name):
    """Chunks, embeds, and uploads to Pinecone."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(text)
    
    vectors = []
    for i, chunk in enumerate(chunks):
        # Using text-embedding-004
        emb = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        vectors.append({
            "id": f"{source_name}_{i}",
            "values": emb.embeddings[0].values,
            "metadata": {"text": chunk, "source": source_name}
        })
    
    index.upsert(vectors=vectors)
    return len(vectors)

def get_answer(query):
    """RAG Flow: Retrieve -> Rerank -> Generate"""
    start_time = time.time()

    # 1. Retrieval
    query_emb = genai_client.models.embed_content(
        model="gemini-embedding-001", 
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    res = index.query(vector=query_emb.embeddings[0].values, top_k=10, include_metadata=True)
    
    # 2. Reranking (Cohere)
    docs = [match.metadata['text'] for match in res.matches]
    reranked = co.rerank(query=query, documents=docs, top_n=3, model='rerank-english-v3.0')
    
    # 3. Augmentation
    context_list = [f"[{i+1}] {docs[r.index]}" for i, r in enumerate(reranked.results)]
    context_text = "\n\n".join(context_list)
    
    # 4. Generation (Using 2.5 Flash with Retry logic)
    prompt = f"Use ONLY the context to answer. If unsure, say 'I don't know'.\n\nContext:\n{context_text}\n\nQuestion: {query}"
    response = generate_with_retry(prompt)
    
    # 5. Metadata for Assessment
    usage = response.usage_metadata
    metadata = {
        "tokens": usage.total_token_count,
        "time": f"{time.time() - start_time:.2f}s",
        "model": MODEL_NAME
    }
    
    return response.text, context_text, metadata