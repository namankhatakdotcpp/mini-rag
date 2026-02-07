import os
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone
import cohere

load_dotenv()

def test_gemini():
    print("Testing Gemini API...")
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        # Test Generation
        res = client.models.generate_content(model="gemini-2.0-flash", contents="Hi")
        # Test Embedding
        emb = client.models.embed_content(model="text-embedding-004", contents="Hi")
        print("✅ Gemini: OK (Generation & Embeddings)")
        return True
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        return False

def test_pinecone():
    print("\nTesting Pinecone Connection...")
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        idx_name = os.getenv("PINECONE_INDEX", "mini-rag")
        # Check if index exists
        if idx_name in [idx.name for idx in pc.list_indexes()]:
            index = pc.Index(idx_name)
            stats = index.describe_index_stats()
            print(f"✅ Pinecone: OK (Index '{idx_name}' found with {stats['total_vector_count']} vectors)")
        else:
            print(f"⚠️ Pinecone: Index '{idx_name}' not found. It will be created on first run.")
        return True
    except Exception as e:
        print(f"❌ Pinecone Error: {e}")
        return False

def test_cohere():
    print("\nTesting Cohere Reranker...")
    try:
        co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
        res = co.rerank(
            model="rerank-english-v3.0",
            query="What is AI?",
            documents=["AI is math.", "I like pizza."],
            top_n=1
        )
        print("✅ Cohere: OK (Reranking)")
        return True
    except Exception as e:
        print(f"❌ Cohere Error: {e}")
        return False

if __name__ == "__main__":
    print("--- RAG SYSTEM PRE-FLIGHT CHECK ---")
    g = test_gemini()
    p = test_pinecone()
    c = test_cohere()
    print("\n-----------------------------------")
    if all([g, p, c]):
        print("🚀 ALL SYSTEMS GO! Your backend is ready.")
    else:
        print("🚧 Please fix the errors above before running main.py.")