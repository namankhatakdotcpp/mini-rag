# 🚀 Mini RAG — Production-Grade Retrieval Augmented Generation System

A full-stack **production-style Retrieval-Augmented Generation (RAG)** system that enables grounded question answering over custom documents using modern LLM infrastructure.

This project demonstrates how real-world AI systems like **ChatGPT Retrieval, Perplexity AI, and enterprise copilots** are built.

---

# 🧠 System Overview

Traditional LLMs hallucinate.
This system eliminates hallucinations by retrieving real context before generating answers.

### Pipeline

```
User Query
   ↓
Embedding (Gemini)
   ↓
Vector Search (Pinecone)
   ↓
Reranking (Cohere)
   ↓
Grounded Generation (Gemini)
   ↓
Answer + Sources
```

---

# ✨ Features

### 🔹 Document Ingestion

* Upload text or files
* Smart chunking with overlap
* Metadata tracking for sources

### 🔹 Vector Retrieval

* Semantic search via Pinecone
* Cosine similarity matching
* Top-K context retrieval

### 🔹 Reranking Layer

* Cohere rerank model
* Improves relevance before generation
* Production-style search pipeline

### 🔹 Grounded Answer Generation

* Gemini LLM
* Uses retrieved context only
* Prevents hallucinations
* Source-aware answers

### 🔹 Full Stack System

* FastAPI backend
* Interactive frontend console
* Real-time ingestion and querying
* Deploy-ready architecture

---

# 🏗️ Tech Stack

## AI/ML

* Gemini API (LLM + embeddings)
* Pinecone (vector database)
* Cohere (reranking)

## Backend

* FastAPI
* Python

## Frontend

* HTML/CSS/JS

## Deployment

* Render (backend)
* Vercel (frontend)
* GitHub

---

# ⚙️ Architecture (Production Style)

## Ingestion Pipeline

1. Document → chunking
2. Chunk → embedding
3. Embedding → vector DB
4. Metadata stored with vectors

## Query Pipeline

1. Query → embedding
2. Vector search (top-k)
3. Cohere rerank
4. Context construction
5. LLM generation

## Grounded Response

* Uses retrieved context only
* Returns answer with sources
* Prevents hallucinations

---

# 🧪 Example Use Cases

* Chat with PDFs
* AI research assistant
* Internal company knowledge bot
* Customer support AI
* Documentation search engine

---

# 📊 Why This Project Matters

Most AI demos = simple chatbot calls.

This project replicates **real-world LLM infrastructure**:

| Feature                 | Basic Chatbot | This Project |
| ----------------------- | ------------- | ------------ |
| Vector search           | ❌             | ✅            |
| RAG pipeline            | ❌             | ✅            |
| Reranking               | ❌             | ✅            |
| Grounded answers        | ❌             | ✅            |
| Production architecture | ❌             | ✅            |

Used in systems like:

* OpenAI retrieval stack
* Perplexity AI
* Google Gemini search
* Enterprise AI copilots

---

# 🧑‍💻 How to Run Locally

### Clone repo

```
git clone https://github.com/namankhatakdotcpp/mini-rag
cd mini-rag
```

### Create environment

```
python -m venv venv
source venv/bin/activate
```

### Install dependencies

```
pip install -r requirements.txt
```

### Add API keys (.env)

```
GEMINI_API_KEY=
PINECONE_API_KEY=
COHERE_API_KEY=
PINECONE_INDEX=mini-rag
```

### Run server

```
python main.py
```

Open browser:

```
http://localhost:8000
```

---

# 🧠 Engineering Highlights

* Retrieval-first architecture
* Modular RAG pipeline
* Grounded answer generation
* Vector DB integration
* Production-style design

---

# 🏆 Resume Value

Demonstrates:

* Applied LLM engineering
* Vector databases
* RAG architecture
* Full-stack AI system design
* Production deployment thinking

---

# 👨‍💻 Author

**Naman**
AI/ML Engineer | Full Stack Developer

Built to demonstrate real-world LLM infrastructure and production RAG pipelines.
