# 🚀 Mini RAG System (Production-Style)

A full-stack Retrieval-Augmented Generation (RAG) system built using modern LLM infrastructure.  
This project demonstrates how real-world AI systems retrieve, rank, and generate grounded responses from custom data.

## 🔥 Features
- Upload documents or text
- Chunking + embedding pipeline
- Vector search using Pinecone
- Reranking using Cohere
- Grounded answer generation using Gemini
- FastAPI backend + interactive frontend
- Source-aware responses (no hallucinations)

## 🧠 Tech Stack
- FastAPI (backend)
- Gemini API (LLM + embeddings)
- Pinecone (vector database)
- Cohere (reranking)
- HTML/CSS/JS frontend
- Render + Vercel deployment

## ⚙️ How it works
1. User uploads text/document
2. Text is chunked and embedded
3. Stored in Pinecone vector DB
4. User asks question
5. Relevant chunks retrieved
6. Cohere reranks context
7. Gemini generates grounded answer

## 🎯 Use Cases
- AI document search engine
- Chat with PDFs
- Knowledge base assistant
- Internal company AI chatbot

## 🏆 Why this project matters
Most chatbots hallucinate.  
This system retrieves real data first and then generates answers — just like production AI systems at OpenAI, Google, and Perplexity.

## 👨‍💻 Author
Naman  
AI/ML + Full Stack Developer  
