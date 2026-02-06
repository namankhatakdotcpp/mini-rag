import os
import requests
from fastapi import HTTPException
from typing import List

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_EMBEDDING_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_LLM_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

def generate_embedding(text: str) -> List[float]:
	"""
	Generate embedding for the given text using Hugging Face Inference API.
	"""
	if not HF_API_TOKEN:
		raise HTTPException(status_code=500, detail="Hugging Face API token is not set.")
	
	headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
	payload = {"inputs": text}

	try:
		response = requests.post(HF_EMBEDDING_URL, headers=headers, json=payload, timeout=10)
		response.raise_for_status()
		embedding = response.json()
		if isinstance(embedding, list) and len(embedding) > 0:
			return embedding[0]  # Return the first embedding vector
		else:
			raise HTTPException(status_code=500, detail="Invalid response format from embedding service.")
	except requests.exceptions.RequestException:
		raise HTTPException(status_code=500, detail="Embedding service unavailable.")

def generate_answer(question: str, context_chunks: List[str]) -> str:
	"""
	Generate an answer using the Hugging Face Inference API.
	"""
	if not HF_API_TOKEN:
		raise HTTPException(status_code=500, detail="Hugging Face API token is not set.")
	
	headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
	context = "\n".join(context_chunks)
	prompt = f"Answer the question using ONLY the context below:\n\n{context}\n\nQuestion: {question}"
	payload = {"inputs": prompt}

	try:
		response = requests.post(HF_LLM_URL, headers=headers, json=payload, timeout=20)
		response.raise_for_status()
		result = response.json()
		if isinstance(result, list) and "generated_text" in result[0]:
			return result[0]["generated_text"]
		else:
			raise HTTPException(status_code=500, detail="Invalid response format from answer generation service.")
	except requests.exceptions.RequestException:
		raise HTTPException(status_code=500, detail="Answer generation service unavailable.")