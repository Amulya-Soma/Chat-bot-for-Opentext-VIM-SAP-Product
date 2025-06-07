# rag_engine.py

import numpy as np
import faiss
import subprocess
from sentence_transformers import SentenceTransformer, CrossEncoder

# Load FAISS & Embeddings
chunks = np.load("Data/chunks.npy", allow_pickle=True)
embeddings = np.load("Data/embeddings.npy")
dimension = embeddings.shape[1]

index = faiss.IndexHNSWFlat(dimension, 32)
index.add(embeddings)

# Models
embed_model = SentenceTransformer("all-mpnet-base-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def generate_reformulations(query):
    return [
        query,
        f"Explain how to configure {query}",
        f"What are the SAP VIM TCodes related to {query}?",
        f"How to define configuration for {query}?",
        f"Which fields or tables are used to setup {query}?",
    ]

def rag_fusion_search_with_rerank(query, k_per_query=5, final_top_n=7):
    variations = generate_reformulations(query)
    seen_chunks = set()
    candidate_chunks = []

    for variation in variations:
        query_embedding = embed_model.encode([variation])
        D, I = index.search(query_embedding, k_per_query)
        for i in I[0]:
            chunk = chunks[i]
            if chunk not in seen_chunks:
                seen_chunks.add(chunk)
                candidate_chunks.append(chunk)

    rerank_inputs = [(query, chunk) for chunk in candidate_chunks]
    scores = reranker.predict(rerank_inputs)
    ranked_chunks = sorted(zip(candidate_chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in ranked_chunks[:final_top_n]]

def build_prompt(query, contexts):
    joined_context = "\n\n".join(contexts)
    return f"""
You are an SAP VIM domain expert helping engineers understand different VIM (Vendor Invoice Management) functionalities in SAP.

Tone: Friendly, clear, and helpful.

Context:
{joined_context}

User Question:
{query}

Please provide a thoughtful, concise answer based mostly on the context above.
Focus on configuration details where possible.
Reply in clear concise summarized bullet points.
"""

def call_ollama(prompt, model="mistral"):
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return result.stdout.decode("utf-8")

def generate_response(query, model="mistral"):
    retrieved_chunks = rag_fusion_search_with_rerank(query)
    prompt = build_prompt(query, retrieved_chunks)
    return call_ollama(prompt, model=model), retrieved_chunks
