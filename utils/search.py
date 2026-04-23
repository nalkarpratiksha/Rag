# ================= SEMANTIC SEARCH + LLM =================

import pickle
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ------------------ CONFIG ------------------
INDEX_PATH = "./faiss_data/index.faiss"
CHUNKS_PATH = "./faiss_data/chunks.pkl"

# ------------------ MODELS ------------------
embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

generator = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=150,
    temperature=0.3,
)


# ------------------ LOAD FAISS STORE ------------------
def load_store():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise RuntimeError("No index found. Please upload a document first.")
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunk_store = pickle.load(f)
    return index, chunk_store


# ------------------ SEMANTIC SEARCH ------------------
def semantic_search(query: str, top_k: int = 5):
    index, chunk_store = load_store()

    query_vector = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(query_vector)

    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vector, k)

    return [
        chunk_store[indices[0][i]]["text"]
        for i in range(k)
        if indices[0][i] != -1
    ]


# ------------------ ANSWER GENERATION ------------------
def generate_answer(question: str, chunks: list) -> str:
    context = "\n".join(chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not present, say "Answer not found".

Context:
{context}

Question:
{question}

Answer:"""

    response = generator(prompt)
    answer = response[0]["generated_text"].split("Answer:")[-1].strip()

    # distilgpt2 may return empty — fall back to raw context
    if not answer:
        answer = context

    return answer


# ------------------ FULL RAG PIPELINE ------------------
def rag_qa(question: str, top_k: int = 5) -> str:
    chunks = semantic_search(question, top_k=top_k)
    return generate_answer(question, chunks)
