# =============== HYBRID SEARCH + LLM GENERATION (RAG) =================

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from rank_bm25 import BM25Okapi
import re


# ------------------ LOAD EMBEDDING MODEL ------------------
embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")


# ------------------ CONNECT TO QDRANT ------------------
client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "rag_collection"


# ------------------ TOKENIZER ------------------
def tokenize(text):
    return re.findall(r"\w+", text.lower())


# ------------------ HYBRID SEARCH ------------------
def hybrid_search(query, top_k=5, candidate_pool=30):
    # 1️ Semantic search
    query_vector = embedder.encode(query).tolist()

    semantic_hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=candidate_pool
    )

    # 2️ Keyword search (BM25)
    texts = [hit.payload["text"] for hit in semantic_hits]
    tokenized_corpus = [tokenize(text) for text in texts]

    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = tokenize(query)
    keyword_scores = bm25.get_scores(tokenized_query)

    # 3️ Combine scores
    combined = []
    for i, hit in enumerate(semantic_hits):
        score = hit.score + keyword_scores[i]
        combined.append((score, hit.payload["text"]))

    # 4️ Sort & return
    combined.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in combined[:top_k]]


# ------------------ LOAD LLM ------------------
generator = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=150,
    temperature=0.3
)


# ------------------ ANSWER GENERATION ------------------
def generate_answer(question, retrieved_chunks):
    context = "\n".join(retrieved_chunks)

    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say "Answer not found".

Context:
{context}

Question:
{question}

Answer:
"""

    response = generator(prompt)
    return response[0]["generated_text"].split("Answer:")[-1].strip()


# ------------------ FULL HYBRID RAG PIPELINE ------------------
def hybrid_rag_qa(question, top_k=3):
    chunks = hybrid_search(
        query=question,
        top_k=top_k
    )

    answer = generate_answer(question, chunks)
    return answer

