# ================= METADATA SEARCH (ALL PAGES) + LLM =================

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ------------------ EMBEDDING MODEL ------------------
embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# ------------------ LLM GENERATOR ------------------
generator = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=150,
    temperature=0.3
)

# ------------------ QDRANT CONNECTION ------------------
client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "rag_collection"

# ------------------ METADATA RETRIEVE (ALL PAGES) ------------------
def metadata_retrieve(query, top_k=5):
    """
    Retrieve top_k chunks from the Qdrant collection based purely on semantic similarity.
    Metadata filtering removed, searches all pages.
    """
    # 1️ Convert query to embedding
    query_vector = embedder.encode(query).tolist()

    #  Query Qdrant (all chunks)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )

    # 3️ Extract text from payload
    chunks = [point.payload["text"] for point in results.points]
    return chunks

# ------------------ LLM ANSWER GENERATION ------------------
def generate_answer(question, chunks):
    """
    Generate answer using LLM based on retrieved chunks.
    """
    context = "\n".join(chunks)

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

# ------------------ FULL RAG PIPELINE ------------------
def rag_metadata_qa(question, top_k=5):
    """
    Retrieve chunks from ALL pages and generate answer.
    """
    chunks = metadata_retrieve(query=question, top_k=top_k)
    return generate_answer(question, chunks)




