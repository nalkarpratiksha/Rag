# ================= DOCUMENT INGESTION =================

import re
import io
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ------------------ CONFIG ------------------
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 50      # overlap between chunks
INDEX_PATH = "./faiss_data/index.faiss"
CHUNKS_PATH = "./faiss_data/chunks.pkl"

# ------------------ SHARED INSTANCES ------------------
embedder = SentenceTransformer(EMBEDDING_MODEL)


# ------------------ CHUNKING ------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ------------------ PARSE DOCUMENT ------------------
def parse_document(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from PDF or TXT file."""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
            return text
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {e}")

    elif ext == ".txt":
        return file_bytes.decode("utf-8", errors="ignore")

    else:
        raise ValueError(f"Unsupported file type: {ext}. Only PDF and TXT are supported.")


# ------------------ LOAD OR CREATE INDEX ------------------
def load_store():
    """Load existing FAISS index and chunk metadata, or return empty ones."""
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunk_store = pickle.load(f)
    else:
        index = None
        chunk_store = []  # list of dicts: {text, source, chunk_index}
    return index, chunk_store


def save_store(index, chunk_store):
    """Persist FAISS index and chunk metadata to disk."""
    os.makedirs("./faiss_data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunk_store, f)


# ------------------ INGEST DOCUMENT ------------------
def ingest_document(file_bytes: bytes, filename: str) -> dict:
    """
    Full pipeline: parse -> chunk -> embed -> store in FAISS.
    Returns a summary dict.
    """
    # 1. Parse
    text = parse_document(file_bytes, filename)
    if not text.strip():
        raise ValueError("Document appears to be empty or unreadable.")

    # 2. Chunk
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No chunks could be created from document.")

    # 3. Embed
    embeddings = embedder.encode(chunks, show_progress_bar=False).astype("float32")
    vector_size = embeddings.shape[1]

    # 4. Load existing store
    index, chunk_store = load_store()

    # 5. Create index if first time
    if index is None:
        index = faiss.IndexFlatIP(vector_size)  # Inner product (cosine after normalization)

    # 6. Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # 7. Add to index
    start_idx = len(chunk_store)
    index.add(embeddings)

    # 8. Store metadata
    for i, chunk in enumerate(chunks):
        chunk_store.append({
            "text": chunk,
            "source": filename,
            "chunk_index": start_idx + i,
        })

    # 9. Save
    save_store(index, chunk_store)

    return {
        "filename": filename,
        "total_characters": len(text),
        "total_chunks": len(chunks),
        "collection": "faiss_index",
    }
