# RAG System

A simple Retrieval-Augmented Generation (RAG) pipeline using FAISS for vector storage and semantic search.

## How It Works

1. Upload a PDF or TXT document
2. Document is chunked, embedded, and stored in a FAISS index
3. Ask a question — the system finds the most relevant chunks and generates an answer

## Tech Stack

- **Embeddings** — `Qwen/Qwen3-Embedding-0.6B`
- **LLM** — `distilgpt2`
- **Vector Store** — FAISS (local, no server needed)
- **API** — FastAPI

## Project Structure

```
app.py              # FastAPI app + entrypoint
utils/
  ingest.py         # Parse, chunk, embed, store in FAISS
  search.py         # Semantic search + answer generation
faiss_data/         # Auto-created after first upload
requirement.txt
```

## Setup

```bash
pip install -r requirement.txt
```

## Run

```bash
python app.py
```

API will be available at `http://localhost:8000`
Swagger docs at `http://localhost:8000/docs`

## API Endpoints

### `GET /health`
Check if the API is running.

### `POST /upload`
Upload a PDF or TXT document.

- **Body**: `multipart/form-data` with a `file` field
- **Supported formats**: `.pdf`, `.txt`

**Response:**
```json
{
  "message": "Document uploaded and indexed successfully.",
  "filename": "example.pdf",
  "total_chunks": 42
}
```

### `POST /query`
Ask a question against the uploaded document.

**Request:**
```json
{
  "question": "What is the refund policy?",
  "top_k": 5
}
```

**Response:**
```json
{
  "question": "What is the refund policy?",
  "answer": "..."
}
```
