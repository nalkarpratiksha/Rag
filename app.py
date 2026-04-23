# ================= RAG SYSTEM =================

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.ingest import ingest_document
from utils.search import rag_qa


# ================================================================
#  APP SETUP
# ================================================================

app = FastAPI(
    title="RAG System API",
    description="Upload a document and ask questions using semantic search.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------ SCHEMAS ------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    question: str
    answer: str


class UploadResponse(BaseModel):
    message: str
    filename: str
    total_chunks: int


# ------------------ HEALTH CHECK ------------------

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "message": "RAG API is running"}


# ------------------ UPLOAD DOCUMENT ------------------

@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF or TXT file. It will be chunked, embedded, and stored."""
    allowed_types = ["application/pdf", "text/plain"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Only PDF and TXT are allowed.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = ingest_document(file_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return UploadResponse(
        message="Document uploaded and indexed successfully.",
        filename=result["filename"],
        total_chunks=result["total_chunks"],
    )


# ------------------ QUERY ------------------

@app.post("/query", response_model=QueryResponse, tags=["Query"])
def query_document(request: QueryRequest):
    """Ask a question against the uploaded document."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer = rag_qa(question=request.question, top_k=request.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return QueryResponse(question=request.question, answer=answer)


# ================================================================
#  ENTRYPOINT
# ================================================================

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
