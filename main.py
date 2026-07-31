"""
Crop Disease RAG Diagnosis -- FastAPI backend

Architecture (mirrors the Azure cloud design, running locally / or deployable to
Azure App Service + Azure AI Search + Azure OpenAI later):

  1. Knowledge base ingestion  -> chunk + embed (Gemini) + store in an in-memory FAISS index
  2. Image diagnosis request   -> Gemini Vision describes visible symptoms from the photo
  3. Retrieval                -> embed the symptom description, search FAISS for top-k chunks
  4. Generation                -> Gemini composes a farmer-readable diagnosis grounded ONLY
                                   in the retrieved chunks (reduces hallucination)

Setup (Windows):
    python -m venv venv
    venv\\Scripts\\activate
    pip install -r requirements.txt

    Create a file named ".env" in this same folder containing exactly one line:
        GEMINI_API_KEY=your-key-here

    uvicorn main:app --reload --port 8000

Then open http://localhost:8000
"""

import os
import io
import mimetypes
import traceback
from typing import List

import numpy as np
import faiss
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
load_dotenv()  # reads GEMINI_API_KEY from a local .env file if present

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is not set. Create a .env file with GEMINI_API_KEY=your-key "
          "or set it as an environment variable before calling /api/diagnose or /api/knowledge-base/upload.")
else:
    print("Gemini API key loaded successfully.")

# Model IDs -- kept current as of the Gemini API (July 2026).
# IMPORTANT: gemini-2.0-flash was shut down by Google on 1 June 2026 -- that was
# the root cause of the original "Diagnosis failed: Internal Server Error".
# If Google retires gemini-2.5-flash in the future (currently scheduled for
# October 2026), swap these two lines for whatever the docs list as GA at
# https://ai.google.dev/gemini-api/docs/models/gemini
EMBED_MODEL = "gemini-embedding-2"    # multimodal embedding model, no "models/" prefix needed
VISION_MODEL = "gemini-2.5-flash"     # multimodal: image -> text
GEN_MODEL = "gemini-2.5-flash"        # text -> grounded diagnosis

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
SAMPLE_KB_DIR = os.path.join(os.path.dirname(__file__), "sample_kb")

app = FastAPI(title="Crop Disease RAG Diagnosis API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Global safety net -- guarantees the frontend always gets a readable JSON
# error instead of a bare "Internal Server Error" popup with no detail.
# --------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()  # full traceback still prints in your terminal
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )


# --------------------------------------------------------------------------
# In-memory vector knowledge base (FAISS) -- swap for Azure AI Search later
# --------------------------------------------------------------------------
class KnowledgeBase:
    def __init__(self):
        self.index = None
        self.chunks: List[dict] = []

    def add(self, text_chunks: List[str], source_name: str) -> int:
        if not text_chunks:
            return 0
        vectors = [embed_text(c, task_type="retrieval_document") for c in text_chunks]
        arr = np.array(vectors, dtype="float32")
        faiss.normalize_L2(arr)

        if self.index is None:
            self.index = faiss.IndexFlatIP(arr.shape[1])
        self.index.add(arr)

        for c in text_chunks:
            self.chunks.append({"text": c, "source": source_name})
        return len(text_chunks)

    def search(self, query: str, k: int = TOP_K) -> List[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q_vec = np.array([embed_text(query, task_type="retrieval_query")], dtype="float32")
        faiss.normalize_L2(q_vec)
        scores, idxs = self.index.search(q_vec, min(k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            item = self.chunks[idx]
            results.append({"text": item["text"], "source": item["source"], "score": round(float(score), 4)})
        return results

    def stats(self):
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(set(c["source"] for c in self.chunks)),
        }


kb = KnowledgeBase()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def embed_text(text: str, task_type: str = "retrieval_document") -> List[float]:
    if not client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on the server.")
    try:
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text[:8000],
            config=types.EmbedContentConfig(task_type=task_type.upper()),
        )
        return result.embeddings[0].values
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Embedding call failed ({EMBED_MODEL}): {e}")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start <= 0:
            break
    return [c.strip() for c in chunks if c.strip()]


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def guess_mime_type(filename: str, provided: str = None) -> str:
    """Best-effort MIME type detection -- browsers (especially on Windows) sometimes
    send an empty or generic content_type, so fall back to the file extension."""
    if provided and provided != "application/octet-stream":
        return provided
    guessed, _ = mimetypes.guess_type(filename or "")
    return guessed or "image/jpeg"


def load_sample_knowledge_base():
    """Auto-index the bundled sample_kb/*.txt files on startup so the demo works out of the box."""
    if not os.path.isdir(SAMPLE_KB_DIR):
        print(f"sample_kb folder not found at {SAMPLE_KB_DIR} -- skipping auto-index.")
        return
    for fname in sorted(os.listdir(SAMPLE_KB_DIR)):
        if not fname.lower().endswith(".txt"):
            continue
        path = os.path.join(SAMPLE_KB_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            added = kb.add(chunk_text(text), source_name=fname)
            print(f"Indexed {added} chunks from {fname}")
        except Exception as e:
            print(f"Could not index {fname}: {e}")


@app.on_event("startup")
def startup_event():
    if client:
        load_sample_knowledge_base()
    else:
        print("Skipping sample knowledge base indexing -- GEMINI_API_KEY not set.")


# --------------------------------------------------------------------------
# API routes
# --------------------------------------------------------------------------
@app.post("/api/knowledge-base/upload")
async def upload_knowledge_base(files: List[UploadFile] = File(...)):
    """Upload one or more .txt / .pdf documents to add to the retrieval knowledge base."""
    total_added = 0
    per_file = []
    for f in files:
        content = await f.read()
        if f.filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(content)
        else:
            text = content.decode("utf-8", errors="ignore")
        chunks = chunk_text(text)
        added = kb.add(chunks, source_name=f.filename)
        total_added += added
        per_file.append({"filename": f.filename, "chunks_added": added})
    return {"status": "ok", "files": per_file, "total_chunks_added": total_added, **kb.stats()}


@app.get("/api/knowledge-base/status")
async def knowledge_base_status():
    return kb.stats()


@app.post("/api/diagnose")
async def diagnose(image: UploadFile = File(...)):
    """
    Upload a crop / leaf photo (simulating a single drone-captured frame) and get back:
      - the symptom description Gemini Vision extracted from the image
      - a diagnosis grounded in the retrieved knowledge-base chunks
      - the source chunks actually used, for transparency
    """
    if not client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on the server.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload.")

    mime_type = guess_mime_type(image.filename, image.content_type)

    # ---- Step 1: Vision layer -- describe visible symptoms ----
    vision_prompt = (
        "You are an agricultural plant pathologist reviewing a drone or handheld photo of a crop. "
        "Describe: (1) the crop type if identifiable, (2) visible symptoms such as spots, lesions, "
        "discoloration, wilting, powdery coating, holes, or pest damage, and (3) which plant part is "
        "affected (leaf, stem, fruit). Be specific and concise, 3-5 sentences. Only describe what is "
        "visible -- do not name a disease yet."
    )
    try:
        vision_response = client.models.generate_content(
            model=VISION_MODEL,
            contents=[
                vision_prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Vision model call failed ({VISION_MODEL}): {e}")

    symptom_description = (vision_response.text or "").strip()
    if not symptom_description:
        raise HTTPException(status_code=502, detail="Vision model returned no description. Try a clearer photo.")

    # ---- Step 2: Retrieval layer -- fetch grounding context ----
    retrieved = kb.search(symptom_description, k=TOP_K)
    context_text = "\n\n".join(
        f"[Source: {r['source']} | similarity: {r['score']}]\n{r['text']}" for r in retrieved
    ) or "No knowledge base documents are indexed yet."

    # ---- Step 3: Generation layer -- grounded diagnosis ----
    gen_prompt = f"""You are an agricultural extension advisor helping a farmer understand a drone/photo-based
crop disease detection result.

OBSERVED SYMPTOMS (from image analysis):
{symptom_description}

RETRIEVED REFERENCE MATERIAL (from the agricultural knowledge base):
{context_text}

Using ONLY the observed symptoms and the reference material above, respond with:
1. Most likely disease/condition, with a confidence level (high / medium / low)
2. Brief reasoning that references the source material
3. Recommended treatment or next steps for the farmer
4. An honest caveat if the reference material is insufficient for a confident diagnosis

Keep the response under 200 words, in clear, farmer-friendly language. Do not invent facts that are
not supported by the retrieved material."""

    try:
        gen_response = client.models.generate_content(model=GEN_MODEL, contents=gen_prompt)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Generation model call failed ({GEN_MODEL}): {e}")

    diagnosis = (gen_response.text or "").strip()

    return {
        "symptom_description": symptom_description,
        "diagnosis": diagnosis,
        "sources": retrieved,
    }


# --------------------------------------------------------------------------
# Frontend
# --------------------------------------------------------------------------
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()