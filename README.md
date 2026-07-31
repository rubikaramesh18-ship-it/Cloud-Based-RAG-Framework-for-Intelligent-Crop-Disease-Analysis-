<<<<<<< HEAD
# CropDiagnosis RAG
A minimal, two-layer FastAPI + FAISS + Gemini RAG app that diagnoses crop diseases from a
photo (simulating a single drone-captured frame) and grounds the diagnosis in a retrieval-
augmented knowledge base of agricultural reference material -- the working prototype behind
the "Cloud-Based RAG Framework for Intelligent Crop Disease Diagnosis using Drone Imagery
and Agricultural Knowledge Bases" project.

Built the same way DocMind was: one FastAPI backend file, one plain HTML/JS frontend file,
an in-memory FAISS index instead of a managed vector DB, and direct Gemini API calls for
embeddings, vision, and generation. No LangChain, no separate frontend framework -- easy to
reason about, easy to demo, easy to explain in an interview.

## How it works

1. **Ingestion** -- `.txt` / `.pdf` documents are chunked and embedded with Gemini's
   `text-embedding-004` model, then added to an in-memory FAISS `IndexFlatIP` (cosine
   similarity). Five sample disease-reference documents are bundled in `sample_kb/` and
   auto-indexed on startup so the demo works immediately.
2. **Vision** -- an uploaded crop/leaf photo is sent to Gemini's multimodal model, which
   describes the visible symptoms (spots, discoloration, wilting, pest damage, etc.)
   without naming a disease yet.
3. **Retrieval** -- the symptom description is embedded and used to search the FAISS index
   for the top-5 most relevant knowledge-base chunks.
4. **Generation** -- Gemini composes a farmer-readable diagnosis using ONLY the observed
   symptoms and the retrieved chunks, reducing hallucination and making every claim
   traceable back to a source.

This mirrors the Azure architecture from the Phase-I report: Blob Storage -> Azure ML ->
Azure AI Search -> Azure OpenAI -> App Service. Swapping FAISS for Azure AI Search and
adding Azure Blob Storage / Functions is the natural next step for a cloud deployment.

## Project structure

```
cropdiagnosis_rag/
├── main.py              FastAPI backend (ingestion, vision, retrieval, generation)
├── requirements.txt
├── static/
│   └── index.html        Single-file frontend (upload KB docs, upload image, view result)
└── sample_kb/             Bundled reference docs, auto-indexed on startup
    ├── late_blight.txt
    ├── powdery_mildew.txt
    ├── bacterial_leaf_blight.txt
    ├── leaf_rust_wheat.txt
    └── fall_armyworm_maize.txt
```

## Setup

```bash
cd cropdiagnosis_rag
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

export GEMINI_API_KEY="your-gemini-api-key"   # Windows: set GEMINI_API_KEY=...
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

Get a free Gemini API key at https://aistudio.google.com/apikey if you don't already have
the one used for DocMind.

## Using the app

1. On first launch, the five sample knowledge-base documents are indexed automatically
   (visible in the "Knowledge Base" card badge in the UI).
2. Optionally upload more `.pdf` / `.txt` reference material (e.g. the 15 papers from your
   literature survey, converted to plain text or kept as PDFs) to expand the knowledge base.
3. Upload a crop/leaf photo under "Diagnose a Crop Image" and click **Diagnose**.
4. The result panel shows the observed symptoms, the grounded diagnosis with a confidence
   level and recommended treatment, and the exact source chunks that were retrieved.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/knowledge-base/upload` | Upload one or more `.pdf`/`.txt` files to index |
| `GET`  | `/api/knowledge-base/status` | Return chunk/document counts |
| `POST` | `/api/diagnose` | Upload an image, get symptoms + grounded diagnosis + sources |
| `GET`  | `/` | Serves the frontend |

## Known limitations (good talking points for your project review)

- The FAISS index is **in-memory only** -- it resets on server restart. For production this
  maps directly to Azure AI Search (persistent, managed, scalable), as described in the
  Phase-I architecture diagram.
- Disease detection relies on Gemini Vision's general image understanding rather than a
  custom-trained CNN/ViT fine-tuned on drone-altitude crop imagery -- a reasonable
  prototype trade-off, and a clear "future work" item (fine-tune on Agriculture-Vision /
  PlantVillage per the Phase-I dataset plan).
- No authentication, rate limiting, or persistent result storage yet -- Phase-II can add
  Cosmos DB storage and Entra ID auth as planned in the architecture.
=======
# Cloud-Based-RAG-Framework-for-Intelligent-Crop-Disease-Analysis-
Developed an AI-powered crop disease diagnosis system using **FastAPI, Gemini AI, and RAG** to detect plant diseases from images and provide accurate treatment recommendations using a knowledge base.
>>>>>>> 5df35c886226bf0b898925daa64b9ceb3caceb76
