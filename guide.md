# BTP-2026: Five-Model Comparison Guide

## Running the Persistent Web Application

The web application runs the same five conditions as the unified terminal runner. It initializes the shared resources once when the backend starts, then reuses them for each submitted question and for both domains.

### Prerequisites

- PowerShell opened at the project root (`BTP-2026`).
- The existing `venv` virtual environment. The repository's documented baseline is Python 3.12.7; the current local `venv` was verified with Python 3.14.0.
- The project dependencies in the root `requirements.txt`, including FastAPI, Uvicorn, Jinja2, ChromaDB, Sentence Transformers, `rank-bm25`, Groq, and the LangChain packages used by Model 5.
- A Groq API key available as `GROQ_API_KEY` in the root `.env` file.
- The existing ChromaDB database and both collections described below.

Use the root `requirements.txt` for the web application. `requirements-lock.txt` is an existing pinned package list, but it does not include all of the web/runtime direct dependencies, including FastAPI, Groq, and `rank-bm25`.

### First-time setup

Open PowerShell at the project root and activate the existing environment:

```powershell
cd "C:\Users\Abhishek Narayan\Desktop\BTP 2026\BTP-2026"
.\venv\Scripts\Activate.ps1
python --version
```

Install the project dependencies (or install any missing packages in an already provisioned environment):

```powershell
python -m pip install -r requirements.txt
```

Set the Groq credential in the root `.env` file. Keep the real key private:

```text
GROQ_API_KEY=your_groq_api_key_here
```

At backend startup, Sentence Transformers initializes `BAAI/bge-base-en-v1.5`, and the retrieval optimizer initializes `cross-encoder/ms-marco-MiniLM-L-6-v2`. If these models are not already cached, their first use downloads them from Hugging Face and can take longer. The first start also opens both Chroma collections and builds one BM25 index for each domain.

### Start the backend and open the website

From the project root with `venv` active, run the verified command:

```powershell
.\venv\Scripts\python.exe -m uvicorn web_app.main:app --host 127.0.0.1 --port 8000
```

Keep this PowerShell window open while using the app. Open this URL in a browser:

```text
http://127.0.0.1:8000
```

The backend also exposes `GET /api/health` and `POST /api/ask`.

### Use the web interface

1. Choose **Legal** or **ECE**.
2. Enter one question and select **Run all five models**.
3. Wait for the five results. Each panel shows its answer or error, execution time, and available retrieval/verification details. Expand **Retrieved sources** to inspect passages and metadata. The evaluation header shows total execution time.
4. Enter another question, or switch domains, and submit again. The backend remains running; do not restart it between questions.

The health indicator reports backend/runtime readiness and the collection counts. For detailed status and initialization counters, run this in another PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health | ConvertTo-Json -Depth 6
```

When ready, the response reports `runtime_status: "ready"`, Legal collection `indian_statutory_acts`, ECE collection `research_papers`, and `initialization_counts`. The embedding function, Chroma client, Groq client, CrossEncoder, both domain retrievers, both BM25 indexes, and the generation pipeline should each show a count of `1`. Check the endpoint again after submitting questions or switching domains; these values should remain unchanged. The server log also reports startup initialization and per-model execution.

Switching between Legal and ECE selects the other collection and its already built retriever/index. Submitting a new question reuses the initialized resources; it does not reload the embedding or reranker models.

Previous evaluations remain visible in the website during the browser tab's session. They are kept for comparison only: each request sends one domain and one question, and the model prompts do not receive earlier questions, answers, or conversation history.

To stop the backend, return to the PowerShell window running Uvicorn and press **Ctrl+C**.

### Troubleshooting

- **The page says the backend is unavailable:** Confirm that Uvicorn is still running in the first PowerShell window and that the address is `http://127.0.0.1:8000`.
- **The runtime is not ready:** Check `GET /api/health` and the Uvicorn startup log. `component_errors` identifies the failed initialization component, such as ChromaDB/embeddings, Groq client, CrossEncoder, or RAG pipelines. The API does not accept questions until initialization succeeds.
- **Missing package or import error:** Activate `venv` and install from the root `requirements.txt`. Installing only from `requirements-lock.txt` may omit the web/API, Groq, or retrieval dependencies.
- **Groq key missing:** Confirm that `.env` is in the project root and contains `GROQ_API_KEY=...`. Restart the backend after changing `.env`.
- **Groq rate limit or network error:** Check the backend log for the affected model's error. Retry after the provider becomes available; one model's request failure is reported separately from the other models.
- **Database or collection unavailable:** Confirm that `data_preprocessing/chromaDB/indian_laws_chromadb` exists and contains `indian_statutory_acts` and `research_papers`. The web runtime opens the existing database; startup does not ingest or rebuild it.
- **First startup takes a while:** The embedding and reranker model files may need to download. Keep the process online until the health endpoint reports `ready`.
- **Port 8000 is already in use:** Stop the other process using that port before starting this backend.

## Five model roles

Some Legal chunks do not contain page metadata. For those results, the interface displays the available source metadata and may show `Unknown` for the page.

| # | Model | Retrieval | Answer generation |
|---|---|---|---|
| 1 | Baseline LLM | None | Direct Groq generation using the question and general-knowledge baseline prompt. |
| 2 | Normal RAG | Standard Chroma query | Normal evidence-grounded Groq generation. |
| 3 | Retrieval-Optimized RAG | Semantic retrieval plus BM25, reciprocal-rank fusion, and CrossEncoder reranking | Normal RAG prompt and Groq generation. |
| 4 | Augmentation-Optimized RAG | Query expansion, multi-query, HyDE retrieval, context filtering, deduplication, and compression | Normal evidence-grounded Groq generation. |
| 5 | Generation-Optimized RAG | Standard Chroma query | Evidence-grounded generation, claim-level verification, heuristic support score, and abstention. |

All answer-generating adapters use Groq model `openai/gpt-oss-120b`. The five conditions retain separate prompts and generation/retrieval methods. The runner does not pass reference answers, gold evidence, or annotation labels to any model. Model 5's support score is heuristic, not a calibrated probability.

## Dataset and ChromaDB mapping

The existing persistent database is:

```text
data_preprocessing/chromaDB/indian_laws_chromadb
```

| Selection | Collection | Source data | Embedding model |
|---|---|---|---|
| Legal | `indian_statutory_acts` | Consumer Protection Act 2019, Digital Personal Data Protection Act 2023, and Motor Vehicles Act 1988 chunks; 571 documents in the current local collection | `BAAI/bge-base-en-v1.5` |
| ECE | `research_papers` | `data_preprocessing/all_research_papers_chunks.json`; 118 documents in the current local collection | `BAAI/bge-base-en-v1.5` |

Both ingestion scripts now target this same database path and use the BGE embedding model. Normal, augmentation, and generation RAG use the selected collection. Retrieval-optimized RAG opens only that collection and uses the matching Legal JSON sources or ECE research-paper JSON source for BM25. The runner does not combine domain corpora or ingest data during startup.

## Unified terminal application

From the repository root, activate the Windows PowerShell environment and run:

```powershell
.\venv\Scripts\Activate.ps1
python run_all_models.py
```

The application asks for one dataset (`1` Legal or `2` ECE) and one question. It passes the same question to all five models and runs them sequentially. Every model has its own result and status; a failure in one adapter does not stop the others. The final table summarizes status and elapsed time.

For this one-shot CLI, per-model elapsed time includes adapter imports and setup, retrieval, and generation. Shared dataset/collection initialization and user input are excluded. The persistent web application's per-model times exclude startup initialization and include that request's adapter work, retrieval, and generation.

## Individual commands

Run these from the repository root with the virtual environment active:

| Model | Command | Notes |
|---|---|---|
| Baseline LLM | `python -m src.rag_pipeline` | Select `2` for direct Groq mode. |
| Normal RAG | `python -m src.rag_pipeline` | Select `1`, then Legal or Electronics; Electronics opens `research_papers`. |
| Retrieval-Optimized | `python -m RAG_pipeline_optimization.retrieval.test_retrieve` | Interactive retrieval-only tool. The unified runner adds the normal answer-generation condition. |
| Augmentation-Optimized | `python -m RAG_pipeline_optimization.augmentation.augment` | Interactive RAG using the selected collection. |
| Generation-Optimized | `python -m RAG_pipeline_optimization.Generation.generate` | Select RAG and a domain; streaming is optional in standalone mode. |

The standalone baseline in `LLM/local_model.py` is a separate fixed-question Qwen demo; it is not Model 1 in the five-condition comparison.

## Current project structure

```text
BTP-2026/
|-- run_all_models.py
|-- web_app/
|   |-- main.py
|   |-- runtime.py
|   |-- templates/index.html
|   `-- static/
|       |-- app.js
|       `-- style.css
|-- src/rag_pipeline.py
|-- data_preprocessing/
|   |-- query.py
|   `-- chromaDB/indian_laws_chromadb/
`-- RAG_pipeline_optimization/
    |-- augmentation/augment.py
    |-- retrieval/retrieve.py
    `-- Generation/generate.py
```
