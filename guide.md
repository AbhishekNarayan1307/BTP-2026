# BTP-2026: Five-Model Comparison Guide

This project compares five answering configurations over the existing Legal and ECE ChromaDB collections. All answer generation uses a local Ollama model.

## Windows setup

Use PowerShell from the repository root and the working `.venv-win64` virtual environment with Python 3.14.

```powershell
.\.venv-win64\Scripts\Activate.ps1
python --version
python -m pip install -r requirements.txt
```

The root `requirements.txt` includes the OpenAI SDK, ChromaDB, Sentence Transformers, `rank-bm25`, FastAPI, Uvicorn, and the other project dependencies. Ollama requires no provider credential. `.env.example` contains only a note about the local Ollama endpoint; the application does not require `.env` values for Ollama.

### ChromaDB and Python architecture

Run pip through the activated environment using `python -m pip`, so the packages install into the same Python that runs the project. If Python reports `No module named 'chromadb'`, confirm the active interpreter and pip location:

```powershell
python -c "import platform, sys; print(sys.version); print(platform.machine()); print(sys.executable)"
python -m pip --version
```

Then install the project requirements again:

```powershell
python -m pip install -r requirements.txt
```

If ChromaDB reports an architecture or wheel error, confirm the activated interpreter is the project's working Windows 64-bit `.venv-win64` Python 3.14 environment. Avoid mixing packages from a different global Python installation or another virtual environment.

## Install and run Ollama

1. Install Ollama for Windows from [ollama.com/download](https://ollama.com/download).
2. In PowerShell, download the 3B model:

   ```powershell
   ollama pull llama3.2:3b
   ```

3. Verify that it is installed:

   ```powershell
   ollama list
   ```

   Confirm the list includes `llama3.2:3b`. Ollama normally runs in the background after installation. If it is not running, start it in a separate PowerShell window with `ollama serve` and leave that window open. The app connects to Ollama at `http://localhost:11434/v1` using the OpenAI-compatible SDK and model name `llama3.2`.

Inference runs locally and does not incur API usage charges. The first pull downloads the model to the computer.

## Run all five configurations in the CLI

From the repository root, with `.venv-win64` activated and Ollama running with `llama3.2:3b` available:

```powershell
python run_all_models.py
```

Select `1` for Legal or `2` for ECE, then enter one question. The runner passes that question to all five configurations in sequence and reports a separate status, answer, and elapsed time for each. A failure in one configuration does not stop the other adapters.

## Run the website

The root requirements install Uvicorn. If it is missing, install it in the active environment:

```powershell
python -m pip install uvicorn
```

From the repository root with `.venv-win64` activated and Ollama running, start the FastAPI app:

```powershell
python -m uvicorn web_app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open the homepage at [http://127.0.0.1:8000](http://127.0.0.1:8000). FastAPI's interactive API documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). The app also provides `GET /api/health` and `POST /api/ask`.

The backend initializes shared resources when the application starts:

- The Ollama client and availability of `llama3.2`.
- The BGE embedding function `BAAI/bge-base-en-v1.5` and the existing Legal (`indian_statutory_acts`) and ECE (`research_papers`) ChromaDB collections.
- The CrossEncoder reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`, the Legal and ECE hybrid retrievers and BM25 indexes, and the generation prompts.

These resources are reused across consecutive questions and domain selections while the server remains running. The first startup can download the embedding and reranker models from Hugging Face, so internet access is needed until those model files are cached. The existing database is opened; startup does not ingest or rebuild datasets.

To stop the backend, focus its PowerShell window and press **Ctrl+C**.

The website works locally because the browser and backend can both reach Ollama through `localhost`. A cloud-hosted backend's `localhost` refers to the cloud machine, so it cannot reach Ollama running on a personal computer through that address. A deployed backend would need Ollama hosted at a network endpoint it can access.

## Five model roles

| # | Configuration | Retrieval and generation |
|---|---|---|
| 1 | Baseline | No retrieval; uses the existing general-knowledge prompt. |
| 2 | Normal RAG | Standard Chroma retrieval and the existing evidence-grounded prompt. |
| 3 | Retrieval-Optimized RAG | Semantic retrieval plus BM25, reciprocal-rank fusion, and CrossEncoder reranking; uses the Normal RAG prompt. |
| 4 | Augmentation-Optimized RAG | Query expansion, multi-query, HyDE retrieval, context filtering, deduplication, and compression. |
| 5 | Generation-Optimized RAG | Standard Chroma retrieval, evidence-grounded generation, claim-level verification, heuristic support score, and abstention. |

All five configurations use the local `llama3.2` model. Their separate prompts, retrieval methods, response parsing, and evaluation logic are preserved. The runner does not pass reference answers, gold evidence, or annotation labels to any model. Model 5's support score is heuristic, not a calibrated probability.

## Data and ChromaDB

The existing persistent database is `data_preprocessing/chromaDB/indian_laws_chromadb`.

| Selection | Collection | Source data | Embedding model |
|---|---|---|---|
| Legal | `indian_statutory_acts` | Consumer Protection Act 2019, Digital Personal Data Protection Act 2023, and Motor Vehicles Act 1988 chunks | `BAAI/bge-base-en-v1.5` |
| ECE | `research_papers` | `data_preprocessing/all_research_papers_chunks.json` | `BAAI/bge-base-en-v1.5` |

The runner opens the selected existing collection. It does not combine the two domain corpora or ingest data during startup.

## Troubleshooting

- **Ollama is unavailable or the connection is refused:** Confirm Ollama is installed and running. If needed, start `ollama serve` in another PowerShell window. You can check the model list with `ollama list`.
- **The model is missing:** Run `ollama pull llama3.2:3b`, confirm it appears in `ollama list`, then restart the CLI or website backend.
- **Missing Python package:** Activate `.venv-win64`, then run `python -m pip install -r requirements.txt`. Check `python -m pip --version` and `python -c "import sys; print(sys.executable)"` to confirm pip and the app use the same environment.
- **ChromaDB installation or architecture error:** Confirm that `.venv-win64` is the working Windows 64-bit Python 3.14 environment, activate it, and reinstall with `python -m pip install -r requirements.txt`. Do not run pip from a different interpreter.
- **The website or API does not open:** Confirm the Uvicorn process is still running and use `http://127.0.0.1:8000`. Check `http://127.0.0.1:8000/api/health` for startup errors and the server console for details.
- **Port 8000 is already occupied:** Stop the other process using the port, or choose another free port in the Uvicorn command and open the matching local URL.
- **Hugging Face model download fails:** Check internet access and retry startup. The BGE embedding and CrossEncoder reranker must be available locally; once cached, startup does not need to download them again.
- **A collection cannot be opened:** Confirm `data_preprocessing/chromaDB/indian_laws_chromadb` exists and contains `indian_statutory_acts` and `research_papers`. The app does not rebuild a missing database.
