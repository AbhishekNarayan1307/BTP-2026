# Five RAG and LLM Models: Usage Guide

This guide describes the implementations currently present in this checkout. The baseline and normal RAG modes are both in `src/rag_pipeline.py`. The augmentation-optimized implementation has its own interactive entry point. The retrieval-optimized and generation-optimized files are currently empty, so they cannot yet accept questions or return answers.

## 1. Project Overview

| Model | What it does | Current status |
|---|---|---|
| Baseline LLM | Sends the question directly to Groq without document retrieval. | Interactive mode in `src/rag_pipeline.py`. |
| Normal RAG | Retrieves ChromaDB passages, then asks Groq to answer from those passages. | Interactive mode in `src/rag_pipeline.py`; the current query helper has an indentation/syntax error and its configured database directory is absent in this checkout. |
| Retrieval-Optimized RAG | Intended to use a retrieval-optimized implementation. | `RAG_pipeline_optimization/retrieval/retrieve.py` is empty; no callable or runnable model is implemented. |
| Augmentation-Optimized RAG | Expands and reformulates queries, optionally uses HyDE retrieval, filters and deduplicates results, compresses repeated context, then asks Groq to answer from retrieved passages. | Interactive entry point in `RAG_pipeline_optimization/augment.py`. |
| Generation-Optimized RAG | Intended to use a generation-optimized implementation. | `RAG_pipeline_optimization/Generation/generate.py` is empty; no callable or runnable model is implemented. |

## 2. Prerequisites

### Python and VS Code

The project README specifies **Python 3.12.7**. From VS Code, choose **File > Open Folder...** and open the repository root (the folder containing `README.md`, `requirements.txt`, `src`, and `RAG_pipeline_optimization`).

To open PowerShell at the project root, use **Terminal > New Terminal** in VS Code and check the current directory:

```powershell
Get-Location
```

It should end in the repository folder, `BTP-2026`. If needed, change to it (adjust the parent path to match where you saved the project):

```powershell
Set-Location "C:\path\to\BTP-2026"
```

### Virtual environment and dependencies

No `venv` or `.venv` directory is present in the inspected checkout. Create one if you do not already have one elsewhere, then activate it in PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation because of the execution policy, use the current-process-only setting and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

The repository requirements include the Python packages used by the project, including `groq`, `python-dotenv`, `chromadb`, `sentence-transformers`, `transformers`, and `torch`. The local Qwen helper also downloads `Qwen/Qwen2.5-0.5B-Instruct` from Hugging Face the first time it is loaded; it does not implement a question-answering command-line interface by itself.

### API key

The baseline and both RAG answer-generation implementations use Groq model `openai/gpt-oss-120b`. They require `GROQ_API_KEY`. Create a `.env` file in the project root (or set the variable in the PowerShell session) and insert your own key; do not put a real key in this guide or commit it to source control.

`.env` contents:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Alternatively, set it for the current PowerShell session:

```powershell
$env:GROQ_API_KEY = "your_groq_api_key_here"
```

The two empty optimization files have no implementation to run and therefore have no working API-key or dependency requirements of their own yet.

## 3. Running the Five Models Individually

Run commands from the project root with the virtual environment activated. Each model below is described independently; running one does not require running another model first.

### 3.1 Baseline LLM Model

**Purpose:** Answers directly with Groq and does not retrieve documents.

- **Python file:** `src/rag_pipeline.py`
- **Entry point:** `main()`; direct answer function: `answer_question_direct(question)`.
- **Terminal input:** Yes. The program menu offers direct LLM mode.
- **Required key:** `GROQ_API_KEY` in `.env` or the environment.
- **Database/domain:** None for direct mode. No ECE or Legal selection is needed.
- **Command from the project root:**

```powershell
python -m src.rag_pipeline
```

At the first menu, enter `2` for **Without retrieved documents (Direct LLM)**, then type your question. For example: `What is the purpose of a retrieval-augmented generation system?` The answer is printed under `ANSWER`.

The file initializes its Groq client at import time, so the key and the `groq` and `python-dotenv` packages are needed even before the menu appears.

### 3.2 Normal RAG Model

**Purpose:** Retrieves passages from ChromaDB, then generates an evidence-based answer with Groq.

- **Python file:** `src/rag_pipeline.py`
- **Entry point:** `main()`; question-answer function: `answer_question_with_rag(question, collection, top_k=5)`.
- **Terminal input:** Yes. The program asks for RAG mode, a domain, and a question.
- **Required key:** `GROQ_API_KEY`.
- **Domain selection:** Choose `1` for Legal or `2` for Electronics (ECE).
- **Database configuration in `data_preprocessing/query.py`:** `data_preprocessing/chroma_db`, with `all-MiniLM-L6-v2`; collection names `legal_data_v3` and `ece_data`.
- **Command from the project root:**

```powershell
python -m src.rag_pipeline
```

Enter `1` for **With retrieved documents (RAG)**, choose Legal or Electronics, and type your question. Example: `What does the evidence say about the selected topic?` The answer and retrieved source information are printed.

**Current checkout limitation:** `data_preprocessing/query.py` contains a malformed indentation in `retrieve_documents`, which prevents Python from importing it. Also, the configured `data_preprocessing/chroma_db` directory was not present when this guide was prepared. The checkout does contain `data_preprocessing/indian_laws_chromadb`, but the normal RAG query code does not point to that database. Consequently, the command is the implementation's actual interactive command, but normal RAG will need those existing setup issues resolved before it can retrieve successfully. This guide does not change the implementation.

### 3.3 Retrieval-Optimized RAG Model

**Purpose:** A retrieval-optimized variant is named in the project structure, but its implementation is not present.

- **Python file:** `RAG_pipeline_optimization/retrieval/retrieve.py`
- **Main entry point:** None; the inspected file is empty.
- **Question-answer function:** None implemented.
- **Terminal input:** No.
- **Database, collections, API key, and domain selection:** Not configured in this file.
- **Execution command:** None. There is no valid command to accept a question or return an answer from the current file.

The file can be opened in an editor for inspection, but it cannot be run as the retrieval-optimized question-answering model until an implementation is added.

### 3.4 Augmentation-Optimized RAG Model

**Purpose:** Augments retrieval queries, retrieves for each query, optionally uses HyDE text as a retrieval query, deduplicates and ranks real passages, removes repeated sentences, and generates an answer using the original question and retrieved passages.

- **Python file:** `RAG_pipeline_optimization/augment.py`
- **Entry point:** `main()`; question-answer function: `answer_question_with_rag(question, collection, top_k=None, *, config=DEFAULT_CONFIG, generator=...)`.
- **Terminal input:** Yes. The script lists collections it can open, then asks for a dataset choice and a question.
- **Required key:** `GROQ_API_KEY` for query generation, HyDE generation, and final answer generation.
- **Feature settings:** `AugmentationConfig` controls `query_expansion`, `multi_query`, `hyde`, `context_filtering`, and `context_compression` independently. It also controls `generated_query_count`, `retrieval_top_k`, `final_context_passages`, and `max_distance`.
- **Database configuration:** First checks `data_preprocessing/chroma_db` for `legal_data_v3` and `ece_data` using `all-MiniLM-L6-v2`. If the Legal collection is unavailable there, it falls back to `data_preprocessing/indian_laws_chromadb`, collection `indian_statutory_acts`, using `BAAI/bge-base-en-v1.5`. The current checkout contains the latter Legal database; no ECE database was present at the inspected project paths.
- **Command from the project root:**

```powershell
python -m RAG_pipeline_optimization.augment
```

Choose an available dataset from the displayed list, then enter your question. For example, for Legal: `What penalty does the Act provide for failing to take reasonable security safeguards for personal data?`

The script only lists a dataset when it can open its configured database and collection. Opening a collection requires `chromadb` and `sentence-transformers` to be installed and the matching embedding model to be available. The Legal fallback uses the BGE embedding model; the primary collections use MiniLM. HyDE output is used only to search the database and is not passed as factual answer evidence.

### 3.5 Generation-Optimized RAG Model

**Purpose:** A generation-optimized variant is named in the project structure, but its implementation is not present.

- **Python file:** `RAG_pipeline_optimization/Generation/generate.py`
- **Main entry point:** None; the inspected file is empty.
- **Question-answer function:** None implemented.
- **Terminal input:** No.
- **Database, collections, API key, and domain selection:** Not configured in this file.
- **Execution command:** None. There is no valid command to accept a question or return an answer from the current file.

The file can be opened in an editor for inspection, but it cannot be run as the generation-optimized question-answering model until an implementation is added.

## 4. Model Comparison

| Model | Python File | Execution Command | Retrieval Used |
|---|---|---|---|
| Baseline LLM | `src/rag_pipeline.py` | `python -m src.rag_pipeline` (choose mode `2`) | No |
| Normal RAG | `src/rag_pipeline.py` | `python -m src.rag_pipeline` (choose mode `1`, then domain) | Yes; current query helper has a syntax issue and configured DB path is absent |
| Retrieval-Optimized RAG | `RAG_pipeline_optimization/retrieval/retrieve.py` (empty) | No command; no entry point | Not implemented |
| Augmentation-Optimized RAG | `RAG_pipeline_optimization/augment.py` | `python -m RAG_pipeline_optimization.augment` | Yes |
| Generation-Optimized RAG | `RAG_pipeline_optimization/Generation/generate.py` (empty) | No command; no entry point | Not implemented |

## 5. Troubleshooting

### Check your project directory

```powershell
Get-Location
Test-Path .\README.md
Test-Path .\src\rag_pipeline.py
```

Both `Test-Path` commands should return `True` when PowerShell is at the project root.

### Check Python and the active environment

```powershell
python --version
Get-Command python
```

The project documents Python 3.12.7. If the intended virtual environment exists, activate it with `.\venv\Scripts\Activate.ps1` and check which interpreter is selected with `Get-Command python`.

### Install missing packages

```powershell
python -m pip install -r requirements.txt
python -m pip show groq python-dotenv chromadb sentence-transformers
```

If imports still fail, verify that the same `python` command is being used to install packages and run the model.

### Check the API key without printing its value

```powershell
if ($env:GROQ_API_KEY) { "GROQ_API_KEY is set" } else { "GROQ_API_KEY is missing in this session" }
```

If the key is in the root `.env`, run the program from the project root so `python-dotenv` can find it. A missing or invalid key prevents Groq calls.

### Check ChromaDB paths and the normal RAG helper

```powershell
Test-Path .\data_preprocessing\chroma_db
Test-Path .\data_preprocessing\indian_laws_chromadb
python -m py_compile .\data_preprocessing\query.py
```

In this checkout, the first path was absent and the query helper failed Python's syntax check because of indentation. The second path exists as the Legal fallback used by the augmentation model; it is not the path used by normal RAG. Fixing either issue would require changing Python code or database setup, which is outside this guide.

### Import errors or no execution entry point

Use the commands below to inspect the relevant files and search for interactive entry points:

```powershell
Get-Content .\RAG_pipeline_optimization\retrieval\retrieve.py
Get-Content .\RAG_pipeline_optimization\Generation\generate.py
Select-String -Path .\src\rag_pipeline.py, .\RAG_pipeline_optimization\augment.py -Pattern 'if __name__|def main|def answer_question'
```

The retrieval and generation optimization files are empty in this checkout. Installing packages cannot add their missing `main()` functions or question-answer logic.

## 6. Important Notes

- Each implemented model has its own callable pipeline; the baseline and normal RAG are selectable modes of `src/rag_pipeline.py`, while augmentation RAG has a separate module and command.
- Running one model does not require editing another model's implementation.
- The retrieval-optimized and generation-optimized question-answering models are not currently runnable because their named Python files are empty. No execution commands or question-answer functions are claimed for them.
- Normal RAG currently cannot import its query helper as written, and its configured Chroma directory was not present in the inspected checkout. The augmentation script uses a separate Legal fallback when available.
- The augmentation module was present at the exact path `RAG_pipeline_optimization/augment.py`; another similarly named file exists under the misspelled `RAG_pipeline_optimization/Augmenation/` directory. This guide's command and description refer to the exact path requested above.
