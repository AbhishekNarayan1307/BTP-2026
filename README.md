# BTP-2026
baseline LLM vs RAG performance evaluation study
data/questions.csv: Question, reference, required claims, category, answerability, split.
data/gold_evidence.jsonl: Question ID, evidence IDs/text/source, claim-to-evidence mapping, verifier.
data/documents/: Source documents used by the pipeline, subject to permissions.
docs/annotation_guidelines.md: Label definitions, examples, disagreement/adjudication rules.
docs/dataset_card.md: Scope, sources, construction, limitations, licensing/privacy.

## Phase 4: Baseline LLM Implementation

### Objective
Implement a no-retrieval baseline LLM using Hugging Face API.

### Experimental Condition
E0 - Baseline LLM without corpus evidence.

### Model Provider
Hugging Face API

### Project Structure
- Data: Dataset and gold evidence
- src: Python implementation
- prompts: Baseline prompts
- outputs: Raw experiment outputs
- configs: Experiment configurations

## Development Environment

### Python Version
Python 3.12.7

### Virtual Environment
venv

### Dependencies
- requirements.txt: Direct project dependencies
- requirements-lock.txt: Complete pinned package versions

### Environment Setup
Create the virtual environment:

    /opt/anaconda3/bin/python3 -m venv venv

Activate it:

    source venv/bin/activate

Install dependencies:

    pip install -r requirements-lock.txt
