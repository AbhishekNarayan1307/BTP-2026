"""Augmentation-optimized RAG pipeline built on the project Chroma/Ollama stack.

Run from the project root with::

    python -m RAG_pipeline_optimization.augmentation.augment

The five retrieval/context techniques are independently controlled by
AugmentationConfig. Chroma remains responsible for embedding query_texts with
the same embedding function configured on each collection. HyDE text is only
used as a retrieval query and is never included in answer evidence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.ollama_client import OLLAMA_MODEL, create_chat_completion

INSUFFICIENT_EVIDENCE = "The provided evidence is insufficient to answer this question."
MODEL_NAME = OLLAMA_MODEL
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AugmentationConfig:
    """Feature switches and retrieval/context limits for experiments."""

    query_expansion: bool = True
    multi_query: bool = True
    hyde: bool = True
    context_filtering: bool = True
    context_compression: bool = True
    generated_query_count: int = 3
    retrieval_top_k: int = 5
    final_context_passages: int = 5
    # Chroma cosine distance is generally in [0, 2]. Set None to disable.
    max_distance: float | None = 0.95


DEFAULT_CONFIG = AugmentationConfig()


def generate_ollama_response(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.2,
    max_completion_tokens: int = 2048,
) -> str:
    """Call the same local Ollama model used by the other configurations."""
    response = create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_completion_tokens,
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""


def _parse_query_list(raw: str, original_question: str, count: int) -> list[str]:
    """Parse model output defensively; reject empty, duplicate, or broad items."""
    candidates: list[str] = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed = parsed.get("queries", parsed.get("expansions", []))
        if isinstance(parsed, list):
            candidates = [str(item) for item in parsed]
    except (json.JSONDecodeError, TypeError):
        candidates = re.findall(r"(?:^|\n)\s*(?:[-*\d.)]+\s*)?(.+?)\s*$", raw)

    original_key = _normalize(original_question)
    seen = {original_key}
    valid: list[str] = []
    for item in candidates:
        item = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", item).strip().strip('"\' ')
        key = _normalize(item)
        if not item or key in seen or len(item) < 8 or len(item) > 400:
            continue
        if len(item.split()) > max(8, len(original_question.split()) * 5):
            continue
        seen.add(key)
        valid.append(item)
        if len(valid) >= max(1, count):
            break
    return valid


def generate_query_variants(
    question: str,
    *,
    expansion_count: int = 2,
    multi_query_count: int = 3,
    query_expansion: bool = True,
    multi_query: bool = True,
    generator=generate_ollama_response,
) -> tuple[list[str], list[str]]:
    """Generate expansion terms and/or diverse question formulations in one call."""
    if not query_expansion and not multi_query:
        return [], []

    requested = max(1, expansion_count if query_expansion else 0) + max(
        1, multi_query_count if multi_query else 0
    )
    request = {
        "expansions": max(0, expansion_count) if query_expansion else 0,
        "alternative_questions": max(0, multi_query_count) if multi_query else 0,
    }
    prompt = f"""Create concise retrieval queries for this question. Keep them tightly scoped to its subject.
Return only valid JSON with two arrays named \"expansions\" and \"alternative_questions\".
An expansion should preserve the original question and add likely synonyms, related terms,
technical phrases, or keywords. Alternative questions should express distinct useful search
angles while preserving the same intent. Do not answer the question or add assumptions.
Requested counts: {json.dumps(request)}
Question: {question}"""
    try:
        raw = generator(
            "You create focused search queries. Return only the requested JSON.",
            prompt,
            temperature=0.2,
            max_completion_tokens=min(700, max(200, requested * 100)),
        )
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        expansions = _parse_query_list(
            json.dumps(payload.get("expansions", [])), question, expansion_count
        ) if query_expansion else []
        alternatives = _parse_query_list(
            json.dumps(payload.get("alternative_questions", [])), question, multi_query_count
        ) if multi_query else []
        return expansions, alternatives
    except Exception as exc:
        # Search still proceeds with the original question when expansion fails.
        print(f"Query generation failed; using the original query ({exc}).")
        return [], []


def generate_hypothetical_document(
    question: str, *, generator=generate_ollama_response
) -> str:
    """Generate a hypothetical retrieval aid; callers must not use it as evidence."""
    try:
        return generator(
            "Draft a brief hypothetical passage that could contain the answer. "
            "Do not claim it is verified or cite invented sources.",
            f"Write a short, topic-specific hypothetical passage useful for retrieving real evidence.\n"
            f"Question: {question}",
            temperature=0.2,
            max_completion_tokens=350,
        ).strip()
    except Exception as exc:
        print(f"HyDE generation failed; continuing without it ({exc}).")
        return ""


def _metadata_fields(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Map common project metadata names while preserving every original field."""
    result = dict(metadata or {})
    result.setdefault(
        "source",
        result.get("document", result.get("act", result.get("paper_title", "Unknown"))),
    )
    result.setdefault("page", result.get("page_start", result.get("pages", "Unknown")))
    result.setdefault("page_end", result.get("page_end", result.get("page", "Unknown")))
    result.setdefault("section", result.get("section", "Unknown"))
    result.setdefault("chunk_index", result.get("part", result.get("chunk", "Unknown")))
    return result


def retrieve_documents(
    collection: Any,
    query: str,
    top_k: int = 5,
    *,
    query_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Query a compatible Chroma collection and return normalized source records.

    For ordinary/expanded questions Chroma embeds query_texts with the collection's
    configured embedding function. HyDE uses query_texts too, so the hypothetical
    text is embedded by that same collection function and never enters evidence.
    """
    if not query or not query.strip() or top_k <= 0:
        return []
    count = int(collection.count())
    if count <= 0:
        return []
    kwargs: dict[str, Any] = {
        "n_results": min(int(top_k), count),
        "include": ["documents", "metadatas", "distances"],
    }
    if query_embedding is not None:
        kwargs["query_embeddings"] = [query_embedding]
    else:
        kwargs["query_texts"] = [query]
    results = collection.query(**kwargs)
    documents = (results.get("documents") or [[]])[0] or []
    metadatas = (results.get("metadatas") or [[]])[0] or []
    distances = (results.get("distances") or [[]])[0] or []
    ids = (results.get("ids") or [[]])[0] or []
    found: list[dict[str, Any]] = []
    for index, text in enumerate(documents):
        if not isinstance(text, str) or not text.strip():
            continue
        metadata = _metadata_fields(metadatas[index] if index < len(metadatas) else {})
        distance = distances[index] if index < len(distances) else None
        chunk_id = ids[index] if index < len(ids) else None
        metadata.setdefault("chunk_index", chunk_id or "Unknown")
        found.append({
            "text": text.strip(),
            "source": str(metadata.get("source", "Unknown")),
            "page": metadata.get("page", "Unknown"),
            "page_end": metadata.get("page_end", "Unknown"),
            "section": metadata.get("section", "Unknown"),
            "chapter": metadata.get("chapter", "Unknown"),
            "subsection": metadata.get("subsection", "Unknown"),
            "chunk_index": metadata.get("chunk_index", "Unknown"),
            "distance": float(distance) if distance is not None else None,
            "id": ids[index] if index < len(ids) else None,
            "metadata": metadata,
        })
    return found


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold(), flags=re.UNICODE).strip()


def _tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[\w]+", text.casefold()) if len(word) > 2}


def _relevance(query: str, document: dict[str, Any]) -> float:
    """Combine Chroma distance with lexical overlap for filtering and ordering."""
    distance = document.get("distance")
    vector_score = max(0.0, 1.0 - float(distance) / 2.0) if distance is not None else 0.0
    query_terms = _tokens(query)
    doc_terms = _tokens(document.get("text", ""))
    lexical = len(query_terms & doc_terms) / max(1, len(query_terms))
    return 0.75 * vector_score + 0.25 * lexical


def filter_deduplicate_reorder(
    documents: Iterable[dict[str, Any]],
    question: str,
    *,
    enabled: bool = True,
    max_distance: float | None = 0.95,
) -> list[dict[str, Any]]:
    """Drop duplicate/weak results and rank remaining real passages by relevance."""
    by_key: dict[str, dict[str, Any]] = {}
    for original in documents:
        document = dict(original)
        text = document.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        distance = document.get("distance")
        if enabled and max_distance is not None and distance is not None:
            try:
                if float(distance) > max_distance:
                    continue
            except (TypeError, ValueError):
                pass
        key = _normalize(text)
        if not key:
            continue
        score = _relevance(question, document)
        document["relevance"] = score
        existing = by_key.get(key)
        if existing is None or score > existing["relevance"]:
            by_key[key] = document
    results = list(by_key.values())
    if enabled:
        query_terms = _tokens(question)
        # Reject passages with no shared query terms when lexical evidence is usable;
        # retain vector matches for short questions with too little signal.
        if len(query_terms) >= 2:
            results = [
                doc for doc in results
                if _tokens(doc["text"]) & query_terms or doc["relevance"] >= 0.55
            ]
    return sorted(results, key=lambda item: item["relevance"], reverse=True)


def compress_context(
    documents: list[dict[str, Any]],
    question: str,
    *,
    enabled: bool = True,
) -> list[dict[str, Any]]:
    """Apply safe extractive compression by removing repeated sentences only."""
    if not enabled:
        return [dict(document) for document in documents]
    seen_sentences: set[str] = set()
    compressed: list[dict[str, Any]] = []
    for original in documents:
        document = dict(original)
        kept: list[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", document.get("text", "")):
            sentence = sentence.strip()
            key = _normalize(sentence)
            if not key or key in seen_sentences:
                continue
            seen_sentences.add(key)
            kept.append(sentence)
        document["text"] = " ".join(kept)
        if document["text"]:
            compressed.append(document)
    return compressed


def build_rag_prompt(question: str, retrieved_documents: list[dict[str, Any]]) -> str:
    """Build the original evidence-only answer prompt with full source traceability."""
    evidence_parts = []
    for index, document in enumerate(retrieved_documents, start=1):
        metadata = document.get("metadata") or {}
        evidence_parts.append(
            f"Evidence {index}\n"
            f"Source: {document.get('source', 'Unknown')}\n"
            f"Page: {document.get('page', 'Unknown')}"
            f"{(' - ' + str(document['page_end'])) if document.get('page_end') not in (None, 'Unknown', document.get('page')) else ''}\n"
            f"Section: {document.get('section', metadata.get('section', 'Unknown'))}\n"
            f"Chapter: {document.get('chapter', metadata.get('chapter', 'Unknown'))}\n"
            f"Chunk: {document.get('chunk_index', 'Unknown')}\n"
            f"Content:\n{document.get('text', '')}"
        )
    evidence = "\n\n".join(evidence_parts)
    return f"""Use ONLY the following retrieved evidence to answer the user's question.

STRICT RULES:
1. Use only information explicitly stated in the evidence.
2. Never use outside knowledge or unsupported assumptions.
3. Copy numerical values, units, definitions, and legal references accurately.
4. Do not calculate or invent values unless explicitly required and supported.
5. If the evidence does not support an answer, respond exactly: "{INSUFFICIENT_EVIDENCE}"
6. Give a short, direct answer.

RETRIEVED EVIDENCE:

{evidence}

ORIGINAL QUESTION:

{question}

ANSWER:"""


def answer_question_with_rag(
    question: str,
    collection: Any,
    top_k: int | None = None,
    *,
    config: AugmentationConfig = DEFAULT_CONFIG,
    generator=generate_ollama_response,
) -> dict[str, Any]:
    """Run query augmentation, independent retrieval, evidence cleanup, and answer."""
    if not question or not question.strip():
        return {"question": question, "answer": INSUFFICIENT_EVIDENCE, "sources": []}
    retrieval_k = max(1, top_k if top_k is not None else config.retrieval_top_k)
    print("\nPreparing focused retrieval queries...")
    expansions, alternatives = generate_query_variants(
        question,
        expansion_count=config.generated_query_count,
        multi_query_count=config.generated_query_count,
        query_expansion=config.query_expansion,
        multi_query=config.multi_query,
        generator=generator,
    )
    queries: list[str] = [question]
    if config.query_expansion:
        queries.extend(expansions)
    if config.multi_query:
        queries.extend(alternatives)
    unique_queries: list[str] = []
    seen_queries: set[str] = set()
    for query in queries:
        key = _normalize(query)
        if key and key not in seen_queries:
            seen_queries.add(key)
            unique_queries.append(query)

    retrieved: list[dict[str, Any]] = []
    for query in unique_queries:
        try:
            retrieved.extend(retrieve_documents(collection, query, retrieval_k))
        except Exception as exc:
            print(f"Retrieval failed for one query; continuing ({exc}).")

    # HyDE is a separate retrieval query. Its generated text is deliberately not
    # added to `retrieved`; only the real Chroma results are retained.
    if config.hyde:
        hypothetical = generate_hypothetical_document(question, generator=generator)
        if hypothetical:
            try:
                retrieved.extend(retrieve_documents(collection, hypothetical, retrieval_k))
            except Exception as exc:
                print(f"HyDE retrieval failed; continuing with question results ({exc}).")

    print(f"Retrieved {len(retrieved)} passages before filtering and deduplication.")
    processed = filter_deduplicate_reorder(
        retrieved,
        question,
        enabled=config.context_filtering,
        max_distance=config.max_distance,
    )
    processed = compress_context(
        processed, question, enabled=config.context_compression
    )
    processed = processed[: max(1, config.final_context_passages)]
    if not processed:
        return {"question": question, "answer": INSUFFICIENT_EVIDENCE, "sources": []}

    prompt = build_rag_prompt(question, processed)
    system_prompt = (
        "You are a precise, evidence-based question-answering system. "
        "Answer strictly from the supplied verified retrieved passages. "
        "The question is the original user question. Do not claim unsupported facts."
    )
    try:
        answer = generator(system_prompt, prompt)
    except Exception as exc:
        print(f"Answer generation failed ({exc}).")
        answer = INSUFFICIENT_EVIDENCE

    sources = []
    for document in processed:
        source = {
            "source": document.get("source", "Unknown"),
            "page": document.get("page", "Unknown"),
            "chunk_index": document.get("chunk_index", "Unknown"),
            "distance": document.get("distance"),
            "section": document.get("section", "Unknown"),
            "page_end": document.get("page_end", "Unknown"),
            "text": document.get("text", ""),
        }
        if document.get("metadata"):
            source["metadata"] = document["metadata"]
        sources.append(source)
    return {"question": question, "answer": answer or INSUFFICIENT_EVIDENCE, "sources": sources}


def _existing_collection(db_path: Path, collection_name: str, embedding_model: str):
    """Open an existing Chroma collection without creating a database or collection."""
    if not db_path.exists():
        return None
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        client = chromadb.PersistentClient(path=str(db_path))
        return client.get_collection(name=collection_name, embedding_function=embedding)
    except Exception as exc:
        print(f"Could not open {collection_name} at {db_path}: {exc}")
        return None


def load_collections() -> dict[str, Any]:
    """Open the existing Legal and ECE source collections without rebuilding them."""
    db_path = (
        PROJECT_ROOT / "data_preprocessing" / "chromaDB" / "indian_laws_chromadb"
    )
    legal = _existing_collection(
        db_path, "indian_statutory_acts", "BAAI/bge-base-en-v1.5"
    )
    ece = _existing_collection(
        db_path, "research_papers", "BAAI/bge-base-en-v1.5"
    )
    result = {}
    if legal is not None:
        result["legal"] = legal
    if ece is not None:
        result["ece"] = ece
    return result


def main() -> None:
    print("AUGMENTATION-OPTIMIZED RAG")
    collections = load_collections()
    if not collections:
        print("No configured Chroma collections are available in this checkout.")
        return
    choices = list(collections)
    print("Available datasets:")
    for index, name in enumerate(choices, start=1):
        print(f"{index}. {name.title()}")
    choice = input("Choose a dataset: ").strip()
    if not choice.isdigit() or not 1 <= int(choice) <= len(choices):
        print("Invalid dataset selection.")
        return
    question = input("Enter your question: ").strip()
    if not question:
        print("Please enter a valid question.")
        return
    result = answer_question_with_rag(question, collections[choices[int(choice) - 1]])
    print("\nANSWER\n" + result["answer"])
    print("\nRETRIEVED SOURCES")
    for index, source in enumerate(result["sources"], start=1):
        print(
            f"{index}. {source['source']} | Page: {source['page']} | "
            f"Section: {source['section']} | Chunk: {source['chunk_index']} | "
            f"Distance: {source['distance']}"
        )


if __name__ == "__main__":
    main()
