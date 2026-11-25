"""Run the five BTP-2026 answering conditions on one selected dataset/question."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
import sys
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data_preprocessing"
CHROMA_PATH = DATA_DIR / "chromaDB" / "indian_laws_chromadb"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
INSUFFICIENT_EVIDENCE = "The provided evidence is insufficient to answer this question."


@dataclass
class DatasetContext:
    label: str
    collection_name: str
    json_sources: list[Path]
    collection: Any = None
    error: str | None = None
    hybrid_retriever: Any = None
    groq_client: Any = None

    @property
    def available(self) -> bool:
        return self.collection is not None and self.error is None


@dataclass
class ModelResult:
    name: str
    status: str
    elapsed_seconds: float
    answer: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    retrieval_count: int | None = None
    confidence: float | None = None
    abstained: bool | None = None
    verification: Any = None
    error: str | None = None


Adapter = Callable[[str, DatasetContext], dict[str, Any]]


def get_dataset_context(choice: str) -> DatasetContext:
    """Open only the selected, pre-existing dataset collection; never ingest data."""
    configs = {
        "1": (
            "Legal",
            "indian_statutory_acts",
            [
                DATA_DIR / "cpa_2019_rag_final.json",
                DATA_DIR / "dpdp_2023_rag_final.json",
                DATA_DIR / "mva_1988_rag_final.json",
            ],
        ),
        "2": (
            "ECE",
            "research_papers",
            [DATA_DIR / "all_research_papers_chunks.json"],
        ),
    }
    if choice not in configs:
        raise ValueError("Choose 1 for Legal or 2 for ECE.")
    label, collection_name, sources = configs[choice]
    context = DatasetContext(label, collection_name, sources)

    if not CHROMA_PATH.is_dir():
        context.error = f"Chroma database directory is missing: {CHROMA_PATH}"
        return context
    try:
        # The query module opens these exact existing collections with the BGE
        # embedding function used by both local ingestion scripts.
        from data_preprocessing import query

        context.collection = getattr(
            query, "legal_collection" if choice == "1" else "ece_collection"
        )
        if context.collection is None:
            detail = getattr(query, "COLLECTION_ERRORS", {}).get(collection_name)
            context.error = detail or f"Collection '{collection_name}' is unavailable."
    except Exception as exc:
        context.error = f"Could not open collection '{collection_name}': {exc}"
    return context


def _run_baseline(question: str, _context: DatasetContext) -> dict[str, Any]:
    from src.rag_pipeline import answer_question_direct

    return answer_question_direct(question)


class _CaptureCollection:
    """Forward collection calls while retaining raw query metadata for citations."""

    def __init__(self, collection: Any):
        self.collection = collection
        self.last_query: dict[str, Any] | None = None

    def count(self) -> int:
        return self.collection.count()

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.last_query = self.collection.query(**kwargs)
        return self.last_query


def _run_normal_rag(question: str, context: DatasetContext) -> dict[str, Any]:
    from src.rag_pipeline import answer_question_with_rag

    captured = _CaptureCollection(context.collection)
    result = answer_question_with_rag(question, captured, top_k=5)
    query_result = captured.last_query or {}
    metadatas = (query_result.get("metadatas") or [[]])[0] or []
    ids = (query_result.get("ids") or [[]])[0] or []
    sources = result.get("sources") or []
    for index, source in enumerate(sources):
        metadata = metadatas[index] if index < len(metadatas) else {}
        metadata = metadata or {}
        chunk_id = ids[index] if index < len(ids) else "Unknown"
        source.update(
            {
                "source": metadata.get(
                    "source",
                    metadata.get("act", metadata.get("paper_title", source.get("source"))),
                ),
                "page": metadata.get("page", metadata.get("pages", source.get("page"))),
                "section": metadata.get("section", "Unknown"),
                "chunk_index": metadata.get("chunk_index", chunk_id),
                "text": (query_result.get("documents") or [[]])[0][index]
                if index < len((query_result.get("documents") or [[]])[0])
                else "",
                "metadata": metadata,
            }
        )
    return result


def _run_retrieval_optimized(question: str, context: DatasetContext) -> dict[str, Any]:
    from RAG_pipeline_optimization.retrieval.retrieve import HybridRetriever
    from src.rag_pipeline import build_rag_prompt, generate_groq_response

    retriever = context.hybrid_retriever
    if retriever is None:
        # Preserve standalone CLI usage while allowing the web runtime to inject
        # the startup-initialized retriever and reuse it for every question.
        from data_preprocessing import query as query_module

        retriever = HybridRetriever(
            chroma_path=CHROMA_PATH,
            collection_names=[context.collection_name],
            json_sources=context.json_sources,
            embedding_model=EMBEDDING_MODEL,
            chunk_size=None,
            embedding_function=query_module.embedding_function,
        )
    hits = retriever.retrieve(
        query=question,
        top_k=5,
        candidate_pool_size=40,
        mode="hybrid",
        use_reranker=True,
    )
    documents = []
    for hit in hits:
        metadata = dict(hit.metadata or {})
        documents.append(
            {
                "text": hit.content,
                "source": metadata.get(
                    "source", metadata.get("act", metadata.get("paper_title", "Unknown"))
                ),
                "page": metadata.get("page", metadata.get("pages", "Unknown")),
                "page_end": metadata.get("page_end", "Unknown"),
                "section": metadata.get("section", "Unknown"),
                "chunk_index": metadata.get("chunk_index", hit.chunk_id),
                "distance": None,
                "metadata": metadata,
            }
        )

    if not documents:
        return {"question": question, "answer": INSUFFICIENT_EVIDENCE, "sources": []}

    prompt = build_rag_prompt(question, documents)
    system_prompt = """You are a precise, evidence-based question-answering system.

Your responsibility is to answer questions strictly using the evidence supplied by the user.
Never introduce external knowledge or unsupported assumptions."""
    answer = generate_groq_response(system_prompt, prompt)
    sources = [
        {
            "source": doc["source"],
            "page": doc["page"],
            "section": doc["section"],
            "chunk_index": doc["chunk_index"],
            "distance": doc["distance"],
                "text": doc["text"],
            "metadata": doc["metadata"],
        }
        for doc in documents
    ]
    return {"question": question, "answer": answer, "sources": sources}


def _run_augmentation(question: str, context: DatasetContext) -> dict[str, Any]:
    from RAG_pipeline_optimization.augmentation import augment

    generation_errors: list[Exception] = []
    if context.groq_client is not None:
        augment.configure_groq_client(context.groq_client)

    def tracked_generator(*args: Any, **kwargs: Any) -> str:
        try:
            return augment.generate_groq_response(*args, **kwargs)
        except Exception as exc:
            generation_errors.append(exc)
            raise

    result = augment.answer_question_with_rag(
        question,
        context.collection,
        top_k=5,
        generator=tracked_generator,
    )
    if generation_errors:
        raise RuntimeError(
            "Groq call failed inside the augmentation pipeline: "
            + "; ".join(f"{type(exc).__name__}: {exc}" for exc in generation_errors)
        )
    return result


def _run_generation_optimized(question: str, context: DatasetContext) -> dict[str, Any]:
    from RAG_pipeline_optimization.Generation.generate import answer_question_with_rag

    return answer_question_with_rag(
        question, context.collection, top_k=5, stream=False
    )


MODEL_ADAPTERS: list[tuple[str, Adapter, bool]] = [
    ("MODEL 1: BASELINE LLM", _run_baseline, False),
    ("MODEL 2: NORMAL RAG", _run_normal_rag, True),
    ("MODEL 3: RETRIEVAL-OPTIMIZED RAG", _run_retrieval_optimized, True),
    ("MODEL 4: AUGMENTATION-OPTIMIZED RAG", _run_augmentation, True),
    ("MODEL 5: GENERATION-OPTIMIZED RAG", _run_generation_optimized, True),
]


def run_models(
    question: str,
    context: DatasetContext,
    adapters: list[tuple[str, Adapter, bool]] | None = None,
) -> list[ModelResult]:
    """Execute adapters once each, sequentially, isolating failures per model."""
    # The RAG implementations print retrieved source passages. Some contain
    # characters outside the Windows console code page; keep those diagnostic
    # prints from aborting otherwise valid model calls.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="backslashreplace")
        except (OSError, ValueError):
            pass
    results = []
    for name, adapter, needs_dataset in (adapters or MODEL_ADAPTERS):
        started = time.perf_counter()
        if needs_dataset and not context.available:
            results.append(
                ModelResult(
                    name,
                    "Unavailable",
                    time.perf_counter() - started,
                    error=context.error or "The selected dataset is unavailable.",
                )
            )
            continue
        try:
            value = adapter(question, context)
            answer = value.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("The model returned an empty answer.")
            results.append(
                ModelResult(
                    name=name,
                    status="Success",
                    elapsed_seconds=time.perf_counter() - started,
                    answer=answer,
                    sources=value.get("sources") or [],
                    retrieval_count=(
                        len(value.get("sources") or []) if needs_dataset else None
                    ),
                    confidence=value.get("confidence"),
                    abstained=value.get("abstained"),
                    verification=value.get("verification"),
                )
            )
        except Exception as exc:
            results.append(
                ModelResult(
                    name=name,
                    status="Failed",
                    elapsed_seconds=time.perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return results


def _print_model_result(result: ModelResult) -> None:
    print("\n" + "=" * 56)
    print(result.name)
    print("=" * 56 + "\n")
    if result.status == "Success":
        print("Answer:")
        print(result.answer)
        if result.name != "MODEL 1: BASELINE LLM":
            print("\nSources:")
            if result.sources:
                for index, source in enumerate(result.sources, start=1):
                    details = [
                        f"Source: {source.get('source', 'Unknown')}",
                        f"Page: {source.get('page', 'Unknown')}",
                        f"Section: {source.get('section', 'Unknown')}",
                        f"Chunk: {source.get('chunk_index', 'Unknown')}",
                    ]
                    if source.get("distance") is not None:
                        details.append(f"Distance: {source['distance']}")
                    print(f"{index}. " + " | ".join(details))
            else:
                print("No source passages returned.")
        if result.confidence is not None:
            print(f"\nConfidence/support score: {result.confidence:.2f}")
        if result.abstained is not None:
            print(f"Abstained: {result.abstained}")
    else:
        print(f"{result.status}: {result.error}")
    print(f"\nExecution time: {result.elapsed_seconds:.2f}s")


def main() -> None:
    print("=" * 56)
    print("              BTP-2026: FIVE MODEL COMPARISON")
    print("=" * 56)
    print("\nSelect your dataset:\n\n1. Legal\n2. ECE")
    choice = input("\nEnter your choice: ").strip()
    if choice not in {"1", "2"}:
        print("Invalid choice. Select 1 for Legal or 2 for ECE.")
        return

    question = input("\nEnter your question:\n")
    if not question.strip():
        print("Please enter a question.")
        return

    context = get_dataset_context(choice)
    print("\n" + "=" * 56)
    print("Running all five models...")
    print("=" * 56)
    print(
        "Per-model times include adapter imports/initialization, retrieval, and "
        "generation. Shared dataset/collection initialization and user input "
        "are excluded."
    )

    results = run_models(question, context)
    for result in results:
        _print_model_result(result)

    print("\n" + "=" * 76)
    print("FINAL STATUS SUMMARY")
    print("=" * 76)
    print(f"{'Model':<38} {'Status':<14} {'Execution Time':>18}")
    print("-" * 76)
    for result in results:
        print(
            f"{result.name.removeprefix('MODEL '):<38} "
            f"{result.status:<14} {result.elapsed_seconds:>15.2f}s"
        )


if __name__ == "__main__":
    main()
