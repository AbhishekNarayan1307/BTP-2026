from __future__ import annotations

import logging
import os
from pathlib import Path
import threading
import time
from typing import Any

from run_all_models import (
    CHROMA_PATH,
    EMBEDDING_MODEL,
    MODEL_ADAPTERS,
    DatasetContext,
    get_dataset_context,
    run_models,
)

LOGGER = logging.getLogger("btp.web.runtime")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RuntimeInitializationError(RuntimeError):
    pass


class EvaluationRuntime:
    """Owns expensive resources shared by all API requests."""

    def __init__(self) -> None:
        self.status = "not_initialized"
        self.component_errors: dict[str, str] = {}
        self.initialization_counts: dict[str, int] = {}
        self.contexts: dict[str, DatasetContext] = {}
        self.collection_counts: dict[str, int] = {}
        self.embedding_function: Any = None
        self.chroma_client: Any = None
        self.reranker: Any = None
        self.groq_client: Any = None
        self.request_lock = threading.Lock()
        self.started_at: float | None = None

    def _count(self, component: str) -> None:
        self.initialization_counts[component] = (
            self.initialization_counts.get(component, 0) + 1
        )

    def initialize(self) -> None:
        if self.status == "ready":
            return
        self.status = "initializing"
        self.started_at = time.time()
        LOGGER.info("Initializing BTP evaluation runtime")

        try:
            try:
                from dotenv import load_dotenv

                load_dotenv(PROJECT_ROOT / ".env")
            except ImportError:
                pass

            # Importing query initializes the one shared SentenceTransformer
            # embedding function and opens both existing Chroma collections.
            try:
                from data_preprocessing import query

                self.embedding_function = query.embedding_function
                self.chroma_client = query.client
                self._count("sentence_transformer_embedding_function")
                if query.client is None:
                    raise RuntimeError(getattr(query, "DB_INIT_ERROR", "Chroma client unavailable"))
                for choice, domain in (("1", "Legal"), ("2", "ECE")):
                    context = get_dataset_context(choice)
                    if not context.available:
                        raise RuntimeError(context.error or "Collection unavailable")
                    count = int(context.collection.count())
                    if count <= 0:
                        raise RuntimeError(f"{context.collection_name} is empty")
                    self.contexts[domain] = context
                    self.collection_counts[domain] = count
                    LOGGER.info(
                        "Opened %s collection %s with %d documents",
                        domain,
                        context.collection_name,
                        count,
                    )
                self._count("chroma_persistent_client")
                self._count("chroma_collection:Legal")
                self._count("chroma_collection:ECE")
            except Exception as exc:
                self.component_errors["ChromaDB/embeddings"] = str(exc)
                raise RuntimeInitializationError(
                    f"ChromaDB/embeddings initialization failed: {exc}"
                ) from exc

            try:
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise RuntimeError("GROQ_API_KEY is not configured in .env or environment")
                from groq import Groq

                self.groq_client = Groq(api_key=api_key)
                from src import rag_pipeline

                rag_pipeline.configure_groq_client(self.groq_client)
                from RAG_pipeline_optimization.augmentation import augment

                augment.configure_groq_client(self.groq_client)
                self._count("groq_client")
            except Exception as exc:
                self.component_errors["Groq client"] = str(exc)
                raise RuntimeInitializationError(
                    f"Groq client initialization failed: {exc}"
                ) from exc

            try:
                from sentence_transformers import CrossEncoder

                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                self._count("cross_encoder")
            except Exception as exc:
                self.component_errors["CrossEncoder"] = str(exc)
                raise RuntimeInitializationError(
                    f"CrossEncoder initialization failed: {exc}"
                ) from exc

            try:
                from RAG_pipeline_optimization.retrieval.retrieve import HybridRetriever

                for domain, context in self.contexts.items():
                    context.hybrid_retriever = HybridRetriever(
                        chroma_path=CHROMA_PATH,
                        collection_names=[context.collection_name],
                        json_sources=context.json_sources,
                        embedding_model=EMBEDDING_MODEL,
                        chunk_size=None,
                        embedding_function=self.embedding_function,
                        reranker=self.reranker,
                        chroma_client=self.chroma_client,
                    )
                    self._count(f"hybrid_retriever:{domain}")
                    self._count(f"bm25_index:{domain}")
                    context.groq_client = self.groq_client

                # Import Model 5's chain once so its LangChain client and prompt
                # objects are retained across all subsequent requests.
                from RAG_pipeline_optimization.Generation import generate

                generate.configure_groq_client(self.groq_client)
                self._count("generation_pipeline")
            except Exception as exc:
                self.component_errors["RAG pipelines"] = str(exc)
                raise RuntimeInitializationError(
                    f"RAG pipeline initialization failed: {exc}"
                ) from exc

            self.status = "ready"
            LOGGER.info("BTP evaluation runtime ready")
        except RuntimeInitializationError:
            self.status = "failed"
            LOGGER.exception("BTP evaluation runtime initialization failed")
        except Exception as exc:
            self.component_errors["runtime"] = str(exc)
            self.status = "failed"
            LOGGER.exception("BTP evaluation runtime initialization failed")

    def health(self) -> dict[str, Any]:
        collection_status = {}
        for domain in ("Legal", "ECE"):
            context = self.contexts.get(domain)
            collection_status[domain] = {
                "collection": context.collection_name if context else None,
                "available": bool(context and context.available),
                "document_count": (
                    self.collection_counts.get(domain, 0)
                ),
                "database_path": str(CHROMA_PATH),
            }
        return {
            "status": "ok" if self.status == "ready" else "degraded",
            "runtime_status": self.status,
            "available_models": [name for name, _, _ in MODEL_ADAPTERS],
            "collections": collection_status,
            "component_errors": dict(self.component_errors),
            "initialization_counts": dict(self.initialization_counts),
        }

    def ask(self, domain: str, question: str) -> dict[str, Any]:
        if self.status != "ready":
            details = "; ".join(
                f"{component}: {error}"
                for component, error in self.component_errors.items()
            ) or "Runtime is not initialized"
            raise RuntimeInitializationError(details)
        context = self.contexts.get(domain)
        if context is None or not context.available:
            raise RuntimeInitializationError(f"{domain} collection is unavailable")

        LOGGER.info("Starting five-model evaluation for %s", domain)
        started = time.perf_counter()
        with self.request_lock:
            results = run_models(question, context)
        total_seconds = time.perf_counter() - started
        for result in results:
            LOGGER.info(
                "%s finished with status=%s in %.2fs",
                result.name,
                result.status,
                result.elapsed_seconds,
            )
        return {
            "domain": domain,
            "question": question,
            "total_execution_seconds": total_seconds,
            "results": [
                {
                    "model_name": result.name,
                    "status": result.status,
                    "answer": result.answer,
                    "sources": result.sources,
                    "retrieval_count": result.retrieval_count,
                    "execution_seconds": result.elapsed_seconds,
                    "verification_score": result.confidence,
                    "abstained": result.abstained,
                    "verification": result.verification,
                    "error": result.error,
                }
                for result in results
            ],
        }
