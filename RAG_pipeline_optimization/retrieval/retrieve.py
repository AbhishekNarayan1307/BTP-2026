from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
import numpy as np
from rank_bm25 import BM25Okapi  # pip install rank-bm25
from sentence_transformers import CrossEncoder  # pip install sentence-transformers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PREPROC_DIR = PROJECT_ROOT / "data_preprocessing"

DEFAULT_CHROMA_CANDIDATES = [
    DATA_PREPROC_DIR / "ChromaDB",
    DATA_PREPROC_DIR / "chromaDB" / "indian_laws_chromadb",
    DATA_PREPROC_DIR / "ChromaDB" / "indian_laws_chromadb",
    DATA_PREPROC_DIR / "indian_laws_chromadb",
]

DEFAULT_JSON_SOURCES = [
    DATA_PREPROC_DIR / "cpa_2019_rag_final.json",
    DATA_PREPROC_DIR / "dpdp_2023_rag_final.json",
    DATA_PREPROC_DIR / "mva_1988_rag_final.json",
    DATA_PREPROC_DIR / "all_research_papers_chunks.json",
]

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RetrievedChunk:
  chunk_id: str
  content: str
  text_for_embedding: str
  metadata: dict[str, Any]
  semantic_rank: int | None = None
  bm25_rank: int | None = None
  rrf_score: float | None = None
  rerank_score: float | None = None


def resolve_chroma_path(custom_path: str | Path | None = None) -> Path:
  if custom_path is not None:
    return Path(custom_path).resolve()
  for candidate in DEFAULT_CHROMA_CANDIDATES:
    if candidate.exists():
      return candidate.resolve()
  return DEFAULT_CHROMA_CANDIDATES[0].resolve()


def _tokenize_for_bm25(text: str) -> list[str]:
  """Keeps alphanumeric tokens, section numbers, and hyphenated terms intact."""
  return re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower())


def _rechunk_text(
    text: str, max_chars: int, overlap_chars: int = 150
) -> list[str]:
  """Re-segments text to a target character budget on paragraph/sentence boundaries."""
  text = text.strip()
  if not text or max_chars <= 0 or len(text) <= max_chars:
    return [text] if text else []

  paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
  units: list[str] = []
  for para in paragraphs:
    if len(para) <= max_chars:
      units.append(para)
    else:
      sentences = re.split(r"(?<=[.!?])\s+", para)
      buf = ""
      for sent in sentences:
        if len(buf) + len(sent) + 1 > max_chars and buf:
          units.append(buf.strip())
          buf = sent
        else:
          buf = f"{buf} {sent}".strip()
      if buf:
        units.append(buf.strip())

  chunks: list[str] = []
  current = ""
  for unit in units:
    if not current:
      current = unit
    elif len(current) + len(unit) + 2 <= max_chars:
      current = f"{current}\n\n{unit}"
    else:
      chunks.append(current.strip())
      tail = current[-overlap_chars:].strip() if overlap_chars > 0 else ""
      if tail and " " in tail:
        tail = tail.split(" ", 1)[1]
      current = f"{tail}\n\n{unit}".strip() if tail else unit

  if current.strip():
    chunks.append(current.strip())

  return chunks


def _matches_filter(meta: dict[str, Any], where: dict[str, Any]) -> bool:
  for key, val in where.items():
    if meta.get(key) != val:
      return False
  return True


class HybridRetriever:
  """Reusable retrieval engine supporting semantic search, BM25, RRF hybrid search,
    dynamic chunk-size experiments, and CrossEncoder reranking.
    """

  def __init__(
      self,
      chroma_path: str | Path | None = None,
      collection_names: list[str] | None = None,
      json_sources: list[str | Path] | None = None,
      embedding_model: str = EMBEDDING_MODEL,
      reranker_model: str | None = RERANKER_MODEL,
      chunk_size: int | None = None,
      chunk_overlap: int = 150,
      embedding_function: Any | None = None,
      reranker: CrossEncoder | None = None,
      chroma_client: Any | None = None,
  ) -> None:
    self.chroma_path = resolve_chroma_path(chroma_path)
    self.collection_names = collection_names or [
        "indian_statutory_acts",
        "research_papers",
    ]
    self.json_sources = [
        Path(p).resolve() for p in (json_sources or DEFAULT_JSON_SOURCES)
    ]
    self.embedding_model = embedding_model
    self.reranker_model_name = reranker_model
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap

    self._embed_fn = embedding_function or embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=self.embedding_model
    )
    if not self.chroma_path.is_dir():
      raise FileNotFoundError(
          f"Chroma database directory does not exist: {self.chroma_path}"
      )
    self._client = chroma_client or chromadb.PersistentClient(path=str(self.chroma_path))
    self._collections = self._load_collections()
    loaded_names = {collection.name for collection in self._collections}
    missing_names = [
        name for name in self.collection_names if name not in loaded_names
    ]
    if missing_names:
      raise ValueError(
          f"Chroma collection(s) {missing_names} not found in "
          f"{self.chroma_path}. Available collections: "
          f"{sorted({collection.name for collection in self._client.list_collections()})}"
      )

    self._reranker: CrossEncoder | None = reranker
    if self._reranker is None and self.reranker_model_name:
      self._reranker = CrossEncoder(self.reranker_model_name)

    self._corpus: list[dict[str, Any]] = []
    self._bm25: BM25Okapi | None = None
    self._rebuild_corpus_and_bm25(
        chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
    )

  def _load_collections(self) -> list[Any]:
    loaded = []
    existing_names = {c.name for c in self._client.list_collections()}
    for name in self.collection_names:
      if name in existing_names:
        loaded.append(
            self._client.get_collection(
                name=name, embedding_function=self._embed_fn
            )
        )
    return loaded

  def _load_records_from_json(self) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in self.json_sources:
      if not path.exists():
        continue
      with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
      for idx, item in enumerate(items):
        content = item.get("content", "")
        if not content:
          continue
        meta = {
            k: v
            for k, v in item.items()
            if k not in {"content", "text_for_embedding"}
            and isinstance(v, (str, int, float, bool))
        }
        meta["source_file"] = path.name
        chunk_id = f"{path.stem}_{idx}"
        records.append({
            "id": chunk_id,
            "content": content,
            "text_for_embedding": item.get("text_for_embedding") or content,
            "metadata": meta,
        })
    return records

  def _load_records_from_chroma(self) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for col in self._collections:
      res = col.get(include=["documents", "metadatas"])
      ids = res.get("ids") or []
      docs = res.get("documents") or []
      metas = res.get("metadatas") or []
      for cid, doc, meta in zip(ids, docs, metas):
        clean_meta = dict(meta or {})
        raw_content = str(clean_meta.pop("raw_content", "") or doc or "")
        clean_meta["collection"] = col.name
        records.append({
            "id": cid,
            "content": raw_content,
            "text_for_embedding": doc or raw_content,
            "metadata": clean_meta,
        })
    return records

  def _rebuild_corpus_and_bm25(
      self, chunk_size: int | None, chunk_overlap: int
  ) -> None:
    base_records = self._load_records_from_json()
    if not base_records:
      base_records = self._load_records_from_chroma()

    processed: list[dict[str, Any]] = []
    for rec in base_records:
      pieces = (
          _rechunk_text(rec["content"], chunk_size, chunk_overlap)
          if chunk_size
          else [rec["content"]]
      )
      total_parts = len(pieces)
      for part_i, piece in enumerate(pieces, 1):
        meta = dict(rec["metadata"])
        if total_parts > 1:
          meta["sub_chunk"] = f"{part_i}/{total_parts}"
        source_title = (
            meta.get("act")
            or meta.get("paper_title")
            or meta.get("paper_id")
            or "Document"
        )
        section_label = meta.get("section") or meta.get("chapter") or ""
        header = (
            f"[{source_title} | {section_label}]"
            if section_label
            else f"[{source_title}]"
        )
        emb_text = f"{header}\n{piece}"
        processed.append({
            "id": f"{rec['id']}_p{part_i}" if total_parts > 1 else rec["id"],
            "content": piece,
            "text_for_embedding": emb_text,
            "metadata": meta,
        })

    self._corpus = processed
    tokenized_docs = [
        _tokenize_for_bm25(c["text_for_embedding"]) for c in self._corpus
    ]
    self._bm25 = BM25Okapi(tokenized_docs) if tokenized_docs else None

  def set_chunk_config(
      self, chunk_size: int | None, chunk_overlap: int = 150
  ) -> None:
    """Switch chunk size at runtime to test chunk-size trade-offs."""
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap
    self._rebuild_corpus_and_bm25(chunk_size, chunk_overlap)

  def _semantic_search(
      self,
      query: str,
      top_n: int = 40,
      where: dict[str, Any] | None = None,
  ) -> list[RetrievedChunk]:
    if not self._collections:
      return []

    if self.chunk_size is not None and self._corpus:
      query_vec = self._embed_fn([query])[0]
      doc_texts = [c["text_for_embedding"] for c in self._corpus]
      doc_vecs = self._embed_fn(doc_texts)

      q = np.array(query_vec, dtype=float)
      q_norm = np.linalg.norm(q) or 1.0
      scored = []
      for idx, (rec, d_vec) in enumerate(zip(self._corpus, doc_vecs)):
        if where and not _matches_filter(rec["metadata"], where):
          continue
        d = np.array(d_vec, dtype=float)
        sim = float(np.dot(q, d) / (q_norm * (np.linalg.norm(d) or 1.0)))
        scored.append((sim, idx, rec))

      scored.sort(key=lambda x: x[0], reverse=True)
      results: list[RetrievedChunk] = []
      for rank, (_, _, rec) in enumerate(scored[:top_n], 1):
        results.append(
            RetrievedChunk(
                chunk_id=rec["id"],
                content=rec["content"],
                text_for_embedding=rec["text_for_embedding"],
                metadata=rec["metadata"],
                semantic_rank=rank,
            )
        )
      return results

    merged_hits: list[tuple[float, RetrievedChunk]] = []
    for col in self._collections:
      col_count = col.count()
      if col_count == 0:
        continue
      query_kwargs: dict[str, Any] = {
          "query_texts": [query],
          "n_results": min(top_n, col_count),
          "include": ["documents", "metadatas", "distances"],
      }
      if where:
        query_kwargs["where"] = where
      try:
        res = col.query(**query_kwargs)
      except Exception:
        continue

      ids = (res.get("ids") or [[]])[0]
      docs = (res.get("documents") or [[]])[0]
      metas = (res.get("metadatas") or [[]])[0]
      dists = (res.get("distances") or [[]])[0]

      for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        clean_meta = dict(meta or {})
        raw_content = str(clean_meta.pop("raw_content", "") or doc or "")
        sim = 1.0 - float(dist if dist is not None else 1.0)
        merged_hits.append((
            sim,
            RetrievedChunk(
                chunk_id=cid,
                content=raw_content,
                text_for_embedding=doc or raw_content,
                metadata=clean_meta,
            ),
        ))

    merged_hits.sort(key=lambda x: x[0], reverse=True)
    ranked: list[RetrievedChunk] = []
    for rank, (_, item) in enumerate(merged_hits[:top_n], 1):
      item.semantic_rank = rank
      ranked.append(item)
    return ranked

  def _bm25_search(
      self,
      query: str,
      top_n: int = 40,
      where: dict[str, Any] | None = None,
  ) -> list[RetrievedChunk]:
    if not self._bm25 or not self._corpus:
      return []

    tokens = _tokenize_for_bm25(query)
    if not tokens:
      return []

    scores = self._bm25.get_scores(tokens)
    scored_items: list[tuple[float, int]] = []
    for idx, score in enumerate(scores):
      if score <= 0:
        continue
      if where and not _matches_filter(self._corpus[idx]["metadata"], where):
        continue
      scored_items.append((float(score), idx))

    scored_items.sort(key=lambda x: x[0], reverse=True)
    results: list[RetrievedChunk] = []
    for rank, (_, idx) in enumerate(scored_items[:top_n], 1):
      rec = self._corpus[idx]
      results.append(
          RetrievedChunk(
              chunk_id=rec["id"],
              content=rec["content"],
              text_for_embedding=rec["text_for_embedding"],
              metadata=rec["metadata"],
              bm25_rank=rank,
          )
      )
    return results

  @staticmethod
  def _reciprocal_rank_fusion(
      semantic_list: list[RetrievedChunk],
      bm25_list: list[RetrievedChunk],
      rrf_k: int = 60,
      semantic_weight: float = 1.0,
      bm25_weight: float = 1.0,
  ) -> list[RetrievedChunk]:
    fused_map: dict[str, RetrievedChunk] = {}

    def _dedup_key(item: RetrievedChunk) -> str:
      norm_text = re.sub(r"\s+", " ", item.content.strip().lower())
      return norm_text[:300]

    for item in semantic_list:
      key = _dedup_key(item)
      if key not in fused_map:
        fused_map[key] = item
      else:
        if item.semantic_rank is not None and (
            fused_map[key].semantic_rank is None
            or item.semantic_rank < fused_map[key].semantic_rank
        ):
          fused_map[key].semantic_rank = item.semantic_rank
      if item.semantic_rank:
        fused_map[key].rrf_score = (fused_map[key].rrf_score or 0.0) + (
            semantic_weight / (rrf_k + item.semantic_rank)
        )

    for item in bm25_list:
      key = _dedup_key(item)
      if key not in fused_map:
        fused_map[key] = item
      else:
        if item.bm25_rank is not None and (
            fused_map[key].bm25_rank is None
            or item.bm25_rank < fused_map[key].bm25_rank
        ):
          fused_map[key].bm25_rank = item.bm25_rank
      if item.bm25_rank:
        fused_map[key].rrf_score = (fused_map[key].rrf_score or 0.0) + (
            bm25_weight / (rrf_k + item.bm25_rank)
        )

    fused_list = list(fused_map.values())
    fused_list.sort(key=lambda x: x.rrf_score or 0.0, reverse=True)
    return fused_list

  def rerank(
      self,
      query: str,
      candidates: list[RetrievedChunk],
      top_k: int = 5,
  ) -> list[RetrievedChunk]:
    if not candidates:
      return []
    if self._reranker is None:
      return candidates[:top_k]

    pairs = [[query, c.content] for c in candidates]
    scores = self._reranker.predict(pairs)
    for item, score in zip(candidates, scores):
      item.rerank_score = float(score)

    reranked = sorted(
        candidates, key=lambda x: x.rerank_score or float("-inf"), reverse=True
    )
    return reranked[:top_k]

  def retrieve(
      self,
      query: str,
      top_k: int = 5,
      candidate_pool_size: int = 40,
      mode: str = "hybrid",
      use_reranker: bool = True,
      rrf_k: int = 60,
      where: dict[str, Any] | None = None,
  ) -> list[RetrievedChunk]:
    """Main entry point to call from other files.

        `mode`: "hybrid" (BM25 + Semantic RRF), "semantic", or "bm25".
        Returns a list of `RetrievedChunk` objects ready for augmentation/generation.
        """
    pool_n = max(top_k, candidate_pool_size)

    if mode == "semantic":
      candidates = self._semantic_search(query, top_n=pool_n, where=where)
    elif mode == "bm25":
      candidates = self._bm25_search(query, top_n=pool_n, where=where)
    elif mode == "hybrid":
      sem_hits = self._semantic_search(query, top_n=pool_n, where=where)
      bm25_hits = self._bm25_search(query, top_n=pool_n, where=where)
      candidates = self._reciprocal_rank_fusion(
          sem_hits, bm25_hits, rrf_k=rrf_k
      )
    else:
      raise ValueError(
          f"Unsupported retrieval mode '{mode}'. Use 'hybrid', 'semantic', or 'bm25'."
      )

    if use_reranker and self._reranker is not None:
      return self.rerank(query, candidates[:pool_n], top_k=top_k)

    return candidates[:top_k]

  def retrieve_as_dicts(
      self, query: str, **kwargs: Any
  ) -> list[dict[str, Any]]:
    """Convenience helper returning plain dicts for downstream modules."""
    chunks = self.retrieve(query, **kwargs)
    return [
        {
            "chunk_id": c.chunk_id,
            "content": c.content,
            "metadata": c.metadata,
            "semantic_rank": c.semantic_rank,
            "bm25_rank": c.bm25_rank,
            "rrf_score": c.rrf_score,
            "rerank_score": c.rerank_score,
        }
        for c in chunks
    ]


def run_optimization_experiments(
    retriever: HybridRetriever,
    queries: list[str],
    chunk_sizes: list[int | None] = [None, 800, 1500, 2500],
    top_k_values: list[int] = [3, 5, 8],
) -> None:
  """Grid experiment over chunk sizes and top_k to compare latency, coverage, and scores."""
  print("\n" + "=" * 72)
  print("CHUNK SIZE & TOP-K OPTIMIZATION EXPERIMENT")
  print("=" * 72)

  for c_size in chunk_sizes:
    t_build_start = time.perf_counter()
    retriever.set_chunk_config(c_size)
    build_ms = (time.perf_counter() - t_build_start) * 1000
    size_label = "original" if c_size is None else f"{c_size} chars"

    for k in top_k_values:
      t0 = time.perf_counter()
      all_hits = [
          retriever.retrieve(q, top_k=k, use_reranker=True) for q in queries
      ]
      elapsed_ms = ((time.perf_counter() - t0) / max(1, len(queries))) * 1000

      avg_top_score = sum(
          (hits[0].rerank_score or 0.0) for hits in all_hits if hits
      ) / max(1, len(all_hits))
      avg_chars = sum(
          len(c.content) for hits in all_hits for c in hits
      ) / max(1, sum(len(hits) for hits in all_hits))

      print(
          f"Chunk={size_label:<10} | Top-K={k:<2} | "
          f"Avg Rerank Score={avg_top_score:>6.3f} | "
          f"Avg Chunk Len={avg_chars:>6.0f} chars | "
          f"Query Time={elapsed_ms:>6.1f} ms (build {build_ms:.0f} ms)"
      )


if __name__ == "__main__":
  sample_queries = [
      "What is the pecuniary jurisdiction of the District Commission?",
      "What is the penalty for failing to take reasonable security safeguards for personal data?",
      "What is the punishment for driving under the influence of alcohol or drugs?",
      "How does electroplating affect tensile anisotropy in SLS PA12 parts?",
  ]

  retriever = HybridRetriever()

  run_optimization_experiments(
      retriever,
      queries=sample_queries,
      chunk_sizes=[None, 1000, 2000],
      top_k_values=[3, 5],
  )

  retriever.set_chunk_config(None)
  test_q = "who can file a consumer complaint?"
  results = retriever.retrieve(
      test_q,
      top_k=5,
      candidate_pool_size=40,
      mode="hybrid",
      use_reranker=True,
  )

  print("\n" + "=" * 72)
  print(f"SAMPLE HYBRID + RERANKED RESULTS FOR: {test_q}")
  print("=" * 72)
  for rank, r in enumerate(results, 1):
    sec = r.metadata.get("section") or r.metadata.get("paper_id") or "N/A"
    src = (
        r.metadata.get("act")
        or r.metadata.get("paper_title")
        or r.metadata.get("source_file")
    )
    print(
        f"\n#{rank} | Rerank={r.rerank_score:.4f} | RRF={r.rrf_score:.4f} | "
        f"SemRank={r.semantic_rank} | BM25Rank={r.bm25_rank} | {src} -> {sec}"
    )
    print(
        r.content[:280].replace("\n", " ")
        + ("..." if len(r.content) > 280 else "")
    )
