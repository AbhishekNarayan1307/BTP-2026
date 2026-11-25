from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from RAG_pipeline_optimization.retrieval.retrieve import HybridRetriever


def main() -> None:
  print("Initializing HybridRetriever (ChromaDB + BM25 + CrossEncoder)...")
  retriever = HybridRetriever(chunk_size=None)

  print("\nReady! Type your question and press Enter.")
  print(
      "Optional commands: 'chunk <number|none>' (e.g., 'chunk 1200' or 'chunk"
      " none'), 'k <number>' (e.g., 'k 5'), 'exit' to quit.\n"
  )

  top_k = 5

  while True:
    raw = input("Your Question > ").strip()
    if not raw:
      continue
    if raw.lower() in {"exit", "quit", "q"}:
      break

    if raw.lower().startswith("chunk "):
      arg = raw.split(" ", 1)[1].strip().lower()
      if arg == "none":
        retriever.set_chunk_config(None)
        print("Chunk size reset to original.")
      elif arg.isdigit():
        retriever.set_chunk_config(int(arg))
        print(f"Chunk size set to {int(arg)} chars.")
      else:
        print("Usage: chunk <number|none>")
      continue

    if raw.lower().startswith("k "):
      arg = raw.split(" ", 1)[1].strip()
      if arg.isdigit() and int(arg) > 0:
        top_k = int(arg)
        print(f"Top-K set to {top_k}.")
      else:
        print("Usage: k <positive integer>")
      continue

    hits = retriever.retrieve(
        query=raw,
        top_k=top_k,
        candidate_pool_size=40,
        mode="hybrid",
        use_reranker=True,
    )

    print(f"\nTop {len(hits)} retrieved passages for: \"{raw}\"")
    print("-" * 72)
    for idx, h in enumerate(hits, 1):
      source = (
          h.metadata.get("act")
          or h.metadata.get("paper_id")
          or h.metadata.get("source_file", "Unknown")
      )
      section = h.metadata.get("section") or h.metadata.get("chapter") or "N/A"
      rr_score = (
          f"{h.rerank_score:.4f}" if h.rerank_score is not None else "N/A"
      )
      rrf_score = f"{h.rrf_score:.4f}" if h.rrf_score is not None else "N/A"

      print(
          f"[{idx}] Rerank={rr_score} | RRF={rrf_score} | "
          f"SemRank={h.semantic_rank} | BM25Rank={h.bm25_rank}"
      )
      print(f"    Source : {source} | Section: {section}")
      print(f"    Text   : {h.content}\n")


if __name__ == "__main__":
  main()