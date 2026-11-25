import json
from pathlib import Path
import chromadb  # pip install chromadb
from chromadb.utils import embedding_functions  # pip install sentence-transformers

BASE_DIR = Path(__file__).resolve().parent
INPUT_JSON = BASE_DIR / "all_research_papers_chunks.json"
DB_PATH = str(BASE_DIR / "chromaDB" / "indian_laws_chromadb")
COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


def ingest_research_papers():
  with open(INPUT_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

  print(f"Loaded {len(chunks)} research paper chunks from {INPUT_JSON.name}")

  embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
      model_name=EMBEDDING_MODEL
  )

  client = chromadb.PersistentClient(path=DB_PATH)
  collection = client.get_or_create_collection(
      name=COLLECTION_NAME,
      embedding_function=embed_fn,
      metadata={"hnsw:space": "cosine"},
  )

  batch_size = 64
  for i in range(0, len(chunks), batch_size):
    batch = chunks[i : i + batch_size]
    collection.upsert(
        ids=[
            f"{item['paper_id']}_chunk_{i + idx}"
            for idx, item in enumerate(batch)
        ],
        documents=[item["text_for_embedding"] for item in batch],
        metadatas=[
            {
                "paper_id": item["paper_id"],
                "paper_title": item["paper_title"],
                "domain": item["domain"],
                "section": item["section"],
                "pages": item["pages"],
                "raw_content": item["content"],
            }
            for item in batch
        ],
    )

  print(
      f"Stored {collection.count()} chunks in ChromaDB collection"
      f" '{COLLECTION_NAME}' at {DB_PATH}"
  )


if __name__ == "__main__":
  ingest_research_papers()
