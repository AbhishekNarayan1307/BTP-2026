import json
import chromadb  # pip install chromadb
from chromadb.utils import embedding_functions  # pip install sentence-transformers

JSON_FILES = [
    "cpa_2019_rag_final.json",
    "dpdp_2023_rag_final.json",
    "mva_1988_rag_final.json",
]

DB_PATH = "./indian_laws_chromadb"
COLLECTION_NAME = "indian_statutory_acts"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


def load_all_chunks(file_paths):
  combined = []
  for path in file_paths:
    with open(path, "r", encoding="utf-8") as f:
      data = json.load(f)
      print(f"Loaded {len(data):>3} chunks from {path}")
      combined.extend(data)
  return combined


def build_chromadb_index():
  all_chunks = load_all_chunks(JSON_FILES)
  print(f"\nTotal chunks to index: {len(all_chunks)}")

  # Use BAAI/bge-base-en-v1.5 (768-dim, 512-token window — ideal for legal clauses)
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
  for i in range(0, len(all_chunks), batch_size):
    batch = all_chunks[i : i + batch_size]
    collection.upsert(
        ids=[f"law_chunk_{i + idx}" for idx in range(len(batch))],
        documents=[item["text_for_embedding"] for item in batch],
        metadatas=[
            {
                "act": item["act"],
                "chapter": item["chapter"],
                "section": item["section"],
                "raw_content": item["content"],
            }
            for item in batch
        ],
    )
    print(f"Indexed {min(i + batch_size, len(all_chunks))}/{len(all_chunks)} chunks...")

  print(f"\nDone! Total vectors stored in '{COLLECTION_NAME}': {collection.count()}")
  return collection


if __name__ == "__main__":
  collection = build_chromadb_index()

  # Quick sanity-check query across all three Acts
  test_queries = [
      "What is the penalty for failing to take reasonable security safeguards for personal data?",
      "What is the pecuniary jurisdiction of the District Commission?",
      "What is the punishment for drunken driving?",
  ]

  for q in test_queries:
    res = collection.query(query_texts=[q], n_results=1)
    meta = res["metadatas"][0][0]
    print(f"\nQ: {q}")
    print(f"Top Match -> [{meta['act']} | {meta['chapter']} | {meta['section']}]")
    print(meta["raw_content"][:220].replace("\n", " ") + "...")