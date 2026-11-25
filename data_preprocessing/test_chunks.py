import chromadb
from chromadb.utils import embedding_functions

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "indian_laws_chromadb")
COLLECTION_NAME = "indian_statutory_acts"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Connect to existing persistent ChromaDB
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(
    name=COLLECTION_NAME, embedding_function=embed_fn
)

print(f"Connected to '{COLLECTION_NAME}' | Total vectors: {collection.count()}\n")

# ==========================================
# TEST 1: VERIFY FULL TEXT STORAGE (NO TRUNCATION)
# ==========================================
print("=" * 60)
print("TEST 1: FULL CHUNK STORAGE INTEGRITY CHECK")
print("=" * 60)

record = collection.get(
    where={
        "$and": [
            {"act": "Consumer Protection Act, 2019"},
            {"section": "Section 34"},
        ]
    }
)

for meta in record["metadatas"]:
  print(f"Act: {meta['act']} | {meta['chapter']} | {meta['section']}")
  print(f"Stored character length: {len(meta['raw_content'])} chars")
  print("-" * 40)
  print(meta["raw_content"])  # Full untruncated text
  print("-" * 40)

# ==========================================
# TEST 2: RETRIEVAL RECALL BENCHMARK (TOP-5)
# ==========================================
print("\n" + "=" * 60)
print("TEST 2: RETRIEVAL RECALL BENCHMARK (TOP-5)")
print("=" * 60)

benchmark = [
    (
        "What is the pecuniary jurisdiction of the District Commission?",
        "Consumer Protection Act, 2019",
        "Section 34",
    ),
    (
        "What is the penalty for failing to take reasonable security safeguards for personal data?",
        "Digital Personal Data Protection Act, 2023",
        "Schedule (Penalties)",
    ),
    (
        "What are the duties of a Data Fiduciary regarding children's personal data?",
        "Digital Personal Data Protection Act, 2023",
        "Section 9",
    ),
    (
        "What is the punishment for driving under the influence of alcohol or drugs?",
        "Motor Vehicles Act, 1988",
        "Section 185",
    ),
    (
        "What is the time limit to appeal an order of the District Commission to the State Commission?",
        "Consumer Protection Act, 2019",
        "Section 41",
    ),
    (
        "What is the compensation for death in a hit and run motor accident?",
        "Motor Vehicles Act, 1988",
        "Section 161",
    ),
]

passed = 0
for query, expected_act, expected_sec in benchmark:
  res = collection.query(query_texts=[query], n_results=5)
  retrieved = [
      (m["act"], m["section"], round(1 - dist, 4))
      for m, dist in zip(res["metadatas"][0], res["distances"][0])
  ]
  hit = any(
      act == expected_act and sec == expected_sec for act, sec, _ in retrieved
  )
  if hit:
    passed += 1

  status = "PASS" if hit else "MISS"
  print(f"\n[{status}] Q: {query}")
  print(f"   Expected: {expected_act} -> {expected_sec}")
  print("   Top-5 Retrieved:")
  for rank, (act, sec, score) in enumerate(retrieved, 1):
    marker = " <-- MATCH" if (act == expected_act and sec == expected_sec) else ""
    print(f"     {rank}. {act} | {sec} (similarity: {score}){marker}")

print(f"\nRecall@5 Score: {passed}/{len(benchmark)} ({passed/len(benchmark)*100:.0f}%)")

# ==========================================
# TEST 3: INTERACTIVE LIVE QUERY TESTER
# ==========================================
print("\n" + "=" * 60)
print("TEST 3: INTERACTIVE SEARCH (Type 'exit' to quit)")
print("=" * 60)

while True:
  user_q = input("\nAsk a legal question: ").strip()
  if user_q.lower() in {"exit", "quit", "q"}:
    break
  if not user_q:
    continue

  res = collection.query(query_texts=[user_q], n_results=4)
  for rank, (meta, dist) in enumerate(
      zip(res["metadatas"][0], res["distances"][0]), 1
  ):
    print(
        f"\n--- Result {rank} [Similarity: {1 - dist:.4f}] | "
        f"{meta['act']} | {meta['chapter']} | {meta['section']} ---"
    )
    print(meta["raw_content"])