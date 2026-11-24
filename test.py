from pathlib import Path
import chromadb
from collections import Counter

DB_PATH = (
    Path(__file__).resolve().parent
    / "data_preprocessing"
    / "chroma_db"
)

client = chromadb.PersistentClient(path=str(DB_PATH))

collection = client.get_collection(name="legal_data_v3")

results = collection.get(include=["metadatas"])

counts = Counter(
    metadata.get("source", "Unknown")
    for metadata in results["metadatas"]
)

print("Total chunks:", collection.count())

for source, count in counts.items():
    print(f"{source}: {count} chunks")
    