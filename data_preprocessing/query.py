
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


# Database location
DB_PATH = Path(__file__).resolve().parent / "chroma_db"

# Same embedding model used during ingestion
embedding_function = (
    embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)

client = chromadb.PersistentClient(path=str(DB_PATH))

legal_collection = client.get_collection(
    name="legal_data",
    embedding_function=embedding_function
)

ece_collection = client.get_collection(
    name="ece_data",
    embedding_function=embedding_function
)


def search_documents(collection, question, top_k=5):

    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count())
    )

    print("\nSearch results:")
    print("=" * 70)

    if not results["documents"] or not results["documents"][0]:
        print("No matching documents found.")
        return

    for i, document in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        print(f"\nResult {i + 1}")
        print("-" * 70)

        print("Source:", metadata["source"])
        print("Page:", metadata["page"])
        print("Chunk:", metadata["chunk_index"])
        print("Distance:", round(distance, 4))

        print("\nEvidence:")
        print(document)


if __name__ == "__main__":

    print("Choose your dataset:")
    print("1. Legal")
    print("2. Electronics")

    choice = input("\nEnter choice: ").strip()

    question = input("\nEnter your question: ").strip()

    if choice == "1":
        search_documents(legal_collection, question)

    elif choice == "2":
        search_documents(ece_collection, question)

    else:
        print("Invalid choice.")
