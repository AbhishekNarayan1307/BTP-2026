
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

# Database location
DB_PATH = Path(__file__).resolve().parent / "chroma_db"


# Same embedding model used during document ingestion
embedding_function = (
    embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)


# Initialize persistent ChromaDB client
client = chromadb.PersistentClient(path=str(DB_PATH))


# Load Legal collection
legal_collection = client.get_collection(
    name="legal_data",
    embedding_function=embedding_function
)


# Load Electronics collection
ece_collection = client.get_collection(
    name="ece_data",
    embedding_function=embedding_function
)


# ============================================================
# DOCUMENT RETRIEVAL
# ============================================================

def retrieve_documents(collection, question, top_k=5):
    """
    Retrieve relevant document chunks from ChromaDB.

    Args:
        collection: ChromaDB collection.
        question: User's query.
        top_k: Maximum number of chunks to retrieve.

    Returns:
        List of dictionaries containing:
        - text
        - source
        - page
        - page_end
        - chapter
        - section
        - subsection
        - chunk_index
        - distance
    """

    collection_count = collection.count()

    if collection_count == 0:
        return []

    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection_count),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved_documents = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        metadata = metadata or {}

        retrieved_documents.append({
            "text": document,
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page", "Unknown"),
            "chunk_index": metadata.get("chunk_index", "Unknown"),
            "source_path": metadata.get("source_path", "Unknown"),
            "category": metadata.get("category", "Unknown"),
            "distance": distance
        })

    return retrieved_documents


# ============================================================
# DOCUMENT SEARCH AND DISPLAY
# ============================================================

def search_documents(collection, question, top_k=5):
    """
    Search documents and display retrieved evidence.
    """

    results = retrieve_documents(
        collection,
        question,
        top_k
    )

    print("\nSearch results:")
    print("=" * 70)

    if not results:
        print("No matching documents found.")
        return

    for i, document in enumerate(results, start=1):

        print(f"\nResult {i}")
        print("-" * 70)

        print("Source:", document["source"])
        print("Page:", document["page"])
        print("Chunk:", document["chunk_index"])
        print("Distance:", round(document["distance"], 4))

        print("\nEvidence:")
        print(document["text"])


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("Choose your dataset:")
    print("1. Legal")
    print("2. Electronics")

    choice = input("\nEnter choice: ").strip()

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Please enter a valid question.")

    elif choice == "1":
        search_documents(
            legal_collection,
            question
        )

    elif choice == "2":
        search_documents(
            ece_collection,
            question
        )

    else:
        print("Invalid choice.")

# from pathlib import Path

# import chromadb
# from chromadb.utils import embedding_functions


# # Database location
# DB_PATH = Path(__file__).resolve().parent / "chroma_db"

# # Same embedding model used during ingestion
# embedding_function = (
#     embedding_functions.SentenceTransformerEmbeddingFunction(
#         model_name="all-MiniLM-L6-v2"
#     )
# )

# client = chromadb.PersistentClient(path=str(DB_PATH))

# legal_collection = client.get_collection(
#     name="legal_data",
#     embedding_function=embedding_function
# )

# ece_collection = client.get_collection(
#     name="ece_data",
#     embedding_function=embedding_function
# )


# def search_documents(collection, question, top_k=5):

#     results = collection.query(
#         query_texts=[question],
#         n_results=min(top_k, collection.count())
#     )

#     print("\nSearch results:")
#     print("=" * 70)

#     if not results["documents"] or not results["documents"][0]:
#         print("No matching documents found.")
#         return

#     for i, document in enumerate(results["documents"][0]):

#         metadata = results["metadatas"][0][i]
#         distance = results["distances"][0][i]

#         print(f"\nResult {i + 1}")
#         print("-" * 70)

#         print("Source:", metadata["source"])
#         print("Page:", metadata["page"])
#         print("Chunk:", metadata["chunk_index"])
#         print("Distance:", round(distance, 4))

#         print("\nEvidence:")
#         print(document)


# if __name__ == "__main__":

#     print("Choose your dataset:")
#     print("1. Legal")
#     print("2. Electronics")

#     choice = input("\nEnter choice: ").strip()

#     question = input("\nEnter your question: ").strip()

#     if choice == "1":
#         search_documents(legal_collection, question)

#     elif choice == "2":
#         search_documents(ece_collection, question)

#     else:
#         print("Invalid choice.")

# from pathlib import Path

# import chromadb
# from chromadb.utils import embedding_functions


# # Database location
# DB_PATH = Path(__file__).resolve().parent / "chroma_db"

# # Same embedding model used during ingestion
# embedding_function = (
#     embedding_functions.SentenceTransformerEmbeddingFunction(
#         model_name="all-MiniLM-L6-v2"
#     )
# )

# client = chromadb.PersistentClient(path=str(DB_PATH))

# legal_collection = client.get_collection(
#     name="legal_data_v3",
#     embedding_function=embedding_function
# )

# ece_collection = client.get_collection(
#     name="ece_data",
#     embedding_function=embedding_function
# )


# def retrieve_documents(collection, question, top_k=5):
#     """
#     Retrieve relevant document chunks from ChromaDB.
#     Returns evidence along with source metadata.
#     """

#     collection_count = collection.count()

#     if collection_count == 0:
#         return []

#     results = collection.query(
#         query_texts=[question],
#         n_results=min(top_k, collection_count),
#         include=["documents", "metadatas", "distances"]
#     )

#     retrieved_documents = []

#         for document, metadata, distance in zip(
#         results["documents"][0],
#         results["metadatas"][0],
#         results["distances"][0]
#     ):
#         metadata = metadata or {}

#         retrieved_documents.append({
#             "text": document,
#             "source": metadata.get("source", "Unknown"),
#             "page": metadata.get("page_start", "Unknown"),
#             "page_end": metadata.get("page_end", "Unknown"),
#             "chapter": metadata.get("chapter", "Unknown"),
#             "section": metadata.get("section", "Unknown"),
#             "subsection": metadata.get("subsection", "Unknown"),
#             "chunk_index": metadata.get("part", "Unknown"),
#             "distance": distance
#         })

#     return retrieved_documents


# def search_documents(collection, question, top_k=5):

#     results = retrieve_documents(collection, question, top_k)

#     print("\nSearch results:")
#     print("=" * 70)

#     if not results:
#         print("No matching documents found.")
#         return

#     for i, document in enumerate(results):

#         print(f"\nResult {i + 1}")
#         print("-" * 70)

#         print("Source:", document["source"])
#         print("Page:", document["page"])
#         print("Chunk:", document["chunk_index"])
#         print("Distance:", round(document["distance"], 4))

#         print("\nEvidence:")
#         print(document["text"])


# if __name__ == "__main__":

#     print("Choose your dataset:")
#     print("1. Legal")
#     print("2. Electronics")

#     choice = input("\nEnter choice: ").strip()

#     question = input("\nEnter your question: ").strip()

#     if choice == "1":
#         search_documents(legal_collection, question)

#     elif choice == "2":
#         search_documents(ece_collection, question)

#     else:
#         print("Invalid choice.")