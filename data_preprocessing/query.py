
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

# Existing persisted database in this checkout. Both available collections use
# the BGE embeddings recorded in their Chroma collection metadata.
DB_PATH = Path(__file__).resolve().parent / "chromaDB" / "indian_laws_chromadb"


# Same embedding model used during document ingestion
embedding_function = (
    embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-base-en-v1.5"
    )
)


# Open existing data only. PersistentClient creates an empty database when the
# path is absent, so check first to avoid accidental database creation.
COLLECTION_ERRORS = {}
client = None
if not DB_PATH.is_dir():
    DB_INIT_ERROR = f"Chroma database directory is missing: {DB_PATH}"
else:
    try:
        client = chromadb.PersistentClient(path=str(DB_PATH))
        DB_INIT_ERROR = None
    except Exception as exc:
        DB_INIT_ERROR = f"Could not open Chroma database at {DB_PATH}: {exc}"


def _load_existing_collection(name):
    if client is None:
        COLLECTION_ERRORS[name] = DB_INIT_ERROR or "Chroma database is unavailable."
        return None
    try:
        return client.get_collection(
            name=name,
            embedding_function=embedding_function,
        )
    except Exception as exc:
        COLLECTION_ERRORS[name] = f"Collection '{name}' is unavailable: {exc}"
        return None


legal_collection = _load_existing_collection("indian_statutory_acts")
ece_collection = _load_existing_collection("research_papers")


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
    ids = results.get("ids", [[]])[0]

    for index, (document, metadata, distance) in enumerate(zip(
        documents, metadatas, distances
    )):

        metadata = metadata or {}
        chunk_id = ids[index] if index < len(ids) else "Unknown"

        retrieved_documents.append({
            "text": document,
            "source": metadata.get(
                "source",
                metadata.get("act", metadata.get("paper_title", "Unknown")),
            ),
            "page": metadata.get("page", metadata.get("pages", "Unknown")),
            "page_end": metadata.get("page_end", "Unknown"),
            "section": metadata.get("section", "Unknown"),
            "chapter": metadata.get("chapter", "Unknown"),
            "chunk_index": metadata.get("chunk_index", chunk_id),
            "source_path": metadata.get("source_path", "Unknown"),
            "category": metadata.get("category", "Unknown"),
            "distance": distance,
            "metadata": metadata,
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
