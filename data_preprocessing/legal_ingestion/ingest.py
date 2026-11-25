
import hashlib
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from .config import (
    PDF_DIR,
    DB_PATH,
    COLLECTION_NAME,
    PDF_CONFIG
)

from .extractors import extract_lines
from .chunkers import CHUNKING_STRATEGIES


def make_id(source, chunk, index):

    raw = (
        f"{source}|{chunk['section']}|"
        f"{chunk['subsection']}|{chunk['part']}|{index}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def main():

    embedding_function = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    existing = [
        c.name for c in client.list_collections()
    ]

    if COLLECTION_NAME in existing:

        answer = input(
            f"{COLLECTION_NAME} exists. "
            "Delete and recreate? (yes/no): "
        ).strip().lower()

        if answer != "yes":
            print("Cancelled.")
            return

        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    all_documents = []
    all_metadatas = []
    all_ids = []

    for filename, config in PDF_CONFIG.items():

        pdf_path = PDF_DIR / filename

        if not pdf_path.exists():

            print("Missing PDF:", pdf_path)
            continue

        print("\nProcessing:", filename)

        pages = extract_lines(pdf_path)

        strategy_name = config["strategy"]

        chunker = CHUNKING_STRATEGIES[strategy_name]

        chunks = chunker(pages)

        print("Strategy:", strategy_name)
        print("Chunks created:", len(chunks))

        for index, chunk in enumerate(chunks):

            # Add context to the embedded document.
            document_text = (
                f"Act: {config['act']}\n"
                f"Chapter: {chunk['chapter']}\n"
                f"Section: {chunk['section']}\n"
                f"Subsection: {chunk['subsection']}\n"
                f"Content:\n{chunk['text']}"
            )

            metadata = {
                "source": filename,
                "act": config["act"],
                "chapter": chunk["chapter"],
                "section": str(chunk["section"]),
                "subsection": str(chunk["subsection"]),
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "part": chunk["part"],
                "total_parts": chunk["total_parts"],
                "strategy": strategy_name,
                "document_type": "legal"
            }

            all_documents.append(document_text)
            all_metadatas.append(metadata)

            all_ids.append(
                make_id(filename, chunk, index)
            )

    batch_size = 100

    for start in range(0, len(all_documents), batch_size):

        end = start + batch_size

        collection.upsert(
            ids=all_ids[start:end],
            documents=all_documents[start:end],
            metadatas=all_metadatas[start:end]
        )

        print(
            f"Stored {min(end, len(all_documents))}"
            f"/{len(all_documents)} chunks"
        )

    print("\n" + "=" * 65)
    print("LEGAL INGESTION COMPLETED")
    print("=" * 65)

    print("Collection:", COLLECTION_NAME)
    print("Total chunks:", collection.count())


if __name__ == "__main__":
    main()
