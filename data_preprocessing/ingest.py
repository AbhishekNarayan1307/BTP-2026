
from pathlib import Path
import hashlib
import re

import chromadb
import fitz
from chromadb.utils import embedding_functions


# --------------------------------------------------
# PATHS
# --------------------------------------------------

# This script is inside BTP-2026/data preprocessing
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = PROJECT_ROOT / "Data" / "rawdata"
DB_PATH = Path(__file__).resolve().parent / "chroma_db"

LEGAL_FOLDER = RAW_DATA / "Legal_data"
ECE_FOLDER = RAW_DATA / "electronics articles_training data"


# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

embedding_function = (
    embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)


# --------------------------------------------------
# CHROMADB INITIALIZATION
# --------------------------------------------------

client = chromadb.PersistentClient(path=str(DB_PATH))

legal_collection = client.get_or_create_collection(
    name="legal_data",
    embedding_function=embedding_function,
    metadata={"description": "Indian legal Acts"},
    configuration={"hnsw": {"space": "cosine"}}
)

ece_collection = client.get_or_create_collection(
    name="ece_data",
    embedding_function=embedding_function,
    metadata={"description": "Electronics research articles"},
    configuration={"hnsw": {"space": "cosine"}}
)


# --------------------------------------------------
# TEXT EXTRACTION
# --------------------------------------------------

def extract_pdf_pages(pdf_path):
    """Extract text from a PDF, keeping page numbers."""

    pages = []

    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text")

            if text.strip():
                pages.append({
                    "page": page_number,
                    "text": text
                })

    return pages


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def split_into_chunks(text, chunk_size=1000, overlap=150):
    """
    Split text into chunks of approximately chunk_size
    characters, with overlap between consecutive chunks.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Prefer ending at a sentence or word boundary
        if end < len(text):
            boundary = text.rfind(". ", start, end)

            if boundary > start + chunk_size // 2:
                end = boundary + 1
            else:
                space = text.rfind(" ", start, end)
                if space > start:
                    end = space

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


# --------------------------------------------------
# DOCUMENT PROCESSING
# --------------------------------------------------

def process_folder(folder, collection, category):
    """Extract, chunk, embed and store PDFs from a folder."""

    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    pdf_files = sorted(folder.glob("*.pdf"))

    print(f"\nProcessing {category}: {len(pdf_files)} PDFs")

    documents = []
    metadatas = []
    ids = []

    for pdf_path in pdf_files:
        print(f"Reading: {pdf_path.name}")

        pages = extract_pdf_pages(pdf_path)

        for page_data in pages:
            page_number = page_data["page"]
            page_text = page_data["text"]

            chunks = split_into_chunks(page_text)

            for chunk_index, chunk in enumerate(chunks):

                # Stable ID based on source, page and chunk
                unique_text = (
                    f"{pdf_path.name}_{page_number}_"
                    f"{chunk_index}_{chunk}"
                )

                chunk_id = hashlib.sha256(
                    unique_text.encode("utf-8")
                ).hexdigest()

                documents.append(chunk)

                metadatas.append({
                    "source": pdf_path.name,
                    "source_path": str(
                        pdf_path.relative_to(PROJECT_ROOT)
                    ),
                    "page": int(page_number),
                    "chunk_index": int(chunk_index),
                    "category": category
                })

                ids.append(chunk_id)

    # Add in batches to avoid large memory/API requests
    batch_size = 100

    for start in range(0, len(documents), batch_size):
        end = start + batch_size

        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end]
        )

    print(
        f"Stored {len(documents)} chunks in "
        f"'{collection.name}'"
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    process_folder(
        LEGAL_FOLDER,
        legal_collection,
        "legal"
    )

    process_folder(
        ECE_FOLDER,
        ece_collection,
        "electronics"
    )

    print("\nIngestion completed successfully!")

    print("Legal chunks:", legal_collection.count())
    print("ECE chunks:", ece_collection.count())
