
from pathlib import Path
import chromadb

DB_PATH = (
    Path(__file__).resolve().parent
    / "data_preprocessing"
    / "chroma_db"
)

client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_collection(name="legal_data_v3")

results = collection.get(include=["documents", "metadatas"])

checked = {
    "consumer_protection_act_2019_gazette.pdf": 0,
    "digital_personal_data.pdf": 0
}

for document, metadata in zip(
    results["documents"],
    results["metadatas"]
):
    source = metadata.get("source")

    if source not in checked or checked[source] >= 3:
        continue

    checked[source] += 1

    print("\n" + "=" * 80)
    print("SOURCE:", source)
    print("CHUNK:", checked[source])
    print("SECTION:", metadata.get("section"))
    print("SUBSECTION:", metadata.get("subsection"))
    print("PAGE:", metadata.get("page_start"), "-", metadata.get("page_end"))
    print("=" * 80)
    print(document)
