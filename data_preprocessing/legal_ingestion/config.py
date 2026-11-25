
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PDF_DIR = ROOT / "Data" / "rawdata" / "Legal_data"
DB_PATH = ROOT / "data_preprocessing" / "chroma_db"

COLLECTION_NAME = "legal_data_v3"

MAX_CHUNK_CHARS = 3500

PDF_CONFIG = {
    "consumer_protection_act_2019_gazette.pdf": {
        "act": "Consumer Protection Act, 2019",
        "strategy": "consumer",
    },

    "digital_personal_data.pdf": {
        "act": "Digital Personal Data Protection Act, 2023",
        "strategy": "dpdp",
    },
}
