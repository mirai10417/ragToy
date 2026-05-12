from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

FAISS_PATH = PROCESSED_DIR / "faiss.index"
PARQUET_PATH = PROCESSED_DIR / "chunks.parquet"
SQLITE_PATH = PROCESSED_DIR / "meta.db"

# EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
# EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"

LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/v1/chat/completions")
LLM_API_KEY = os.getenv("LLM_API_KEY", "dummy")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.2:1b")

TOP_K = int(os.getenv("TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))