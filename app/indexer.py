import faiss
import numpy as np
import pandas as pd
from app.config import FAISS_PATH, PARQUET_PATH, PROCESSED_DIR

def build_faiss_index(embeddings):
    arr = np.array(embeddings).astype("float32")
    dim = arr.shape[1]
    index = faiss.IndexFlatIP(dim)  # normalize_embeddings=True라 cosine 유사도처럼 사용
    index.add(arr)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_PATH))
    return index

def load_faiss_index():
    return faiss.read_index(str(FAISS_PATH))

def save_chunks_parquet(rows: list[dict]):
    df = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH, index=False)

def load_chunks_parquet():
    return pd.read_parquet(PARQUET_PATH)