import numpy as np
import pandas as pd
from app.config import FAISS_PATH, PARQUET_PATH, PROCESSED_DIR


class NumpyIndex:
    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors  # shape: (n, dim)

    def search(self, query: np.ndarray, k: int):
        # 정규화된 벡터이므로 내적 = 코사인 유사도
        scores = (query @ self.vectors.T).astype("float32")  # (1, n)
        top_k = min(k, self.vectors.shape[0])
        idx = np.argsort(-scores[0])[:top_k]
        return scores[:, idx].reshape(1, -1), idx.reshape(1, -1)


def build_faiss_index(embeddings):
    arr = np.array(embeddings).astype("float32")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(FAISS_PATH), arr)
    return NumpyIndex(arr)


def load_faiss_index():
    arr = np.load(str(FAISS_PATH))
    return NumpyIndex(arr)


def save_chunks_parquet(rows: list[dict]):
    df = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH, index=False)


def load_chunks_parquet():
    return pd.read_parquet(PARQUET_PATH)
