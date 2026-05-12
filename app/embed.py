import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import EMBED_MODEL_NAME

_model = SentenceTransformer(EMBED_MODEL_NAME)


def _format_passage(text: str) -> str:
    return f"passage: {text}"


def _format_query(text: str) -> str:
    return f"query: {text}"


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    formatted = [_format_passage(t) for t in texts]

    embeddings = _model.encode(
        formatted,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=8,
    )

    return embeddings.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    embedding = _model.encode(
        [_format_query(text)],
        normalize_embeddings=True,
    )

    return embedding.astype(np.float32)