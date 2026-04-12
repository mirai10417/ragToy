import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# 고정 차원 벡터 생성기
# sentence-transformers 대신 가벼운 임시 임베딩 역할
_vectorizer = HashingVectorizer(
    n_features=384,
    alternate_sign=False,
    norm="l2"
)


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    여러 텍스트를 벡터로 변환
    return shape: (N, 384), dtype=float32
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    matrix = _vectorizer.transform(texts)
    return matrix.toarray().astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    """
    단일 질의 텍스트를 벡터로 변환
    return shape: (1, 384), dtype=float32
    """
    matrix = _vectorizer.transform([text])
    return matrix.toarray().astype(np.float32)