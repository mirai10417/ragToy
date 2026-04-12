from sentence_transformers import SentenceTransformer
from app.config import EMBED_MODEL_NAME

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model

def embed_texts(texts: list[str]):
    model = get_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    return embeddings

def embed_query(text: str):
    model = get_model()
    emb = model.encode([text], normalize_embeddings=True)
    return emb[0]