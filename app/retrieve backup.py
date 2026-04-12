import numpy as np
from app.embed import embed_query
from app.indexer import load_faiss_index, load_chunks_parquet

def retrieve(question: str, top_k: int = 4):
    index = load_faiss_index()
    df = load_chunks_parquet()

    q = np.array([embed_query(question)], dtype="float32")
    scores, indices = index.search(q, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        row = df.iloc[idx]
        results.append({
            "chunk_id": row["chunk_id"],
            "doc_id": row["doc_id"],
            "text": row["text"],
            "score": float(score),
        })
    return results