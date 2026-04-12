import re
import numpy as np
from app.indexer import load_chunks_parquet
from app.embed import embed_query, embed_texts


def cosine_similarity(a, b):
    return np.dot(a, b.T).flatten()


def retrieve(question: str, top_k: int = 3):
    df = load_chunks_parquet()

    source_filter = None
    m = re.search(r"(sample\d+\.pdf|sample\d+)", question, re.IGNORECASE)
    if m:
        source_filter = m.group(1)
        if not source_filter.endswith(".pdf"):
            source_filter += ".pdf"

    filtered_df = df.copy()
    if source_filter and "source" in df.columns:
        filtered_df = df[
            filtered_df["source"].fillna("").str.lower() == source_filter.lower()
        ].copy()

    if filtered_df.empty:
        filtered_df = df.copy()

    num_match = re.search(r"\d{4}", question)
    if num_match:
        target_num = num_match.group(0)
        exact_rows = filtered_df[
            filtered_df["text"].fillna("").str.contains(target_num, na=False)
        ].copy()

        if not exact_rows.empty:
            results = []
            for rank, (_, row) in enumerate(exact_rows.head(top_k).iterrows(), start=1):
                results.append({
                    "rank": rank,
                    "source": row.get("source"),
                    "page": int(row["page"]) if row.get("page") is not None else None,
                    "text": row["text"],
                    "score": 1.0,
                })
            return results

    texts = filtered_df["text"].fillna("").tolist()
    query_vec = embed_query(question)
    doc_vecs = embed_texts(texts)

    scores = cosine_similarity(query_vec, doc_vecs)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        row = filtered_df.iloc[idx]
        page_val = row.get("page") if "page" in filtered_df.columns else None

        results.append({
            "rank": rank,
            "source": row.get("source"),
            "page": int(page_val) if page_val is not None else None,
            "text": row["text"],
            "score": float(scores[idx]),
        })

    return results