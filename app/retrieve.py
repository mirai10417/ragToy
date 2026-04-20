import re
from app.indexer import load_chunks_parquet, load_faiss_index
from app.embed import embed_query


def retrieve(question: str, top_k: int = 3):
    df = load_chunks_parquet()

    source_filter = None
    m = re.search(r"(sample\d+\.pdf|sample\d+)", question, re.IGNORECASE)
    if m:
        source_filter = m.group(1)
        if not source_filter.endswith(".pdf"):
            source_filter += ".pdf"

    # 1) source_filter가 있으면 exact match는 해당 source 안에서 먼저 시도
    working_df = df.copy()
    if source_filter and "source" in df.columns:
        filtered = working_df[
            working_df["source"].fillna("").str.lower() == source_filter.lower()
        ].copy()
        if not filtered.empty:
            working_df = filtered

    # 2) 숫자 exact match 우선 처리
    num_match = re.search(r"\d{4}", question)
    if num_match:
        target_num = num_match.group(0)
        exact_rows = working_df[
            working_df["text"].fillna("").str.contains(target_num, na=False)
        ].copy()

        if not exact_rows.empty:
            results = []
            for rank, (_, row) in enumerate(exact_rows.head(top_k).iterrows(), start=1):
                page_val = row.get("page")
                results.append({
                    "rank": rank,
                    "source": row.get("source"),
                    "page": int(page_val) if page_val is not None else None,
                    "text": row["text"],
                    "score": 1.0,
                })
            return results

    # 3) FAISS 검색
    query_vec = embed_query(question).astype("float32")
    index = load_faiss_index()

    total_rows = len(df)
    if total_rows == 0:
        return []

    # source_filter 후처리 때문에 후보를 넉넉히 조회
    search_k = min(max(top_k * 5, 20), total_rows)
    scores, indices = index.search(query_vec, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        row = df.iloc[idx]

        # source_filter는 검색 후 결과에서 걸러냄
        if source_filter:
            row_source = str(row.get("source", "")).lower()
            if row_source != source_filter.lower():
                continue

        page_val = row.get("page")
        results.append({
            "rank": len(results) + 1,
            "source": row.get("source"),
            "page": int(page_val) if page_val is not None else None,
            "text": row["text"],
            "score": float(score),
        })

        if len(results) >= top_k:
            break

    # source_filter 때문에 top_k를 못 채웠으면 전체 검색 결과라도 반환
    if not results and source_filter:
        fallback_k = min(top_k, total_rows)
        scores, indices = index.search(query_vec, fallback_k)

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            row = df.iloc[idx]
            page_val = row.get("page")
            results.append({
                "rank": len(results) + 1,
                "source": row.get("source"),
                "page": int(page_val) if page_val is not None else None,
                "text": row["text"],
                "score": float(score),
            })

            if len(results) >= top_k:
                break

    return results