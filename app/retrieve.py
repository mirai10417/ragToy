import re
import pandas as pd
from app.indexer import load_chunks_parquet, load_faiss_index
from app.embed import embed_query


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).lower()


def get_question_keywords(question: str) -> list[str]:
    q = re.sub(r"sample\d+\.pdf|sample\d+", "", question, flags=re.IGNORECASE)
    stopwords = ["에서", "의", "를", "을", "은", "는", "이", "가", "얼마야", "뭐야", "무엇", "알려줘", "있어", "?"]
    for sw in stopwords:
        q = q.replace(sw, " ")
    return [x.strip() for x in q.split() if x.strip()]


def compute_keyword_score(text: str, keywords: list[str]) -> int:
    score = 0
    norm_text = normalize_text(text)

    # 1) 키워드 매칭
    for kw in keywords:
        if normalize_text(kw) in norm_text:
            score += 3

    # 2) 숫자 포함 → 값일 확률 높음
    if re.search(r"\d+", text):
        score += 3

    # 3) 단위 포함 → 진짜 답일 확률 높음
    if re.search(r"(년|개월|억원|원|%)", text):
        score += 2

    # 4) 너무 짧으면 감점 (목차 방지)
    if len(text) < 30:
        score -= 2

    return score


def retrieve(question: str, top_k: int = 3):
    df = load_chunks_parquet()
    if df.empty:
        return []

    source_filter = None
    m = re.search(r"(sample\d+\.pdf|sample\d+)", question, re.IGNORECASE)
    if m:
        source_filter = m.group(1)
        if not source_filter.endswith(".pdf"):
            source_filter += ".pdf"

    working_df = df.copy()

    if source_filter:
        filtered = working_df[
            working_df["source"].fillna("").str.lower() == source_filter.lower()
        ].copy()
        if not filtered.empty:
            working_df = filtered

    # 🔥 핵심: 키워드 기반 + 점수 계산
    keywords = get_question_keywords(question)

    working_df["score"] = working_df["text"].apply(
        lambda t: compute_keyword_score(t, keywords)
    )

    keyword_hits = working_df[working_df["score"] > 0].copy()

    if not keyword_hits.empty:
        keyword_hits = keyword_hits.sort_values(
            by=["score", "page", "chunk_index"],
            ascending=[False, True, True]
        )

        results = []
        for rank, (_, row) in enumerate(keyword_hits.head(top_k).iterrows(), start=1):
            page_val = row.get("page")
            results.append({
                "rank": rank,
                "source": row.get("source"),
                "page": int(page_val) if pd.notna(page_val) else None,
                "text": row["text"],
                "score": float(row["score"]),
            })
        return results

    # fallback: FAISS
    query_vec = embed_query(question).astype("float32")
    index = load_faiss_index()
    search_k = min(max(top_k * 5, 20), len(df))
    scores, indices = index.search(query_vec, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        row = df.iloc[idx]

        page_val = row.get("page")
        results.append({
            "rank": len(results) + 1,
            "source": row.get("source"),
            "page": int(page_val) if pd.notna(page_val) else None,
            "text": row["text"],
            "score": float(score),
        })

        if len(results) >= top_k:
            break

    return results