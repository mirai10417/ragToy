import re
import pandas as pd
from app.indexer import load_chunks_parquet, load_faiss_index
from app.embed import embed_query


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).lower()


def normalize_price_text(text: str) -> str:
    text = str(text)
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    return text.lower()


def extract_price(question: str) -> str | None:
    match = re.search(r"(\d[\d,]*)\s*원?", question)
    if not match:
        return None
    return match.group(1).replace(",", "")


def extract_source_filter(question: str) -> str | None:
    match = re.search(r"([a-zA-Z0-9_\-]+\.pdf)", question, re.IGNORECASE)
    if match:
        return match.group(1)

    match = re.search(r"(sample\d+)", question, re.IGNORECASE)
    if match:
        return f"{match.group(1)}.pdf"

    return None


def get_question_keywords(question: str) -> list[str]:
    q = question

    # 파일명 제거
    q = re.sub(r"([a-zA-Z0-9_\-]+\.pdf)", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"(sample\d+)", " ", q, flags=re.IGNORECASE)

    # 불용어 제거
    stopwords = [
        "에서", "의", "를", "을", "은", "는", "이", "가",
        "얼마야", "뭐야", "무엇", "알려줘", "있어", "짜리", "인가", "이야", "?"
    ]
    for sw in stopwords:
        q = q.replace(sw, " ")

    return [x.strip() for x in q.split() if x.strip()]


def compute_keyword_score(text: str, keywords: list[str], target_price: str | None = None) -> int:
    score = 0
    norm_text = normalize_text(text)
    norm_price_text = normalize_price_text(text)

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

    # 4) 질문에 가격이 있으면 강하게 반영
    if target_price and target_price in norm_price_text:
        score += 10

    # 5) 너무 짧은 텍스트는 약간 감점
    # 단, 가격이 일치하는 경우는 감점을 약하게
    if len(text) < 30:
        if target_price and target_price in norm_price_text:
            score -= 1
        else:
            score -= 2

    return score


def retrieve(question: str, top_k: int = 3):
    df = load_chunks_parquet()
    if df.empty:
        return []

    source_filter = extract_source_filter(question)
    target_price = extract_price(question)

    working_df = df.copy()

    if source_filter:
        filtered = working_df[
            working_df["source"].fillna("").str.lower() == source_filter.lower()
        ].copy()
        if not filtered.empty:
            working_df = filtered

    keywords = get_question_keywords(question)

    working_df["score"] = working_df["text"].apply(
        lambda t: compute_keyword_score(t, keywords, target_price)
    )

    keyword_hits = working_df[working_df["score"] > 0].copy()

    sort_columns = ["score"]
    ascending_values = [False]

    if "page" in keyword_hits.columns:
        sort_columns.append("page")
        ascending_values.append(True)

    if "chunk_index" in keyword_hits.columns:
        sort_columns.append("chunk_index")
        ascending_values.append(True)

    if not keyword_hits.empty:
        keyword_hits = keyword_hits.sort_values(
            by=sort_columns,
            ascending=ascending_values
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