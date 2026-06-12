import re
import pandas as pd
from app.indexer import load_chunks_parquet


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).lower()


def extract_source_filter(question: str) -> str | None:
    match = re.search(r"([a-zA-Z0-9_\-]+\.pdf)", question, re.IGNORECASE)
    if match:
        return match.group(1)

    match = re.search(r"(sample\d+)", question, re.IGNORECASE)
    if match:
        return f"{match.group(1)}.pdf"

    return None


def get_question_keywords(question: str) -> list[str]:
    q = str(question)

    q = re.sub(r"([a-zA-Z0-9_\-]+\.pdf)", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"(sample\d+)", " ", q, flags=re.IGNORECASE)

    stopwords = [
        "에서", "의", "를", "을", "은", "는", "이", "가",
        "뭐야", "무엇", "알려줘", "있어", "인가", "이야",
        "언제야", "언제", "어디야", "어디",
        "누구야", "누구", "?", "좀", "간단히",
        "설명해줘", "설명", "값", "번호"
    ]

    for sw in stopwords:
        q = q.replace(sw, " ")

    return [x.strip() for x in q.split() if x.strip()]


def is_receipt_order_question(question: str) -> bool:
    q = normalize_text(question)
    return any(k in q for k in ["주문번호", "주문", "영수증번호", "거래번호", "승인번호"])


def is_price_question(question: str) -> bool:
    q = normalize_text(question)
    return any(k in q for k in ["얼마", "가격", "금액", "단가"])


def is_total_question(question: str) -> bool:
    q = normalize_text(question)
    return any(k in q for k in ["총금액", "합계", "총액", "결제금액"])


def get_receipt_full_text(df: pd.DataFrame) -> str:
    if df.empty:
        return ""

    df = df.sort_values(by=["page", "chunk_index"], ascending=True)
    return "\n".join(str(x) for x in df["text"].tolist())


def extract_order_number_from_receipt(full_text: str) -> str | None:
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]

    joined = " ".join(lines)

    # PaddleOCR 결과가 보통 이렇게 분리됨:
    # 20210220
    # 01
    # 00037
    m = re.search(r"(\d{8})\s+(\d{2})\s+(\d{4,6})", joined)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)}"

    # 붙어서 들어오는 경우
    m = re.search(r"(\d{8})(\d{2})(\d{4,6})", joined)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)}"

    return None


def extract_product_price_from_receipt(full_text: str, product_keyword: str) -> str | None:
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]

    for i, line in enumerate(lines):
        if normalize_text(product_keyword) in normalize_text(line):
            window = lines[i:i + 6]
            window_text = " ".join(window)

            prices = re.findall(r"\d{1,3}(?:,\d{3})+|\d{4,6}", window_text)

            if prices:
                return prices[0].replace(",", "")

    return None


def extract_total_from_receipt(full_text: str) -> str | None:
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    joined = " ".join(lines)

    candidates = re.findall(r"\d{1,3}(?:,\d{3})+|\d{4,6}", joined)

    if not candidates:
        return None

    nums = []
    for c in candidates:
        try:
            nums.append(int(c.replace(",", "").replace(" ", "")))
        except Exception:
            pass

    if not nums:
        return None

    return str(max(nums))


def make_receipt_answer(question: str, source: str, df: pd.DataFrame) -> dict | None:
    full_text = get_receipt_full_text(df)

    if not full_text:
        return None

    if is_receipt_order_question(question):
        order_no = extract_order_number_from_receipt(full_text)

        if order_no:
            return {
                "source": source,
                "page": 1,
                "text": f"주문번호는 {order_no}입니다.",
                "score": 100.0
            }

    if is_total_question(question):
        total = extract_total_from_receipt(full_text)

        if total:
            return {
                "source": source,
                "page": 1,
                "text": f"총금액은 {total}원입니다.",
                "score": 90.0
            }

    if is_price_question(question):
        keywords = get_question_keywords(question)

        for kw in keywords:
            price = extract_product_price_from_receipt(full_text, kw)

            if price:
                return {
                    "source": source,
                    "page": 1,
                    "text": f"{kw} 금액은 {price}원입니다.",
                    "score": 80.0
                }

    return None


def _keyword_search(working_df: pd.DataFrame, question: str, top_k: int) -> list[dict]:
    keywords = get_question_keywords(question)
    results = []

    for _, row in working_df.iterrows():
        text = str(row.get("text", ""))
        norm_text = normalize_text(text)

        score = 0.0
        for kw in keywords:
            if normalize_text(kw) in norm_text:
                score += 10.0

        if re.search(r"\d+", question) and re.search(r"\d+", text):
            score += 3.0

        if score <= 0:
            continue

        page_val = row.get("page")
        results.append({
            "rank": 0,
            "source": row.get("source"),
            "page": int(page_val) if pd.notna(page_val) else None,
            "text": text,
            "score": score,
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    final_results = []
    seen = set()
    for item in results:
        key = (item["source"], item["page"], item["text"])
        if key in seen:
            continue
        seen.add(key)
        item["rank"] = len(final_results) + 1
        final_results.append(item)
        if len(final_results) >= top_k:
            break

    return final_results


def retrieve(question: str, top_k: int = 3):
    df = load_chunks_parquet()

    if df.empty:
        return []

    source_filter = extract_source_filter(question)

    # 영수증은 전용 추출 로직 우선 적용
    if source_filter and "receipt" in source_filter.lower():
        receipt_df = df[df["source"].fillna("").str.lower() == source_filter.lower()].copy()
        if not receipt_df.empty:
            receipt_answer = make_receipt_answer(question, source_filter, receipt_df)
            if receipt_answer:
                return [{
                    "rank": 1,
                    "source": receipt_answer["source"],
                    "page": receipt_answer["page"],
                    "text": receipt_answer["text"],
                    "score": receipt_answer["score"],
                }]

    # FAISS 의미 검색 (primary)
    try:
        from app.config import FAISS_PATH
        if FAISS_PATH.exists():
            from app.indexer import load_faiss_index
            from app.embed import embed_query

            index = load_faiss_index()
            q_vec = embed_query(question)

            # source_filter 있을 때는 더 많이 검색 후 필터링
            search_k = len(df) if source_filter else min(top_k * 5, len(df))
            scores, indices = index.search(q_vec, search_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(df):
                    continue
                row = df.iloc[idx]
                row_source = str(row.get("source", ""))
                if source_filter and row_source.lower() != source_filter.lower():
                    continue
                page_val = row.get("page")
                results.append({
                    "rank": 0,
                    "source": row_source,
                    "page": int(page_val) if pd.notna(page_val) else None,
                    "text": str(row.get("text", "")),
                    "score": float(score),
                })
                if len(results) >= top_k:
                    break

            if results:
                for i, r in enumerate(results):
                    r["rank"] = i + 1
                return results
    except Exception as e:
        print(f"FAISS 검색 실패, 키워드 검색으로 대체: {e}")

    # 키워드 검색 fallback
    working_df = df.copy()
    if source_filter:
        filtered = working_df[
            working_df["source"].fillna("").str.lower() == source_filter.lower()
        ].copy()
        if not filtered.empty:
            working_df = filtered

    if working_df.empty:
        return []

    return _keyword_search(working_df, question, top_k)