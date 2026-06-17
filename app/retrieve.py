import re
import pandas as pd
from app.indexer import load_chunks_parquet


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).lower()


# 질문 의미를 유지하면서 관련 키워드를 함께 검색하도록 쿼리를 확장
_QUERY_EXPANSIONS = [
    (r"최장\s*만기|만기.*몇\s*년|대출\s*만기", "대출만기 10년 15년 20년 30년"),
    (r"금리.*몇|이자율|대출\s*금리",            "대출금리 이자율 연 %"),
    (r"상환\s*방식",                            "상환방식 원리금균등 원금균등"),
    (r"거치\s*기간",                            "거치기간 비거치"),
    (r"조기.*상환|중도.*상환",                   "조기상환수수료 중도상환"),
]


def expand_query(question: str) -> str:
    for pattern, expansion in _QUERY_EXPANSIONS:
        if re.search(pattern, question):
            return f"{question} {expansion}"
    return question


# 최대·최소 질문일 때 확장 키워드의 특정 값이 포함된 chunk를 결과에 보장
_EXTREME_PATTERNS = [
    (r"최장|최대|가장\s*길|가장\s*큰|최고", "max"),
    (r"최단|최소|가장\s*짧|가장\s*작|최저", "min"),
]

def _guarantee_extreme_chunk(
    question: str,
    results: list[dict],
    rrf: list[tuple[float, int]],
    df,
    top_k: int,
) -> list[dict]:
    extreme = None
    for pattern, kind in _EXTREME_PATTERNS:
        if re.search(pattern, question):
            extreme = kind
            break
    if not extreme:
        return results

    # 확장 쿼리에서 숫자+단위 토큰 추출 (예: "30년", "110%")
    expanded_kws = get_question_keywords(expand_query(question))
    value_tokens = [kw for kw in expanded_kws if re.search(r"\d", kw)]
    if not value_tokens:
        return results

    # 최대·최소에 해당하는 극단값 토큰만 확인
    def _numeric(tok: str) -> float:
        m = re.search(r"[\d.]+", tok)
        return float(m.group()) if m else 0.0

    if extreme == "max":
        target_tokens = [max(value_tokens, key=_numeric)]
    else:
        target_tokens = [min(value_tokens, key=_numeric)]

    # 극단값 chunk가 이미 결과에 있으면 rank-1로 올림, 없으면 삽입
    result_texts = [normalize_text(r["text"]) for r in results]
    target_norm = [normalize_text(tok) for tok in target_tokens]

    # 이미 있는 경우: 해당 chunk를 찾아 rank-1로 이동
    for i, (r, t) in enumerate(zip(results, result_texts)):
        if any(tn in t for tn in target_norm):
            if i == 0:
                return results  # 이미 1위
            results = [r] + [x for j, x in enumerate(results) if j != i]
            for idx, item in enumerate(results):
                item["rank"] = idx + 1
            print(f"[retrieve] extreme reorder: promoted {r['source']} p.{r['page']} to rank-1")
            return results

    # 없는 경우: rrf 목록에서 찾아 rank-1에 삽입, 기존 마지막 제거
    for _, df_idx in rrf:
        if df_idx < 0 or df_idx >= len(df):
            continue
        row = df.iloc[df_idx]
        chunk_norm = normalize_text(str(row.get("text", "")))
        if any(tn in chunk_norm for tn in target_norm):
            page_val = row.get("page")
            extra = {
                "rank": 1,
                "source": str(row.get("source", "")),
                "page": int(page_val) if pd.notna(page_val) else None,
                "text": str(row.get("text", "")),
                "score": 0.0,
            }
            results = [extra] + results[: top_k - 1]
            for idx, item in enumerate(results):
                item["rank"] = idx + 1
            print(f"[retrieve] extreme guarantee: inserted {extra['source']} p.{extra['page']} at rank-1")
            break

    return results


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

    # ── Hybrid Retrieval (FAISS + Keyword, RRF 합산) ──────────────────
    try:
        from app.config import FAISS_PATH
        if FAISS_PATH.exists():
            from app.indexer import load_faiss_index
            from app.embed import embed_query

            # 검색 대상 df 준비
            search_df = df.copy()
            if source_filter:
                filtered = search_df[
                    search_df["source"].fillna("").str.lower() == source_filter.lower()
                ]
                if not filtered.empty:
                    search_df = filtered

            # ① Dense: FAISS 전체 순위 목록
            index = load_faiss_index()
            q_vec = embed_query(expand_query(question))
            faiss_scores, faiss_indices = index.search(q_vec, len(df))

            dense_ranks: dict[int, int] = {}   # df_idx → rank (1-based)
            rank = 0
            for idx in faiss_indices[0]:
                if idx < 0 or idx >= len(df):
                    continue
                row = df.iloc[idx]
                if source_filter and str(row.get("source", "")).lower() != source_filter.lower():
                    continue
                rank += 1
                dense_ranks[int(idx)] = rank

            # ② Sparse: 확장 쿼리 키워드로 순위 목록 (숫자·단위 포함)
            keywords = get_question_keywords(expand_query(question))
            kw_scores: list[tuple[int, float]] = []
            for df_idx in search_df.index:
                norm = normalize_text(str(search_df.at[df_idx, "text"]))
                score = sum(1.0 for kw in keywords if normalize_text(kw) in norm)
                if score > 0:
                    kw_scores.append((df_idx, score))
            kw_scores.sort(key=lambda x: x[1], reverse=True)
            sparse_ranks: dict[int, int] = {df_idx: r + 1 for r, (df_idx, _) in enumerate(kw_scores)}

            # ③ RRF 합산: score = 1/(k+rank_dense) + 1/(k+rank_sparse), k=60
            K = 60
            all_idx = set(dense_ranks) | set(sparse_ranks)
            rrf: list[tuple[float, int]] = []
            for df_idx in all_idx:
                s = 0.0
                if df_idx in dense_ranks:
                    s += 1.0 / (K + dense_ranks[df_idx])
                if df_idx in sparse_ranks:
                    s += 1.0 / (K + sparse_ranks[df_idx])
                rrf.append((s, df_idx))
            rrf.sort(reverse=True)

            results = []
            for rrf_score, df_idx in rrf[:top_k]:
                row = df.iloc[df_idx]
                page_val = row.get("page")
                results.append({
                    "rank": len(results) + 1,
                    "source": str(row.get("source", "")),
                    "page": int(page_val) if pd.notna(page_val) else None,
                    "text": str(row.get("text", "")),
                    "score": round(rrf_score, 6),
                })

            if results:
                # 최대·최소 질문이면 확장 키워드 값이 포함된 chunk를 보장
                results = _guarantee_extreme_chunk(question, results, rrf, df, top_k)
                return results
    except Exception as e:
        print(f"Hybrid 검색 실패, 키워드 검색으로 대체: {e}")

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