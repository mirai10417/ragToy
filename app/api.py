import re
from fastapi import FastAPI
from app.schemas import AskRequest, AskResponse, SourceChunk
from app.retrieve import retrieve

app = FastAPI(title="RAG Toy MVP")


@app.get("/health")
def health():
    return {"status": "ok"}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_label(text: str) -> str:
    text = normalize_space(text)
    return text.replace(" ", "")


def normalize_korean_date(text: str) -> str:
    text = normalize_space(text)
    text = re.sub(r"\s*년\s*", "년 ", text)
    text = re.sub(r"\s*월\s*", "월 ", text)
    text = re.sub(r"\s*일\s*", "일", text)
    return text.strip()


def clean_value(text: str) -> str:
    text = normalize_space(text)

    # 번호 항목이 붙어버린 경우 컷
    text = re.split(r"\s+\d+\.\s*", text)[0]
    text = re.split(r"\s+[0-9]+\s+[가-힣A-Za-z]+\s*", text)[0]

    # 불필요하게 길면 1차 정리
    text = re.split(r"\s*※", text)[0]
    text = normalize_korean_date(text)
    return text.strip(" ,.:;-")


def extract_name_answers(retrieved: list[dict], target_num: str) -> str:
    names = []

    for r in retrieved:
        tokens = r["text"].split()

        for i, token in enumerate(tokens):
            if target_num in token and i > 0:
                candidate_name = tokens[i - 1]
                if candidate_name not in names:
                    names.append(candidate_name)

    return ", ".join(names) if names else "검색 결과가 없습니다."


def get_question_keywords(question: str) -> list[str]:
    q = normalize_space(question)
    q = re.sub(r"sample\d+\.pdf", "", q, flags=re.IGNORECASE)
    q = re.sub(r"sample\d+", "", q, flags=re.IGNORECASE)
    q = q.replace("에서", " ")
    q = q.replace("의", " ")
    q = q.replace("를", " ")
    q = q.replace("을", " ")
    q = q.replace("은", " ")
    q = q.replace("는", " ")
    q = q.replace("이", " ")
    q = q.replace("가", " ")
    q = q.replace("언제야", " ")
    q = q.replace("언제", " ")
    q = q.replace("뭐야", " ")
    q = q.replace("무엇", " ")
    q = q.replace("알려줘", " ")
    q = q.replace("있어", " ")
    q = q.replace("인가", " ")
    q = q.replace("?", " ")

    parts = [p.strip() for p in q.split() if p.strip()]
    return parts


def guess_label_candidates(question: str) -> list[str]:
    q = normalize_space(question)

    candidates = []

    # 자주 나오는 꼬리말 패턴 중심으로 항목명 추정
    suffix_patterns = [
        r"([가-힣A-Za-z0-9·\(\)\s]+예정일자)",
        r"([가-힣A-Za-z0-9·\(\)\s]+지급일자)",
        r"([가-힣A-Za-z0-9·\(\)\s]+지급일)",
        r"([가-힣A-Za-z0-9·\(\)\s]+기준일)",
        r"([가-힣A-Za-z0-9·\(\)\s]+결의일(?:\(결정일\))?)",
        r"([가-힣A-Za-z0-9·\(\)\s]+결정일)",
        r"([가-힣A-Za-z0-9·\(\)\s]+총액)",
        r"([가-힣A-Za-z0-9·\(\)\s]+금액)",
        r"([가-힣A-Za-z0-9·\(\)\s]+만기)",
        r"([가-힣A-Za-z0-9·\(\)\s]+기간)",
        r"([가-힣A-Za-z0-9·\(\)\s]+한도)",
        r"([가-힣A-Za-z0-9·\(\)\s]+배당금)",
        r"([가-힣A-Za-z0-9·\(\)\s]+주소)",
        r"([가-힣A-Za-z0-9·\(\)\s]+전화번호)",
        r"([가-힣A-Za-z0-9·\(\)\s]+대표자)",
        r"([가-힣A-Za-z0-9·\(\)\s]+상환방식)",
    ]

    for pattern in suffix_patterns:
        for m in re.finditer(pattern, q):
            value = normalize_space(m.group(1))
            if len(value) >= 2:
                candidates.append(value)

    # 조사 제거 후 남는 짧은 명사구도 후보 추가
    keywords = get_question_keywords(question)
    if keywords:
        merged = " ".join(keywords)
        if len(merged) >= 2:
            candidates.append(merged)

    # 중복 제거
    deduped = []
    seen = set()
    for c in candidates:
        key = normalize_label(c)
        if key and key not in seen:
            deduped.append(c)
            seen.add(key)

    return deduped


def extract_value_by_label(text: str, label: str) -> str | None:
    raw_text = normalize_space(text)
    compact_text = normalize_label(raw_text)
    compact_label = normalize_label(label)

    pos = compact_text.find(compact_label)
    if pos == -1:
        return None

    # compact index -> raw index 대략 매핑
    raw_idx = raw_text.find(label)
    if raw_idx == -1:
        label_tokens = label.split()
        raw_idx = -1
        for token in label_tokens:
            tmp = raw_text.find(token)
            if tmp != -1:
                raw_idx = tmp
                break

    if raw_idx == -1:
        return None

    after = raw_text[raw_idx + len(label):].strip()

    # 콜론 제거
    after = re.sub(r"^[:：]\s*", "", after)

    # 다음 항목 번호가 나오기 전까지만
    stop_patterns = [
        r"\s+\d+\.\s+[가-힣A-Za-z]",
        r"\s+[0-9]+\s+[가-힣A-Za-z]+\(",
        r"\s+[0-9]+\s+[가-힣A-Za-z]{2,}",
        r"\s+※",
    ]

    end = len(after)
    for sp in stop_patterns:
        m = re.search(sp, after)
        if m:
            end = min(end, m.start())

    value = after[:end].strip()
    value = clean_value(value)

    if not value:
        return None

    return value


def extract_sentence_like_answer(text: str, question: str) -> str | None:
    raw = normalize_space(text)
    keywords = get_question_keywords(question)

    if not keywords:
        return None

    best_idx = -1
    best_kw = None

    for kw in keywords:
        idx = raw.find(kw)
        if idx != -1:
            best_idx = idx
            best_kw = kw
            break

    if best_idx == -1:
        return None

    start = max(0, best_idx - 80)
    end = min(len(raw), best_idx + 180)
    snippet = raw[start:end].strip()

    # 앞뒤 자투리 정리
    snippet = snippet.strip(" ,.:;-")
    return snippet if snippet else None


def extract_general_answer(question: str, retrieved: list[dict]) -> str:
    if not retrieved:
        return "검색 결과가 없습니다."

    top_text = retrieved[0]["text"]

    # 1) 질문에서 라벨 후보 추정
    label_candidates = guess_label_candidates(question)

    # 2) 라벨 기반 값 추출
    for label in label_candidates:
        value = extract_value_by_label(top_text, label)
        if value:
            return f"{label}는 {value}입니다."

    # 3) 상위 검색 결과 전체를 훑어 한 번 더 시도
    for r in retrieved[:3]:
        for label in label_candidates:
            value = extract_value_by_label(r["text"], label)
            if value:
                return f"{label}는 {value}입니다."

    # 4) 못 찾으면 질문 관련 스니펫만 반환
    snippet = extract_sentence_like_answer(top_text, question)
    if snippet:
        return snippet

    # 5) 최후 fallback
    return normalize_space(top_text)[:300]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    retrieved = retrieve(req.question, req.top_k)
    sources = [SourceChunk(**r) for r in retrieved]

    num_match = re.search(r"\d{4}", req.question)

    if num_match:
        answer = extract_name_answers(retrieved, num_match.group(0))
    else:
        answer = extract_general_answer(req.question, retrieved)

    return AskResponse(
        question=req.question,
        answer=answer,
        matched_count=len(sources),
        sources=sources
    )