import re
from fastapi import FastAPI
from app.schemas import AskRequest, AskResponse, SourceChunk
from app.retrieve import retrieve

app = FastAPI(title="RAG Toy MVP")


@app.get("/health")
def health():
    return {"status": "ok"}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_label(text: str) -> str:
    return normalize_space(text).replace(" ", "")


def clean_value(text: str) -> str:
    text = normalize_space(text)
    text = re.split(r"\s+\d+\.\s*", text)[0]
    text = re.split(r"\s*※", text)[0]
    return text.strip(" ,.:;-")


def get_question_keywords(question: str) -> list[str]:
    q = normalize_space(question)

    q = re.sub(r"sample\d+\.pdf", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"sample\d+", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"[a-zA-Z0-9_\-]+\.pdf", " ", q, flags=re.IGNORECASE)

    stopwords = [
        "에서", "의", "를", "을", "은", "는", "이", "가",
        "언제야", "언제", "뭐야", "무엇", "알려줘", "있어",
        "인가", "어디야", "어디", "어느", "몇년이야", "몇년",
        "얼마야", "얼마", "?", "좀", "혹시", "설명해줘",
        "설명", "쉽게", "간단히", "한줄로", "무슨", "뭔지",
        "대해", "관련", "내용"
    ]

    for word in stopwords:
        q = q.replace(word, " ")

    return [x.strip() for x in q.split() if x.strip()]


def extract_value_by_label(text: str, label: str) -> str | None:
    raw_text = normalize_space(text)

    pattern = re.escape(label)
    m = re.search(pattern + r"\s*[:：]?\s*([^\n]+)", raw_text)

    if not m:
        compact_text = normalize_label(raw_text)
        compact_label = normalize_label(label)

        pos = compact_text.find(compact_label)
        if pos == -1:
            return None

        raw_pos = raw_text.find(label)
        if raw_pos == -1:
            return None

        value = raw_text[raw_pos + len(label):]
    else:
        value = m.group(1)

    stop_patterns = [
        r"\s+\d+\.\s+[가-힣A-Za-z]",
        r"\s+[0-9]+\s+[가-힣A-Za-z]+\(",
        r"\s+※",
    ]

    end = len(value)
    for sp in stop_patterns:
        sm = re.search(sp, value)
        if sm:
            end = min(end, sm.start())

    value = clean_value(value[:end])

    return value if value else None


def guess_label_candidates(question: str) -> list[str]:
    q = normalize_space(question)

    labels = [
        "대출한도",
        "대출만기",
        "거치기간",
        "상환방식",
        "배당기준일",
        "배당금지급 예정일자",
        "이사회결의일(결정일)",
        "배당금총액",
        "1 주당 배당금(원)",
        "시가배당율(%)",
        "회사명",
        "대표자",
        "전화번호",
        "주소",
    ]

    keywords = get_question_keywords(q)
    if keywords:
        labels.insert(0, "".join(keywords))
        labels.insert(0, " ".join(keywords))

    return list(dict.fromkeys(labels))


def extract_company_name(text: str) -> str | None:
    raw = normalize_space(text)

    patterns = [
        r"(\(주\)\s*[가-힣A-Za-z0-9]+)\s+대표",
        r"(주식회사\s*[가-힣A-Za-z0-9]+)\s+대표",
        r"회사명\s*[:：]?\s*([가-힣A-Za-z0-9\(\)주식회사\s]+)",
        r"(\(주\)\s*[가-힣A-Za-z0-9]+)",
        r"(주식회사\s*[가-힣A-Za-z0-9]+)",
    ]

    for pattern in patterns:
        m = re.search(pattern, raw)
        if m:
            return clean_value(m.group(1))

    return None


def extract_person_name(text: str) -> str | None:
    raw = normalize_space(text)

    patterns = [
        r"대표집행임원\s+([가-힣]{2,4})",
        r"대표이사\s+([가-힣]{2,4})",
        r"대표자\s*[:：]?\s*([가-힣]{2,4})",
        r"대표\s+([가-힣]{2,4})",
    ]

    for pattern in patterns:
        m = re.search(pattern, raw)
        if m:
            return m.group(1).strip()

    return None


def extract_date_candidates(text: str) -> list[str]:
    raw = normalize_space(text)
    return re.findall(r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일", raw)


def extract_sentence_answer(question: str, retrieved: list[dict]) -> str:
    keywords = get_question_keywords(question)

    for r in retrieved[:3]:
        text = normalize_space(r["text"])

        for kw in keywords:
            idx = text.find(kw)
            if idx != -1:
                start = max(0, idx - 80)
                end = min(len(text), idx + 220)
                return text[start:end].strip(" ,.:;-")

    if retrieved:
        return normalize_space(retrieved[0]["text"])[:300]

    return "문서에서 관련 내용을 찾지 못했습니다."


def make_answer(question: str, retrieved: list[dict]) -> str:
    if not retrieved:
        return "문서에서 관련 내용을 찾지 못했습니다."

    q = normalize_space(question)

    for r in retrieved[:3]:
        text = r["text"]

        if any(k in q for k in ["회사", "회사명", "어느 회사", "어디 회사"]):
            company = extract_company_name(text)
            if company:
                return f"해당 회사는 {company}입니다."

        if any(k in q for k in ["대표", "대표자", "누구"]):
            person = extract_person_name(text)
            if person:
                return f"대표자는 {person}입니다."

        if "배당기준일" in q:
            value = extract_value_by_label(text, "배당기준일")
            if value:
                return f"배당기준일은 {value}입니다."

        if "지급" in q:
            value = extract_value_by_label(text, "배당금지급 예정일자")
            if value:
                return f"배당금지급 예정일자는 {value}입니다."

        if "결의일" in q or "결정일" in q:
            value = extract_value_by_label(text, "이사회결의일(결정일)")
            if value:
                return f"이사회결의일(결정일)은 {value}입니다."

        if "대출한도" in q or "한도" in q:
            value = extract_value_by_label(text, "대출한도")
            if value:
                return f"대출한도는 {value}입니다."

        if "대출만기" in q:
            value = extract_value_by_label(text, "대출만기")
            if value:
                return f"대출만기는 {value}입니다."

        if "거치기간" in q:
            value = extract_value_by_label(text, "거치기간")
            if value:
                return f"거치기간은 {value}입니다."

        if "상환방식" in q:
            value = extract_value_by_label(text, "상환방식")
            if value:
                return f"상환방식은 {value}입니다."

        if "배당금총액" in q:
            value = extract_value_by_label(text, "배당금총액")
            if value:
                return f"배당금총액은 {value}입니다."

    for r in retrieved[:3]:
        for label in guess_label_candidates(question):
            value = extract_value_by_label(r["text"], label)
            if value:
                return f"{label}는 {value}입니다."

    return extract_sentence_answer(question, retrieved)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    retrieved = retrieve(req.question, req.top_k)
    sources = [SourceChunk(**r) for r in retrieved]

    if not retrieved:
        return AskResponse(
            question=req.question,
            answer="문서에서 관련 내용을 찾지 못했습니다.",
            matched_count=0,
            sources=[]
        )

    # LLM으로 답변 생성 (우선)
    try:
        from app.llm import generate_answer
        contexts = [r["text"] for r in retrieved[:3]]
        answer = generate_answer(req.question, contexts)
    except Exception as e:
        print(f"LLM 호출 실패, 패턴 기반 추출로 대체: {e}")
        answer = make_answer(req.question, retrieved)

    return AskResponse(
        question=req.question,
        answer=answer,
        matched_count=len(sources),
        sources=sources
    )