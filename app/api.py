import re
from fastapi import FastAPI
from app.schemas import AskRequest, AskResponse, SourceChunk
from app.retrieve import retrieve
from app.llm import generate_answer

app = FastAPI(title="RAG Toy MVP")


@app.get("/health")
def health():
    return {"status": "ok"}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_label(text: str) -> str:
    return normalize_space(text).replace(" ", "")


def normalize_korean_date(text: str) -> str:
    text = normalize_space(text)
    text = re.sub(r"\s*년\s*", "년 ", text)
    text = re.sub(r"\s*월\s*", "월 ", text)
    text = re.sub(r"\s*일\s*", "일", text)
    return text.strip()


def clean_value(text: str) -> str:
    text = normalize_space(text)

    # 다음 번호 항목(예: 2. 대출만기, 3. 상환방식)만 잘라냄
    text = re.split(r"\s+\d+\.\s*", text)[0]

    # 주석 표시는 제거
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

    removable_words = [
        "에서", "의", "를", "을", "은", "는", "이", "가",
        "언제야", "언제", "뭐야", "무엇", "알려줘", "있어",
        "인가", "어디야", "어디", "어느", "몇년이야", "몇년",
        "얼마야", "얼마", "?", "좀", "혹시"
    ]
    for word in removable_words:
        q = q.replace(word, " ")

    parts = [p.strip() for p in q.split() if p.strip()]
    return parts


def guess_label_candidates(question: str) -> list[str]:
    q = normalize_space(question)

    candidates = []

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
        r"([가-힣A-Za-z0-9·\(\)\s]+회사명)",
    ]

    for pattern in suffix_patterns:
        for m in re.finditer(pattern, q):
            value = normalize_space(m.group(1))
            if len(value) >= 2:
                candidates.append(value)

    keywords = get_question_keywords(question)
    if keywords:
        merged = " ".join(keywords)
        if len(merged) >= 2:
            candidates.append(merged)

    # 자주 나오는 핵심 라벨 직접 추가
    direct_candidates = [
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
    ]
    candidates.extend(direct_candidates)

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
    after = re.sub(r"^[:：]\s*", "", after)

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
    for kw in keywords:
        idx = raw.find(kw)
        if idx != -1:
            best_idx = idx
            break

    if best_idx == -1:
        return None

    start = max(0, best_idx - 80)
    end = min(len(raw), best_idx + 180)
    snippet = raw[start:end].strip()
    snippet = snippet.strip(" ,.:;-")
    return snippet if snippet else None


def is_company_question(question: str) -> bool:
    q = normalize_space(question)
    keywords = ["회사", "회사명", "어느 회사", "어디 회사", "기업", "법인"]
    return any(k in q for k in keywords)


def is_person_question(question: str) -> bool:
    q = normalize_space(question)
    keywords = ["누구", "대표", "대표자", "임원", "이름"]
    return any(k in q for k in keywords)


def is_date_question(question: str) -> bool:
    q = normalize_space(question)
    keywords = ["언제", "날짜", "일자", "기준일", "결의일", "결정일", "지급일"]
    return any(k in q for k in keywords)


def is_amount_question(question: str) -> bool:
    q = normalize_space(question)
    keywords = ["얼마", "금액", "총액", "배당금", "한도", "원"]
    return any(k in q for k in keywords)


def extract_company_name(text: str) -> str | None:
    raw = normalize_space(text)

    patterns = [
        r"(\(주\)\s*[가-힣A-Za-z0-9]+)\s+대표",
        r"(주식회사\s*[가-힣A-Za-z0-9]+)\s+대표",
        r"(\(주\)\s*[가-힣A-Za-z0-9]+)",
        r"(주식회사\s*[가-힣A-Za-z0-9]+)",
    ]

    for pattern in patterns:
        m = re.search(pattern, raw)
        if m:
            return m.group(1).strip()

    return None


def extract_person_name(text: str) -> str | None:
    raw = normalize_space(text)

    patterns = [
        r"대표집행임원\s+([가-힣]{2,4})",
        r"대표이사\s+([가-힣]{2,4})",
        r"대표자\s+([가-힣]{2,4})",
        r"대표\s+([가-힣]{2,4})",
    ]

    for pattern in patterns:
        m = re.search(pattern, raw)
        if m:
            return m.group(1).strip()

    return None


def extract_date_candidates(text: str) -> list[str]:
    raw = normalize_space(text)
    matches = re.findall(r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일", raw)
    results = []
    for m in matches:
        date_str = normalize_korean_date(m)
        if date_str not in results:
            results.append(date_str)
    return results


def extract_amount_candidates(text: str) -> list[str]:
    raw = normalize_space(text)

    patterns = [
        r"\d[\d,]*\s*억원",
        r"\d[\d,]*\s*만원",
        r"\d[\d,]*\s*원",
        r"\d[\d,]*\.\d+",
    ]

    values = []
    for pattern in patterns:
        for m in re.findall(pattern, raw):
            cleaned = normalize_space(m)
            if cleaned not in values:
                values.append(cleaned)

    return values


def extract_title_like_text(text: str) -> str | None:
    raw = normalize_space(text)
    # 문서 첫 부분에서 제목처럼 보이는 구간
    m = re.match(r"^([가-힣A-Za-z0-9·\(\)\s]+공고)", raw)
    if m:
        return m.group(1).strip()
    return None


def extract_special_answer(question: str, retrieved: list[dict]) -> str | None:
    if not retrieved:
        return None

    for r in retrieved[:3]:
        text = r["text"]

        if is_company_question(question):
            company = extract_company_name(text)
            if company:
                return f"해당 회사는 {company}입니다."

        if is_person_question(question):
            person = extract_person_name(text)
            if person:
                return f"대표자는 {person}입니다."

        if "제목" in normalize_space(question) or "문서명" in normalize_space(question):
            title = extract_title_like_text(text)
            if title:
                return f"문서 제목은 {title}입니다."

        if is_date_question(question):
            dates = extract_date_candidates(text)
            if dates:
                # 질문 키워드별 우선 처리
                if "배당기준일" in question:
                    value = extract_value_by_label(text, "배당기준일")
                    if value:
                        return f"배당기준일은 {value}입니다."
                if "지급" in question:
                    value = extract_value_by_label(text, "배당금지급 예정일자")
                    if value:
                        return f"배당금지급 예정일자는 {value}입니다."
                if "결의일" in question or "결정일" in question:
                    value = extract_value_by_label(text, "이사회결의일(결정일)")
                    if value:
                        return f"이사회결의일(결정일)은 {value}입니다."

        if is_amount_question(question):
            if "대출한도" in question:
                value = extract_value_by_label(text, "대출한도")
                if value:
                    return f"대출한도는 {value}입니다."
            if "배당금총액" in question:
                value = extract_value_by_label(text, "배당금총액")
                if value:
                    return f"배당금총액은 {value}입니다."

    return None


def extract_general_answer(question: str, retrieved: list[dict]) -> str:
    if not retrieved:
        return "검색 결과가 없습니다."

    # 0) 질문 의도별 특수 추출 먼저
    special_answer = extract_special_answer(question, retrieved)
    if special_answer:
        return special_answer

    # 1) 목차성 chunk는 뒤로 미루기
    non_toc_retrieved = []
    for r in retrieved:
        text = normalize_space(r["text"])
        dot_count = text.count("····")
        if dot_count >= 3 and len(text) < 500:
            continue
        non_toc_retrieved.append(r)

    working_retrieved = non_toc_retrieved if non_toc_retrieved else retrieved
    top_text = working_retrieved[0]["text"]

    # 2) 질문에서 라벨 후보 추정
    label_candidates = guess_label_candidates(question)

    # 3) 라벨 기반 값 추출
    for r in working_retrieved[:3]:
        for label in label_candidates:
            value = extract_value_by_label(r["text"], label)
            if value:
                return f"{label}는 {value}입니다."

    # 4) 대출만기 + 거치기간 동시 질문 보강
    q = normalize_space(question)
    if "대출만기" in q and "거치기간" in q:
        for r in working_retrieved[:3]:
            text = normalize_space(r["text"])
            m = re.search(
                r"대출만기는\s*([^,]+(?:,\s*[^,]+)*)\s*이며,\s*거치기간은\s*([^.]+?)(?:\s*상환방식|$)",
                text
            )
            if m:
                maturity = clean_value(m.group(1))
                grace = clean_value(m.group(2))
                return f"대출만기는 {maturity}이며, 거치기간은 {grace}입니다."

    # 5) 스니펫 반환
    snippet = extract_sentence_like_answer(top_text, question)
    if snippet:
        return snippet

    # 6) 최후 fallback
    return normalize_space(top_text)[:300]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    retrieved = retrieve(req.question, req.top_k)
    sources = [SourceChunk(**r) for r in retrieved]

    if not retrieved:
        return AskResponse(
            question=req.question,
            answer="검색 결과가 없습니다.",
            matched_count=0,
            sources=[]
        )

    num_match = re.search(r"\d{4}", req.question)
    contexts = [r["text"][:700] for r in retrieved[:3]]

    if num_match and not any(k in req.question for k in ["년", "월", "일", "날짜", "일자"]):
        answer = extract_name_answers(retrieved, num_match.group(0))
    else:
        answer = extract_general_answer(req.question, retrieved)

        if not answer or answer in ["검색 결과가 없습니다.", ""]:
            try:
                answer = generate_answer(req.question, contexts)
            except Exception as e:
                print("LLM ERROR:", repr(e))
                answer = f"LLM 오류: {repr(e)}"

    return AskResponse(
        question=req.question,
        answer=answer,
        matched_count=len(sources),
        sources=sources
    )