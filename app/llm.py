import re
import requests


def _extract_extreme_hint(question: str, contexts: list[str]) -> str | None:
    """최장·최대 또는 최단·최소 질문에서 컨텍스트 숫자 목록의 극단값을 힌트로 반환."""
    is_max = bool(re.search(r"최장|최대|가장\s*길|가장\s*큰|최고", question))
    is_min = bool(re.search(r"최단|최소|가장\s*짧|가장\s*작|최저", question))
    if not (is_max or is_min):
        return None

    # 단위 패턴 추출: "30년", "110%" 등
    unit_pattern = re.compile(r"(\d+(?:\.\d+)?)(년|개월|%|원|억|만)")
    all_values: list[tuple[float, str]] = []
    for ctx in contexts:
        for m in unit_pattern.finditer(ctx):
            all_values.append((float(m.group(1)), m.group(0)))

    if not all_values:
        return None

    # 같은 단위끼리만 비교
    from collections import defaultdict
    by_unit: dict[str, list[float]] = defaultdict(list)
    val_to_str: dict[tuple[str, float], str] = {}
    for val, full in all_values:
        unit = re.sub(r"\d+(?:\.\d+)?", "", full)
        by_unit[unit].append(val)
        val_to_str[(unit, val)] = full

    # 질문 키워드와 관련성 높은 단위 선택 (만기→년, 비율→%)
    preferred = None
    if re.search(r"만기|기간", question):
        preferred = "년"
    elif re.search(r"비율|수수료율|배당율|LTV", question):
        preferred = "%"

    target_unit = preferred if preferred and preferred in by_unit else max(by_unit, key=lambda u: len(by_unit[u]))
    nums = by_unit[target_unit]

    if is_max:
        extreme_val = max(nums)
    else:
        extreme_val = min(nums)

    return f"문서에서 {target_unit} 단위 값들: {sorted(set(nums))} → 그 중 {'최대' if is_max else '최소'}값은 {val_to_str[(target_unit, extreme_val)]}입니다."
from app.config import LLM_API_URL, LLM_API_KEY, LLM_MODEL_NAME

RAG_SYSTEM_PROMPT = """당신은 문서 기반 질의응답 도우미입니다. 아래 규칙을 반드시 지키세요.

규칙:
1. 아래 제공된 [문서 내용]만 근거로 답변하세요. 외부 지식을 사용하지 마세요.
2. 답변은 핵심만 담아 한두 문장으로 간결하게 작성하세요.
3. 답변에 "[문서 1]" 같은 태그나 "Context"라는 단어를 절대 포함하지 마세요.
4. [문서 내용]에 답이 없으면 반드시 "문서에서 관련 내용을 찾지 못했습니다."라고만 답하세요.
5. 숫자·날짜·고유명사는 문서에 나온 그대로 인용하세요.
6. 반드시 한국어로만 답변하세요. 영어, 일본어 등 다른 언어를 절대 사용하지 마세요."""

GENERAL_SYSTEM_PROMPT = """You are a helpful assistant.
Answer the user's question clearly and naturally in Korean.
Do not mention the provided context unless it is actually relevant."""


def generate_answer(question: str, contexts: list[str] | None = None, use_context: bool = True) -> str:
    contexts = contexts or []

    if use_context and contexts:
        system_prompt = RAG_SYSTEM_PROMPT
        context_block = "\n\n".join(
            [f"[문서 {i+1}]\n{c}" for i, c in enumerate(contexts)]
        )
        # 최장·최대·최소·최단 질문이면 힌트 추가
        hint = _extract_extreme_hint(question, contexts)
        hint_text = f"\n\n[힌트] {hint}" if hint else ""
        user_content = f"[문서 내용]\n{context_block}\n\n[질문]\n{question}{hint_text}"
    else:
        system_prompt = GENERAL_SYSTEM_PROMPT
        user_content = question

    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1
    }

    headers = {
        "Content-Type": "application/json",
    }

    if LLM_API_KEY and LLM_API_KEY != "dummy":
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    resp = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=180)
    resp.raise_for_status()

    data = resp.json()
    print("LLM RAW RESPONSE:", data)

    if "choices" in data:
        answer = data["choices"][0]["message"]["content"].strip()
        # 히라가나·가타카나 포함 시 한국어 fallback
        if re.search(r"[ぁ-ゖァ-ヶ]", answer):
            return "문서에서 관련 내용을 찾지 못했습니다."
        return answer

    raise ValueError(f"예상하지 못한 응답 형식: {data}")