import requests
from app.config import LLM_API_URL, LLM_API_KEY, LLM_MODEL_NAME

SYSTEM_PROMPT = """You are a helpful RAG assistant.
Answer only from the provided context.
If the answer is not in the context, say you don't know."""


def generate_answer(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(
        [f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts)]
    )

    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nContext:\n{context_block}"
            }
        ],
        "temperature": 0.1
    }

    headers = {
        "Content-Type": "application/json",
    }

    # dummy 키면 Authorization 생략
    if LLM_API_KEY and LLM_API_KEY != "dummy":
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    resp = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=180)
    resp.raise_for_status()

    data = resp.json()
    print("LLM RAW RESPONSE:", data)

    if "choices" in data:
        return data["choices"][0]["message"]["content"]

    raise ValueError(f"예상하지 못한 응답 형식: {data}")