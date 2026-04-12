import requests
from app.config import LLM_API_URL, LLM_API_KEY, LLM_MODEL_NAME

SYSTEM_PROMPT = """You are a helpful RAG assistant.
Answer only from the provided context.
If the answer is not in the context, say you don't know."""

def generate_answer(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join([f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts)])

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
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()

    data = resp.json()

    # OpenAI 스타일 예시
    return data["choices"][0]["message"]["content"]