import requests
from app.config import LLM_API_URL, LLM_API_KEY, LLM_MODEL_NAME

RAG_SYSTEM_PROMPT = """You are a helpful RAG assistant.
Use the provided context to answer the user's question.
If the answer is clearly not in the context, say exactly: I don't know from the provided documents.
Do not make up facts.
Answer in Korean unless the user asked in another language."""

GENERAL_SYSTEM_PROMPT = """You are a helpful assistant.
Answer the user's question clearly and naturally in Korean.
Do not mention the provided context unless it is actually relevant."""


def generate_answer(question: str, contexts: list[str] | None = None, use_context: bool = True) -> str:
    contexts = contexts or []

    if use_context and contexts:
        system_prompt = RAG_SYSTEM_PROMPT
        context_block = "\n\n".join(
            [f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts)]
        )
        user_content = f"Question:\n{question}\n\nContext:\n{context_block}"
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
        return data["choices"][0]["message"]["content"].strip()

    raise ValueError(f"예상하지 못한 응답 형식: {data}")