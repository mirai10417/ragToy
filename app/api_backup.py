from fastapi import FastAPI
from app.schemas import AskRequest, AskResponse, SourceChunk
from app.retrieve import retrieve
from app.llm import generate_answer

app = FastAPI(title="RAG Toy MVP")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    retrieved = retrieve(req.question, req.top_k)
    answer = generate_answer(req.question, [r["text"] for r in retrieved])

    sources = [SourceChunk(**r) for r in retrieved]
    return AskResponse(answer=answer, sources=sources)