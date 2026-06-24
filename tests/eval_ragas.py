"""
RAGAS 기반 RAG 파이프라인 정량 평가
실행: venv\Scripts\python.exe tests/eval_ragas.py
"""
import sys
import json
import urllib.request

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

# ── 평가용 질문 + 정답(ground truth) ──────────────────────────────────────
EVAL_SET = [
    {
        "question": "sample1에서 기본 대출한도가 얼마야?",
        "ground_truth": "기본 대출한도는 2.5억원이며, 신혼가구는 2.7억원, 다자녀가구는 3.1억원까지 가능합니다.",
    },
    {
        "question": "sample1에서 대출 최장 만기는 몇 년이야?",
        "ground_truth": "대출 최장 만기는 30년입니다.",
    },
    {
        "question": "sample1에서 LTV 비율은 얼마야?",
        "ground_truth": "LTV(담보인정비율)는 70%입니다.",
    },
    {
        "question": "sample1에서 조기상환수수료율은 얼마야?",
        "ground_truth": "조기상환수수료율은 1.2%입니다.",
    },
    {
        "question": "sample1에서 근저당권 설정비율은 얼마야?",
        "ground_truth": "근저당권 설정비율은 채권최고액의 110% 이상입니다.",
    },
    {
        "question": "sample3에서 회사명이 뭐야?",
        "ground_truth": "(주)에이블씨엔씨입니다.",
    },
    {
        "question": "sample3에서 배당기준일이 언제야?",
        "ground_truth": "배당기준일은 2024년 04월 01일입니다.",
    },
    {
        "question": "sample3에서 1주당 배당금은 얼마야?",
        "ground_truth": "1주당 배당금은 157원입니다.",
    },
    {
        "question": "sample3에서 배당금 총액은 얼마야?",
        "ground_truth": "배당금 총액은 4,084,222,806원입니다.",
    },
    {
        "question": "sample1에서 대출 신청자의 나이 제한이 있어?",
        "ground_truth": "문서에서 관련 내용을 찾지 못했습니다.",
    },
]


def call_api(question: str) -> dict:
    """RAG 서버 호출 → answer + contexts 반환"""
    body = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:8000/ask",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


def collect_results() -> tuple[list, list, list, list]:
    questions, answers, contexts_list, ground_truths = [], [], [], []

    print("=" * 60)
    print("RAG 응답 수집 중...")
    print("=" * 60)

    for i, item in enumerate(EVAL_SET, 1):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"\n[{i}/{len(EVAL_SET)}] {q}")
        try:
            resp = call_api(q)
            answer = resp.get("answer", "")
            sources = resp.get("sources", [])
            contexts = [s["text"] for s in sources if s.get("text")]
            print(f"  → {answer[:80]}{'...' if len(answer) > 80 else ''}")
        except Exception as e:
            print(f"  → ERROR: {e}")
            answer = ""
            contexts = []

        questions.append(q)
        answers.append(answer)
        contexts_list.append(contexts if contexts else ["관련 문서 없음"])
        ground_truths.append(gt)

    return questions, answers, contexts_list, ground_truths


def run_ragas(questions, answers, contexts_list, ground_truths):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    print("\n" + "=" * 60)
    print("RAGAS 평가 시작 (judge: llama3.2:3b) — 순차 실행, 시간이 걸립니다")
    print("=" * 60)

    llm = ChatOllama(
        model="llama3.2:3b",
        base_url="http://127.0.0.1:11434",
        timeout=300,
    )
    embeddings = OllamaEmbeddings(
        model="llama3.2:3b", base_url="http://127.0.0.1:11434"
    )
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    run_config = RunConfig(
        timeout=600,
        max_retries=1,
        max_wait=10,
        max_workers=1,   # 순차 실행 (로컬 LLM 동시성 제한)
    )

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
    )

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
        raise_exceptions=False,
    )

    return result


def print_report(result, questions, answers, ground_truths):
    print("\n" + "=" * 60)
    print("평가 결과 요약")
    print("=" * 60)

    def _get(key):
        try:
            return float(result[key])
        except Exception:
            return None

    scores = {
        "Faithfulness     (답변이 문서 근거했는가)": _get("faithfulness"),
        "Answer Relevancy (답변이 질문에 관련있는가)": _get("answer_relevancy"),
        "Context Precision(검색 문서가 유용했는가)": _get("context_precision"),
        "Context Recall   (필요한 문서 다 가져왔는가)": _get("context_recall"),
    }

    for name, score in scores.items():
        bar = "█" * int((score or 0) * 20) + "░" * (20 - int((score or 0) * 20))
        val = f"{score:.3f}" if score is not None else "N/A "
        print(f"  {name}: {val} [{bar}]")

    valid = [v for v in scores.values() if v is not None]
    if valid:
        print(f"\n  종합 평균: {sum(valid)/len(valid):.3f}")

    print("\n" + "=" * 60)
    print("질문별 상세 결과")
    print("=" * 60)
    df = result.to_pandas()
    for i, row in df.iterrows():
        q = questions[i] if i < len(questions) else ""
        ans = answers[i] if i < len(answers) else ""
        gt = ground_truths[i] if i < len(ground_truths) else ""
        print(f"\n  Q{i+1}: {q}")
        print(f"  정답: {gt[:60]}")
        print(f"  응답: {ans[:60]}")
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            val = row.get(metric)
            print(f"    {metric}: {val:.3f}" if val is not None else f"    {metric}: N/A")


if __name__ == "__main__":
    questions, answers, contexts_list, ground_truths = collect_results()
    result = run_ragas(questions, answers, contexts_list, ground_truths)
    print_report(result, questions, answers, ground_truths)
