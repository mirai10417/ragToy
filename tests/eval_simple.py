"""
LLM Judge 없이 ground truth 기반 RAG 정확도 평가
실행: python tests/eval_simple.py
"""
import sys
import re
import json
import urllib.request

sys.path.insert(0, ".")

# ── 평가 데이터셋 ─────────────────────────────────────────────────────────────
EVAL_SET = [
    # ── sample1: 수치 ─────────────────────────────────────────────────────────
    {
        "id": "Q01", "category": "수치",
        "question": "sample1에서 기본 대출한도가 얼마야?",
        "ground_truth": "2.5억원",
        "keywords": ["2.5억", "2억5"],
    },
    {
        "id": "Q02", "category": "수치",
        "question": "sample1에서 대출 최장 만기는 몇 년이야?",
        "ground_truth": "30년",
        "keywords": ["30년"],
    },
    {
        "id": "Q03", "category": "수치",
        "question": "sample1에서 LTV 비율은 얼마야?",
        "ground_truth": "70%",
        "keywords": ["70%", "70 %"],
    },
    {
        "id": "Q04", "category": "수치",
        "question": "sample1에서 조기상환수수료율은 얼마야?",
        "ground_truth": "1.2%",
        "keywords": ["1.2%", "1.2 %"],
    },
    {
        "id": "Q05", "category": "수치",
        "question": "sample1에서 근저당권 설정비율은 얼마야?",
        "ground_truth": "110%",
        "keywords": ["110%", "110 %"],
    },
    {
        "id": "Q06", "category": "수치",
        "question": "sample1에서 거치기간은 얼마야?",
        "ground_truth": "1년",
        "keywords": ["1년"],
    },
    # ── sample1: 조건부 ───────────────────────────────────────────────────────
    {
        "id": "Q07", "category": "조건부",
        "question": "sample1에서 신혼가구 대출한도는 얼마야?",
        "ground_truth": "2.7억원",
        "keywords": ["2.7억", "2억7"],
    },
    {
        "id": "Q08", "category": "조건부",
        "question": "sample1에서 다자녀가구 대출한도는 얼마야?",
        "ground_truth": "3.1억원",
        "keywords": ["3.1억", "3억1"],
    },
    # ── sample1: 열거 ─────────────────────────────────────────────────────────
    {
        "id": "Q09", "category": "열거",
        "question": "sample1에서 상환방식 종류가 뭐야?",
        "ground_truth": "원금균등분할상환, 원리금균등분할상환, 체증식분할상환",
        "keywords": ["원금균등", "원리금균등", "체증식"],
    },
    # ── sample1: 동의어 ───────────────────────────────────────────────────────
    {
        "id": "Q10", "category": "동의어",
        "question": "sample1에서 대출 기간이 최대 얼마나 돼?",
        "ground_truth": "30년",
        "keywords": ["30년"],
    },
    {
        "id": "Q11", "category": "동의어",
        "question": "sample1에서 담보인정비율이 몇 퍼센트야?",
        "ground_truth": "70%",
        "keywords": ["70%", "70 %"],
    },
    {
        "id": "Q12", "category": "동의어",
        "question": "sample1에서 중도상환수수료율은?",
        "ground_truth": "1.2%",
        "keywords": ["1.2%", "1.2 %"],
    },
    # ── sample3: 고유명사 ─────────────────────────────────────────────────────
    {
        "id": "Q13", "category": "고유명사",
        "question": "sample3에서 회사명이 뭐야?",
        "ground_truth": "(주)에이블씨엔씨",
        "keywords": ["에이블씨엔씨"],
    },
    # ── sample3: 날짜 ─────────────────────────────────────────────────────────
    {
        "id": "Q14", "category": "날짜",
        "question": "sample3에서 배당기준일이 언제야?",
        "ground_truth": "2024년 04월 01일",
        "keywords": ["2024", "04", "01"],
    },
    {
        "id": "Q15", "category": "날짜",
        "question": "sample3에서 배당금 지급예정일이 언제야?",
        "ground_truth": "2024년 04월 19일",
        "keywords": ["04월", "19일", "4월"],
    },
    # ── sample3: 수치 ─────────────────────────────────────────────────────────
    {
        "id": "Q16", "category": "수치",
        "question": "sample3에서 1주당 배당금은 얼마야?",
        "ground_truth": "157원",
        "keywords": ["157"],
    },
    {
        "id": "Q17", "category": "수치",
        "question": "sample3에서 배당금 총액은 얼마야?",
        "ground_truth": "4,084,222,806원",
        "keywords": ["4,084,222,806", "4084222806"],
    },
    # ── sample3: 동의어 ───────────────────────────────────────────────────────
    {
        "id": "Q18", "category": "동의어",
        "question": "sample3에서 배당금은 주당 얼마야?",
        "ground_truth": "157원",
        "keywords": ["157"],
    },
    {
        "id": "Q19", "category": "동의어",
        "question": "sample3에서 총 배당금액이 얼마야?",
        "ground_truth": "4,084,222,806원",
        "keywords": ["4,084,222,806", "4084222806"],
    },
    # ── sample2: 할루시네이션 방어 ────────────────────────────────────────────
    {
        "id": "Q20", "category": "할루시네이션",
        "question": "sample2에서 대출금리는 얼마야?",
        "ground_truth": "관련 내용 없음",
        "keywords": ["찾지 못했", "없습니다", "관련 내용"],
    },
    {
        "id": "Q21", "category": "할루시네이션",
        "question": "sample2에서 배당금이 얼마야?",
        "ground_truth": "관련 내용 없음",
        "keywords": ["찾지 못했", "없습니다", "관련 내용"],
    },
    # ── 없는 내용 질문 ────────────────────────────────────────────────────────
    {
        "id": "Q22", "category": "할루시네이션",
        "question": "sample1에서 대출 신청자의 나이 제한이 있어?",
        "ground_truth": "관련 내용 없음",
        "keywords": ["찾지 못했", "없습니다", "관련 내용"],
    },
    {
        "id": "Q23", "category": "할루시네이션",
        "question": "sample3에서 대표이사 이름이 뭐야?",
        "ground_truth": "관련 내용 없음",
        "keywords": ["찾지 못했", "없습니다", "관련 내용"],
    },
    {
        "id": "Q24", "category": "할루시네이션",
        "question": "sample1에서 대출 금리는 몇 퍼센트야?",
        "ground_truth": "관련 내용 없음",
        "keywords": ["찾지 못했", "없습니다", "관련 내용"],
    },
    # ── 교차 문서 방어 (다른 문서 내용 묻기) ─────────────────────────────────
    {
        "id": "Q25", "category": "할루시네이션",
        "question": "sample1에서 배당기준일이 언제야?",
        "ground_truth": "관련 내용 없음",
        "keywords": ["찾지 못했", "없습니다", "관련 내용"],
    },
    {
        "id": "Q26", "category": "할루시네이션",
        "question": "sample3에서 LTV 비율은 얼마야?",
        "ground_truth": "관련 내용 없음",
        "keywords": ["찾지 못했", "없습니다", "관련 내용"],
    },
    # ── receipt: 수치 ─────────────────────────────────────────────────────────
    {
        "id": "Q27", "category": "수치",
        "question": "receipt.pdf에서 마늘보쌈 가격은 얼마야?",
        "ground_truth": "20,000원",
        "keywords": ["20000", "20,000"],
    },
    {
        "id": "Q28", "category": "수치",
        "question": "receipt.pdf에서 음료 단가는 얼마야?",
        "ground_truth": "2,000원",
        "keywords": ["2000", "2,000"],
    },
    {
        "id": "Q29", "category": "수치",
        "question": "receipt.pdf에서 총 합계 금액은 얼마야?",
        "ground_truth": "55,000원",
        "keywords": ["55000", "55,000"],
    },
    # ── receipt: 고유명사/날짜 ────────────────────────────────────────────────
    {
        "id": "Q30", "category": "고유명사",
        "question": "receipt.pdf에서 주문번호가 뭐야?",
        "ground_truth": "20210220 01 00037",
        "keywords": ["20210220", "00037"],
    },
]


def call_api(question: str) -> dict:
    body = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:8000/ask",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).lower()


def score_answer(answer: str, keywords: list[str]) -> tuple[bool, list[str]]:
    norm_answer = normalize(answer)
    matched = [kw for kw in keywords if normalize(kw) in norm_answer]
    return len(matched) > 0, matched


def run_eval():
    results = []

    print("=" * 65)
    print(f"RAG 정확도 평가 (ground truth 기반) — 총 {len(EVAL_SET)}개 질문")
    print("=" * 65)

    for item in EVAL_SET:
        q = item["question"]
        print(f"\n[{item['id']}] [{item['category']}] {q}")
        try:
            resp = call_api(q)
            answer = resp.get("answer", "")
            sources = resp.get("sources", [])
        except Exception as e:
            answer = f"ERROR: {e}"
            sources = []

        correct, matched = score_answer(answer, item["keywords"])
        emoji = "✅" if correct else "❌"

        print(f"  정답: {item['ground_truth']}")
        print(f"  응답: {answer[:80]}{'...' if len(answer) > 80 else ''}")
        print(f"  결과: {emoji}  매칭: {matched if matched else '없음'}")

        results.append({
            "id": item["id"],
            "category": item["category"],
            "question": q,
            "ground_truth": item["ground_truth"],
            "answer": answer,
            "correct": correct,
            "sources_count": len(sources),
        })

    return results


def print_report(results: list[dict]):
    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / total

    print("\n" + "=" * 65)
    print("최종 평가 결과")
    print("=" * 65)

    bar = "█" * int(accuracy * 30) + "░" * (30 - int(accuracy * 30))
    print(f"\n  전체 정확도: {correct_count}/{total}  ({accuracy*100:.1f}%)")
    print(f"  [{bar}]")

    # 카테고리별
    print("\n  카테고리별:")
    categories: dict = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, {"total": 0, "correct": 0})
        categories[cat]["total"] += 1
        if r["correct"]:
            categories[cat]["correct"] += 1

    for cat, stat in categories.items():
        ratio = stat["correct"] / stat["total"]
        emoji = "✅" if ratio == 1.0 else "⚠️" if ratio >= 0.5 else "❌"
        print(f"    {emoji} {cat}: {stat['correct']}/{stat['total']} ({ratio*100:.0f}%)")

    # 오답
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print(f"\n  오답 목록 ({len(wrong)}개):")
        for r in wrong:
            print(f"    ❌ [{r['id']}] {r['question']}")
            print(f"       정답: {r['ground_truth']}")
            print(f"       응답: {r['answer'][:60]}{'...' if len(r['answer']) > 60 else ''}")
    else:
        print("\n  모든 질문 정답! 🎉")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    results = run_eval()
    print_report(results)
