import requests
import json
import sys

question = sys.argv[1] if len(sys.argv) > 1 else "sample1.pdf에서 담보주택의 세부평가 방법 및 적용순서를 말해줘"
top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

res = requests.post("http://localhost:8000/ask", json={"question": question, "top_k": top_k})
data = res.json()

print(f"Q: {data['question']}")
print(f"A: {data['answer']}")
print(f"matched: {data['matched_count']}개")
print()
for s in data["sources"]:
    print(f"  [{s['rank']}] {s['source']} p.{s['page']} (score={s['score']:.3f})")
    print(f"      {s['text'][:120]}...")
    print()
