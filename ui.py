import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG 문서 검색 시스템",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 RAG 기반 지능형 문서 검색")
st.caption("자연어로 질문하면 PDF 문서에서 답변을 찾아드립니다.")


def render_sources(sources: list, msg_idx: int = 0):
    if not sources:
        return
    with st.expander(f"📄 근거 문서 ({len(sources)}개 청크)", expanded=False):
        for i, src in enumerate(sources):
            rank = src.get("rank", "?")
            source = src.get("source", "알 수 없음")
            page = src.get("page")
            score = src.get("score", 0)
            text = src.get("text", "")

            page_str = f" · p.{page}" if page else ""
            st.markdown(f"**[{rank}위]** `{source}`{page_str} — RRF 점수 `{score:.4f}` / max `0.0328`")
            st.text_area(
                label="",
                value=text,
                height=90,
                key=f"src_{msg_idx}_{i}_{rank}",
                disabled=True,
                label_visibility="collapsed",
            )


# --- 사이드바 ---
with st.sidebar:
    st.header("⚙️ 설정")
    top_k = st.slider("검색 청크 수 (top-k)", min_value=1, max_value=10, value=3)

    st.divider()

    # 서버 상태
    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        st.success("🟢 서버 연결됨")
    except Exception:
        st.error("🔴 서버 연결 안됨\n\n`uvicorn app.api:app` 을 실행하세요.")

    st.divider()
    st.markdown(
        "**기술 스택**\n\n"
        "- Hybrid Retrieval (Dense + Sparse + RRF)\n"
        "- 임베딩: multilingual-e5-base\n"
        "- LLM: Ollama llama3.2\n"
        "- OCR: PaddleOCR (한글 지원)"
    )

    st.divider()
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()


# --- 대화 이력 초기화 ---
if "history" not in st.session_state:
    st.session_state["history"] = []

# --- 이전 대화 렌더 ---
for msg_idx, msg in enumerate(st.session_state["history"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"], msg_idx=msg_idx)
        if msg["role"] == "assistant" and msg.get("elapsed"):
            st.caption(f"⏱ {msg['elapsed']:.1f}초 · 검색된 청크 {msg.get('matched', 0)}개")

# --- 질문 입력 ---
question = st.chat_input("질문을 입력하세요 (예: 대출 한도가 얼마야?)")

if question:
    st.session_state["history"].append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("문서를 검색하고 답변을 생성 중..."):
            start = time.time()
            try:
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question, "top_k": top_k},
                    timeout=60,
                )
                elapsed = time.time() - start

                if resp.ok:
                    data = resp.json()
                    answer = data.get("answer", "답변을 생성하지 못했습니다.")
                    sources = data.get("sources", [])
                    matched = data.get("matched_count", 0)

                    st.markdown(answer)
                    render_sources(sources, msg_idx=len(st.session_state["history"]))
                    st.caption(f"⏱ {elapsed:.1f}초 · 검색된 청크 {matched}개")

                    st.session_state["history"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "elapsed": elapsed,
                        "matched": matched,
                    })
                else:
                    err = f"서버 오류 ({resp.status_code}): {resp.text}"
                    st.error(err)
                    st.session_state["history"].append({
                        "role": "assistant", "content": err, "sources": []
                    })

            except requests.exceptions.ConnectionError:
                err = "❌ FastAPI 서버에 연결할 수 없습니다. `uvicorn app.api:app --reload` 를 먼저 실행하세요."
                st.error(err)
                st.session_state["history"].append({
                    "role": "assistant", "content": err, "sources": []
                })
            except Exception as e:
                err = f"예기치 않은 오류: {e}"
                st.error(err)
                st.session_state["history"].append({
                    "role": "assistant", "content": err, "sources": []
                })

    st.rerun()
