from pathlib import Path
import uuid

# PaddlePaddle(OCR)가 DLL 환경을 변경하기 전에 torch를 메인 스레드에서 미리 로드.
# Prefect 태스크는 스레드에서 실행되는데, 스레드 안에서 첫 import torch 시
# [WinError 127] shm.dll 오류가 발생함. 미리 import해두면 sys.modules에
# 캐시되어 스레드에서도 재로드 없이 사용 가능.
import torch  # noqa: F401

from prefect import flow, task
from app.config import RAW_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from app.db import init_db, get_conn
from app.parse import parse_pdf
from app.chunk import semantic_chunk
from app.indexer import build_faiss_index, save_chunks_parquet


@task
def list_pdf_files():
    return list(Path(RAW_DIR).glob("*.pdf"))


@task
def process_files(files):
    all_rows = []

    conn = get_conn()
    cur = conn.cursor()

    for file_path in files:
        doc_id = str(uuid.uuid4())
        cur.execute(
            "INSERT OR REPLACE INTO documents (doc_id, filename) VALUES (?, ?)",
            (doc_id, file_path.name)
        )

        elements = parse_pdf(str(file_path))
        chunks = semantic_chunk(elements, max_length=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())

            chunk_text = chunk["text"]
            chunk_source = chunk.get("source", file_path.name)
            chunk_page = chunk.get("page")

            cur.execute(
                "INSERT OR REPLACE INTO chunks (chunk_id, doc_id, chunk_index, text) VALUES (?, ?, ?, ?)",
                (chunk_id, doc_id, i, chunk_text)
            )

            all_rows.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_index": i,
                "filename": file_path.name,
                "source": chunk_source,
                "page": chunk_page,
                "text": chunk_text,
            })

    conn.commit()
    conn.close()
    return all_rows


@task
def embed_and_index(rows):
    # parquet은 항상 저장 (키워드 검색 fallback 보장)
    save_chunks_parquet(rows)

    try:
        from app.embed import embed_texts
        texts = [r["text"] for r in rows]
        embeddings = embed_texts(texts)
        build_faiss_index(embeddings)
        print(f"벡터 인덱스 빌드 완료: {len(rows)}개 청크")
    except Exception as e:
        print(f"[경고] 벡터 인덱스 빌드 실패 (키워드 검색으로 동작): {e}")


@flow(name="ingest-documents")
def ingest_flow():
    init_db()
    files = list_pdf_files()
    rows = process_files(files)

    embed_and_index(rows)


if __name__ == "__main__":
    ingest_flow()
