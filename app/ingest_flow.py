from pathlib import Path
import uuid
import numpy as np
import pandas as pd

# PaddlePaddle(OCR)가 DLL 환경을 변경하기 전에 torch를 메인 스레드에서 미리 로드.
# Prefect 태스크는 스레드에서 실행되는데, 스레드 안에서 첫 import torch 시
# [WinError 127] shm.dll 오류가 발생함. 미리 import해두면 sys.modules에
# 캐시되어 스레드에서도 재로드 없이 사용 가능.
import torch  # noqa: F401

from prefect import flow, task
from app.config import RAW_DIR, CHUNK_SIZE, CHUNK_OVERLAP, FAISS_PATH, PARQUET_PATH
from app.db import init_db, get_conn
from app.parse import parse_pdf
from app.chunk import semantic_chunk
from app.indexer import build_faiss_index, save_chunks_parquet, load_chunks_parquet
from app.hash_store import diff_files, load_hashes, save_hashes


@task
def list_pdf_files():
    return list(Path(RAW_DIR).glob("*.pdf"))


@task
def detect_changes(files: list[Path]):
    stored = load_hashes()
    changed, removed, current_hashes = diff_files(files, stored)

    print(f"[ingest] 전체 {len(files)}개 | 변경 {len(changed)}개 | 삭제 {len(removed)}개")
    for f in changed:
        status = "신규" if f.name not in stored else "변경"
        print(f"  [{status}] {f.name}")
    for name in removed:
        print(f"  [삭제] {name}")

    return changed, removed, current_hashes


@task
def process_files(files: list[Path]):
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
def merge_and_index(new_rows: list[dict], changed_names: list[str], removed_names: list[str]):
    """기존 parquet/FAISS에서 변경·삭제 파일 청크를 제거하고 새 청크를 병합 후 재인덱싱."""
    dirty_names = set(changed_names) | set(removed_names)

    # 기존 parquet에서 dirty 파일 청크 제거
    if PARQUET_PATH.exists():
        existing_df = load_chunks_parquet()
        existing_df = existing_df[~existing_df["filename"].isin(dirty_names)]
    else:
        existing_df = pd.DataFrame()

    new_df = pd.DataFrame(new_rows) if new_rows else pd.DataFrame()
    merged_df = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
    save_chunks_parquet(merged_df.to_dict("records"))
    print(f"[ingest] parquet 저장 완료: 총 {len(merged_df)}개 청크")

    # 기존 벡터에서 dirty 파일 제거 후 신규 벡터 병합
    try:
        from app.embed import embed_texts

        if FAISS_PATH.exists() and not existing_df.empty:
            old_vecs = np.load(str(FAISS_PATH))
            old_vecs = old_vecs[:len(existing_df)]  # dirty 제거 후 남은 행 수만큼 유지
        else:
            old_vecs = None

        if new_rows:
            new_vecs = np.array(embed_texts([r["text"] for r in new_rows])).astype("float32")
        else:
            new_vecs = None

        if old_vecs is not None and new_vecs is not None:
            all_vecs = np.concatenate([old_vecs, new_vecs], axis=0)
        elif old_vecs is not None:
            all_vecs = old_vecs
        elif new_vecs is not None:
            all_vecs = new_vecs
        else:
            return

        build_faiss_index(all_vecs)
        print(f"[ingest] 벡터 인덱스 빌드 완료: 총 {len(all_vecs)}개 벡터")
    except Exception as e:
        print(f"[경고] 벡터 인덱스 빌드 실패 (키워드 검색으로 동작): {e}")


@flow(name="ingest-documents")
def ingest_flow():
    init_db()
    files = list_pdf_files()
    changed, removed, current_hashes = detect_changes(files)

    if not changed and not removed:
        print("[ingest] 변경된 파일 없음 — 인덱싱 생략")
        return

    new_rows = process_files(changed) if changed else []
    merge_and_index(
        new_rows,
        changed_names=[f.name for f in changed],
        removed_names=removed,
    )
    save_hashes(current_hashes)
    print("[ingest] 해시 저장 완료")


if __name__ == "__main__":
    ingest_flow()
