from pathlib import Path
import uuid
from prefect import flow, task
from app.config import RAW_DIR
from app.db import init_db, get_conn
from app.parse import parse_pdf
from app.chunk import semantic_chunk
from app.embed import embed_texts
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
        chunks = semantic_chunk(elements)

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
    texts = [r["text"] for r in rows]
    embeddings = embed_texts(texts)
    build_faiss_index(embeddings)
    save_chunks_parquet(rows)


@flow(name="ingest-documents")
def ingest_flow():
    init_db()
    files = list_pdf_files()
    rows = process_files(files)
    embed_and_index(rows)


if __name__ == "__main__":
    ingest_flow()