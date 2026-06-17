import re


def _is_toc_chunk(text: str) -> bool:
    """청크 단위 목차 잔재 필터 (페이지 필터를 통과한 부분 목차 제거)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    toc_lines = sum(1 for l in lines if re.search(r"[·.]{3,}.*\d+\s*$|[-─]{4,}.*\d+\s*$", l))
    return (toc_lines / len(lines)) >= 0.25


def semantic_chunk(elements, max_length: int = 700, overlap: int = 100):
    chunks = []

    for elem in elements:
        text = elem.get("text", "").strip()
        source = elem.get("source")
        page = elem.get("page")

        if not text:
            continue

        if _is_toc_chunk(text):
            continue

        # 짧은 OCR/영수증/표 문서는 줄 단위 유지
        if len(text) <= 100:
            chunks.append({
                "source": source,
                "page": page,
                "text": text
            })
            continue

        # 일반 문서는 길이 기준으로 분할
        start = 0
        while start < len(text):
            end = start + max_length
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "source": source,
                    "page": page,
                    "text": chunk_text
                })

            if end >= len(text):
                break

            start += max_length - overlap

    return chunks