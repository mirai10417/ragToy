from typing import List
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

def semantic_chunk(elements: List[str]) -> List[str]:
    chunks = []
    current = ""

    for elem in elements:
        elem = elem.strip()
        if not elem:
            continue

        # 제목/문단 단위 느낌을 살리되 길이 제한
        if len(current) + len(elem) + 1 <= CHUNK_SIZE:
            current = f"{current}\n{elem}".strip()
        else:
            if current:
                chunks.append(current)

            # overlap 흉내: 직전 chunk의 tail 일부 유지
            overlap_text = current[-CHUNK_OVERLAP:] if current else ""
            current = f"{overlap_text}\n{elem}".strip()

    if current:
        chunks.append(current)

    return chunks