def semantic_chunk(elements, max_length: int = 500):
    chunks = []
    buffer = ""
    current_source = None
    current_page = None

    for elem in elements:
        if isinstance(elem, dict):
            text = elem.get("text", "")
            source = elem.get("source")
            page = elem.get("page")
        else:
            text = str(elem)
            source = None
            page = None

        text = text.strip()
        if not text:
            continue

        # 새 페이지/새 source로 넘어가면 기존 buffer 저장
        if buffer and (source != current_source or page != current_page):
            chunks.append({
                "source": current_source,
                "page": current_page,
                "text": buffer
            })
            buffer = ""

        current_source = source
        current_page = page

        if len(buffer) + len(text) + 1 <= max_length:
            buffer = f"{buffer} {text}".strip()
        else:
            if buffer:
                chunks.append({
                    "source": current_source,
                    "page": current_page,
                    "text": buffer
                })
            buffer = text

    if buffer:
        chunks.append({
            "source": current_source,
            "page": current_page,
            "text": buffer
        })

    return chunks