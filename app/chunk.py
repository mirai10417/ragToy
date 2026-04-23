def semantic_chunk(elements, max_length: int = 700, overlap: int = 100):
    chunks = []
    current_source = None
    current_page = None
    page_texts = []

    def flush_page_texts(source, page, texts):
        joined = " ".join(t.strip() for t in texts if t.strip())
        if not joined:
            return

        start = 0
        while start < len(joined):
            end = start + max_length
            chunk_text = joined[start:end].strip()
            if chunk_text:
                chunks.append({
                    "source": source,
                    "page": page,
                    "text": chunk_text
                })
            if end >= len(joined):
                break
            start += max_length - overlap

    for elem in elements:
        text = elem.get("text", "").strip()
        source = elem.get("source")
        page = elem.get("page")

        if not text:
            continue

        if current_page is None:
            current_source = source
            current_page = page

        if source != current_source or page != current_page:
            flush_page_texts(current_source, current_page, page_texts)
            page_texts = []
            current_source = source
            current_page = page

        page_texts.append(text)

    if page_texts:
        flush_page_texts(current_source, current_page, page_texts)

    return chunks