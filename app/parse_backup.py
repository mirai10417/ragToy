# from unstructured.partition.pdf import partition_pdf

# def parse_pdf(file_path: str) -> list[str]:
#     elements = partition_pdf(
#         filename=file_path,
#         strategy="fast",   # 처음엔 단순하게
#     )
#     texts = []
#     for el in elements:
#         text = str(el).strip()
#         if text:
#             texts.append(text)
#     return texts

from pathlib import Path
from pypdf import PdfReader


def parse_pdf(file_path: str):
    reader = PdfReader(file_path)
    texts = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            texts.append({
                "source": Path(file_path).name,
                "page": page_num,
                "text": text
            })

    return texts