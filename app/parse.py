from pathlib import Path
from pypdf import PdfReader
import re


def looks_like_table(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 5:
        return False

    # 숫자 / 전화번호 / 반복 패턴이 많으면 표로 판단
    score = 0
    for line in lines[:20]:
        if re.search(r"\d{2,}", line):
            score += 1
        if "010-" in line:
            score += 2
        if len(line.split()) >= 3:
            score += 1

    return score >= 8


def parse_pdf(file_path: str):
    reader = PdfReader(file_path)
    texts = []

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = page_text.strip()

        if not page_text:
            continue

        # 표형 문서면 줄 단위
        if looks_like_table(page_text):
            units = page_text.splitlines()
        else:
            # 일반 문서는 문단 단위
            units = page_text.split("\n\n")

        for unit in units:
            unit = unit.strip()
            if unit:
                texts.append({
                    "source": Path(file_path).name,
                    "page": page_num,
                    "text": unit
                })

    return texts