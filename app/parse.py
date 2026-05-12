from pathlib import Path
from pypdf import PdfReader
import re
from pdf2image import convert_from_path
import pytesseract
from PIL import ImageOps, ImageFilter

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Users\yhs\Downloads\Release-24.07.0-0\poppler-24.07.0\Library\bin"


def looks_like_table(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 5:
        return False

    score = 0
    for line in lines[:20]:
        if re.search(r"\d{2,}", line):
            score += 1
        if "010-" in line:
            score += 2
        if len(line.split()) >= 3:
            score += 1

    return score >= 8


def preprocess_image_for_ocr(img):
    # 그레이스케일
    img = ImageOps.grayscale(img)

    # 크기 2배 확대
    img = img.resize((img.width * 2, img.height * 2))

    # 샤프닝
    img = img.filter(ImageFilter.SHARPEN)

    # 이진화
    img = img.point(lambda x: 0 if x < 180 else 255, mode="1")

    return img


def parse_pdf(file_path: str):
    reader = PdfReader(file_path)
    texts = []

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = page_text.strip()

        if page_text:
            if looks_like_table(page_text):
                units = page_text.splitlines()
            else:
                units = page_text.split("\n\n")

            for unit in units:
                unit = unit.strip()
                if unit:
                    texts.append({
                        "source": Path(file_path).name,
                        "page": page_num,
                        "text": unit
                    })
            continue

        images = convert_from_path(
            file_path,
            first_page=page_num,
            last_page=page_num,
            poppler_path=POPPLER_PATH,
            dpi=300
        )

        if not images:
            continue

        processed = preprocess_image_for_ocr(images[0])

        ocr_text = pytesseract.image_to_string(
            processed,
            lang="kor+eng",
            config="--psm 6"
        ).strip()

        if not ocr_text:
            continue

        units = ocr_text.splitlines()

        for unit in units:
            unit = unit.strip()
            if unit:
                texts.append({
                    "source": Path(file_path).name,
                    "page": page_num,
                    "text": unit
                })

    return texts