from pathlib import Path
from pypdf import PdfReader
from pdf2image import convert_from_path
from PIL import ImageOps, ImageEnhance, ImageFilter
from paddleocr import PaddleOCR
import re

POPPLER_PATH = r"C:\Users\yhs\Downloads\Release-24.07.0-0\poppler-24.07.0\Library\bin"

ocr = PaddleOCR(
    lang="korean",
    use_angle_cls=False
)


def normalize_ocr_text(text: str) -> str:
    text = str(text)
    text = text.replace("|", " ")
    text = text.replace("—", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def preprocess_image_for_receipt(img):
    # 좌우 Original / Highlighted 이미지가 같이 있으면 왼쪽 원본만 사용
    if img.width > img.height * 1.3:
        img = img.crop((0, 0, img.width // 2, img.height))

    margin_x = int(img.width * 0.03)
    margin_y = int(img.height * 0.03)

    img = img.crop((
        margin_x,
        margin_y,
        img.width - margin_x,
        img.height - margin_y
    ))

    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = img.resize((img.width * 2, img.height * 2))
    img = img.filter(ImageFilter.SHARPEN)

    return img


def run_paddle_ocr(image_path: str):
    result = ocr.ocr(image_path)

    print("========== RAW PADDLE OCR RESULT ==========")
    print(result)
    print("===========================================")

    lines = []

    if not result:
        return ""

    for page_result in result:
        if not page_result:
            continue

        for item in page_result:
            try:
                text = item[1][0]
                score = item[1][1]
            except Exception:
                continue

            if score < 0.3:
                continue

            text = normalize_ocr_text(text)

            if len(text) >= 2:
                lines.append(text)

    return "\n".join(lines)


def parse_pdf(file_path: str):
    reader = PdfReader(file_path)
    texts = []

    for page_num, _ in enumerate(reader.pages, start=1):
        images = convert_from_path(
            file_path,
            first_page=page_num,
            last_page=page_num,
            poppler_path=POPPLER_PATH,
            dpi=400
        )

        if not images:
            continue

        processed = preprocess_image_for_receipt(images[0])

        debug_path = Path(f"data/processed/debug_{Path(file_path).stem}_page_{page_num}.png")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        processed.save(debug_path)

        ocr_text = run_paddle_ocr(str(debug_path))

        print("========== OCR RESULT ==========")
        print(ocr_text)
        print("================================")

        if not ocr_text:
            continue

        for line in ocr_text.splitlines():
            line = normalize_ocr_text(line)

            if len(line) < 2:
                continue

            texts.append({
                "source": Path(file_path).name,
                "page": page_num,
                "text": line
            })

    return texts