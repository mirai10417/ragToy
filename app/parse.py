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


def is_toc_text(text: str) -> bool:
    """목차(Table of Contents) 페이지 여부를 판단한다."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False

    # 명시적 목차 헤더가 있으면 즉시 True
    toc_headers = re.compile(r"^(목\s*차|차\s*례|contents|table\s+of\s+contents)$", re.IGNORECASE)
    if any(toc_headers.match(l) for l in lines):
        return True

    if len(lines) < 4:
        return False

    # 점선 뒤 숫자 패턴: "제목 ···· 15" 처럼 점선과 숫자가 같은 줄에 있어야 목차 줄로 판단
    # 단순히 줄 끝 숫자(등수, 수량 등)는 제외
    toc_line_count = sum(
        1 for l in lines
        if re.search(r"[·.]{3,}.*\d+\s*$|[-─]{4,}.*\d+\s*$", l)
    )

    toc_ratio = toc_line_count / len(lines)

    # 목차형 줄이 25% 이상이면 목차 페이지로 판단
    return toc_ratio >= 0.25


def normalize_ocr_text(text: str) -> str:
    text = str(text)
    text = text.replace("|", " ")
    text = text.replace("—", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def is_text_pdf(reader: PdfReader, min_chars: int = 50) -> bool:
    """pypdf로 추출 가능한 텍스트가 있는지 확인"""
    for page in reader.pages:
        text = page.extract_text() or ""
        if len(text.strip()) >= min_chars:
            return True
    return False


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


def preprocess_image_general(img):
    """일반 스캔 문서용 전처리"""
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    return img


def run_paddle_ocr(image_path: str):
    result = ocr.ocr(image_path)

    print("========== RAW PADDLE OCR RESULT ==========")
    print(result)
    print("===========================================")

    if not result:
        return ""

    # 좌표 기반 레이아웃 복원: Y좌표로 행 묶기, X좌표로 열 정렬
    tokens = []  # (y_center, x_center, text)
    for page_result in result:
        if not page_result:
            continue
        for item in page_result:
            try:
                box = item[0]   # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                text = item[1][0]
                score = item[1][1]
            except Exception:
                continue
            if score < 0.3:
                continue
            text = normalize_ocr_text(text)
            if len(text) < 1:
                continue
            ys = [pt[1] for pt in box]
            xs = [pt[0] for pt in box]
            y_center = sum(ys) / len(ys)
            x_center = sum(xs) / len(xs)
            tokens.append((y_center, x_center, text))

    if not tokens:
        return ""

    # Y좌표 기준 정렬 후 같은 행(y 차이 20px 이내)끼리 묶기
    tokens.sort(key=lambda t: t[0])
    row_threshold = 20
    rows: list[list[tuple]] = []
    current_row: list[tuple] = [tokens[0]]

    for tok in tokens[1:]:
        if abs(tok[0] - current_row[-1][0]) <= row_threshold:
            current_row.append(tok)
        else:
            rows.append(current_row)
            current_row = [tok]
    rows.append(current_row)

    # 각 행 내부를 X좌표 순으로 정렬 후 공백으로 연결
    lines = []
    for row in rows:
        row.sort(key=lambda t: t[1])
        lines.append("  ".join(t[2] for t in row))

    raw = "\n".join(lines)
    # "20 000" → "20000" 형태로 분리된 숫자 합치기
    raw = re.sub(r"(\d+)\s{1,2}(\d{3})(?=\D|$)", r"\1\2", raw)
    return raw


def parse_pdf(file_path: str):
    reader = PdfReader(file_path)
    filename = Path(file_path).name
    texts = []

    # 텍스트 기반 PDF: pypdf로 직접 추출 (OCR 불필요)
    if is_text_pdf(reader):
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = normalize_ocr_text(text).strip()
            if len(text) < 2:
                continue
            if is_toc_text(text):
                print(f"[parse] 목차 페이지 건너뜀: {filename} p.{page_num}")
                continue
            texts.append({
                "source": filename,
                "page": page_num,
                "text": text
            })
        return texts

    # 이미지 기반 PDF: OCR 필요
    is_receipt = "receipt" in filename.lower()

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

        if is_receipt:
            processed = preprocess_image_for_receipt(images[0])
        else:
            processed = preprocess_image_general(images[0])

        debug_path = Path(f"data/processed/debug_{Path(file_path).stem}_page_{page_num}.png")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        processed.save(debug_path)

        ocr_text = run_paddle_ocr(str(debug_path))

        print("========== OCR RESULT ==========")
        print(ocr_text)
        print("================================")

        if not ocr_text:
            continue

        if is_receipt:
            # 영수증: 줄 단위로 분리 (행별 매칭 필요)
            for line in ocr_text.splitlines():
                line = normalize_ocr_text(line)
                if len(line) < 2:
                    continue
                texts.append({
                    "source": filename,
                    "page": page_num,
                    "text": line
                })
        else:
            # 일반 스캔 문서: 페이지 전체를 하나의 청크로 (청커가 분할)
            if is_toc_text(ocr_text):
                print(f"[parse] 목차 페이지 건너뜀: {filename} p.{page_num}")
                continue
            texts.append({
                "source": filename,
                "page": page_num,
                "text": ocr_text
            })

    return texts