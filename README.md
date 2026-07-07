# 실행 방법

## 1. 가상환경

python -m venv venv
venv\Scripts\activate

## 2. 설치

pip install -r requirements.txt
pip install pypdf

## 3. 문서 적재

python -m app.ingest_flow

## 4. 서버 실행

uvicorn app.api:app --reload

## 5. python version

Python 3.11.9

## 6. 도커 

docker exec -it ollama ollama pull llama3.2:1b
docker run -d --name ollama -p 11434:11434 -v ollama:/root/.ollama ollama/ollama

## 7. streamlit UI 실행
streamlit run ui.py

## 문서 자동 인덱싱 서버
python -m app.watcher

## 문서별 청크 수 확인하기
python -c "
import pandas as pd
df = pd.read_parquet('E:/ragToy/data/processed/chunks.parquet')
print(df['filename'].value_counts())
"

## 문서별 청크 수 + 내용 확인하기
python -c "
import pandas as pd
pd.set_option('display.max_colwidth', 100)
df = pd.read_parquet('E:/ragToy/data/processed/chunks.parquet')
print(df[df['filename'] == 'sample1.pdf'][['page', 'chunk_index', 'text']].head(20).to_string())
"

## 문서별 청크 수 + 텍스트 길이
python -c "
import pandas as pd
df = pd.read_parquet('E:/ragToy/data/processed/chunks.parquet')
s3 = df[df['filename'] == '퓨전소프트 규정집(종합본)_2024.06.01.pdf']
print('청크 수:', len(s3))
print('텍스트 길이:', s3['text'].str.len().values)
"

python -c "
import pandas as pd
df = pd.read_parquet('data/processed/chunks.parquet')
print(df.head())
"

────────────────────────────────────────────────────────────────────────────────────────────────
문서 기반 질문

{
  "question": "대출만기 및 거치기간",
  "top_k": 2
}

{
  "question": "상환방식은?",
  "top_k": 2
}

{
  "question": "6842는 누구야?",
  "top_k": 10
}

{
  "question": "배당금총액(원)은 얼마야?",
  "top_k": 2
}

{
  "question": "배당금지급 예정일자는?",
  "top_k": 2
}

{
  "question": "receipt.pdf에서 주문번호 뭐야?",
  "top_k": 10
}

{
  "question": "receipt.pdf에서 20210220 보이니?",
  "top_k": 10
}

────────────────────────────────────────────────────────────────────────────────────────────────
LLM 기반 질문

{
  "question": "검색 증강 생성이 뭐야?",
  "top_k": 3
}

{
  "question": "OCR이 왜 필요한지 쉽게 설명해줘",
  "top_k": 3
}

────────────────────────────────────────────────────────────────────────────────────────────────

python 설치
pip install pdf2image pytesseract pillow

pytesseract → OCR 엔진
pdf2image → PDF를 이미지로 바꾸는 도구
pdf2image는 Windows에서 Poppler가 필요함


https://github.com/UB-Mannheim/tesseract/wiki
	tesseract-ocr-w64-setup-5.5.0.20241111.exe (64 bit)

poppler
	https://github.com/oschwartz10612/poppler-windows/releases/tag/v24.07.0-0
		시스템 환경 변수 추가
		C:\Users\yhs\Downloads\Release-24.07.0-0\poppler-24.07.0\Library\bin

※ 환경변수에 안잡고 py파일에 하드코딩으로 경로 잡아도 된다.