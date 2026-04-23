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



현재 scikit-learn 사용중인데 torch가 좋음



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
  "question": "현금 ·현물배당 결정 공고 회사는 어디야?",
  "top_k": 10
}