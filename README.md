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
  "question": "6842는 누구야?",
  "top_k": 2
}

{
  "question": "sample3.pdf에서 배당금총액(원)은 얼마야?",
  "top_k": 2
}

{
  "question": "sample3.pdf에서 배당금지급 예정일자는?",
  "top_k": 2
}