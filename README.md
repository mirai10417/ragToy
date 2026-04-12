# 실행 방법

## 1. 가상환경

python -m venv venv
venv\Scripts\activate

## 2. 설치

pip install -r requiredments.txt

## 3. 문서 적재

python -m app.ingest_flow

## 4. 서버 실행

uvicorn app.api:app --reload

## 5. python version

Python 3.11.9

현재 scikit-learn 사용중인데 torch가 좋음
