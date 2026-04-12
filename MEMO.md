python 3.11.9 64bit 설치
Add python.exe to PATH 체크 후 Install Now
※ 토이 프로젝트라 Customize Installation 아직은 필요없음(기본옵션이면 충분)
Disabled path length limit 클릭
────────────────────────────────
기본설치에 포함 되는 것
· pip
· IDLE
· 문서
· 파일 연결
· python launcher
────────────────────────────────
설치 후 CMD를 켜서
python --version
pip --version

Python 3.11.9
pip 24.C:\경로~~~

mkdir rag-toy
cd rag-toy
python -m venv venv
venv\Scripts\activate
Java의 Spring initializr와 같이 프로젝트 시작 준비 완료

어떤 것과 같냐?
Java -> Maven 프로젝트 생성 -> JDK 연결 끝난 상태
────────────────────────────────
1단계: 필수 라이브러리 설치
이제 MVP를 위해 라이브러리 설치 + 프로젝트 뼈대 생성 필요

venv\Scripts\activate
위의 명령어를 치면 (venv) C:\Users\user\rag-toy>모드로 이동하는데 여기서 필수 라이브러리를 설치해야함

pip install fastapi uvicorn prefect sentence-transformers faiss-cpu unstructured python-multipart pydantic

· RAG MVP 기준으로 딱 필요한 최소 세트입니다.
· FastAPI → API 서버
· Uvicorn → 실행 서버
· Prefect → 문서 파이프라인 orchestration
· Sentence Transformers → 임베딩
· FAISS → 벡터 검색
· Unstructured → 문서 파싱
· Pydantic → API 모델
────────────────────────────────
2단계: 프로젝트 폴더 만들기
mkdir app
mkdir data
mkdir index
────────────────────────────────
3단계: 핵심 파일 만들기
이제 Java 기준으로

· Controller
· Service
· Config

만드는 느낌입니다.
type nul > app\main.py
type nul > app\ingest.py
type nul > app\retriever.py
type nul > app\pipeline.py
────────────────────────────────
4단계: 추천 IDE > VSCODE 추천
code .
────────────────────────────────
5단계: VSCODE에서 해당 폴더를 열고 터미널에서 아래 명령어를 친다.
source venv/Scripts/activate
(ven) user~~~~~/rag-toy
python -m pip show prefect
설치정보 확인
python -m app.ingest_flow

※ python -m app.ingest_flow 오류가 난다면
python -m pip install "unstructured[pdf]" pdfminer.six pillow pi-heif
python -m pip install unstructured-inference

＃parse.py 내용 변경
python -m pip install pypdf

＃torch 문제발생
$ python -c "import torch; print(torch.**version**)"
Traceback (most recent call last):
File "<string>", line 1, in <module>
File "C:\Users\user\rag-toy\venv\Lib\site-packages\torch\_\_init**.py", line 285, in <module>
\_load_dll_libraries()
File "C:\Users\user\rag-toy\venv\Lib\site-packages\torch\_\_init**.py", line 268, in \_load_dll_libraries
raise err
OSError: [WinError 1114] DLL 초기화 루틴을 실행할 수 없습니다. Error loading "C:\Users\user\rag-toy\venv\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
(venv)

python -m pip uninstall -y torch torchvision torchaudio
python -m pip cache purge
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

＃설치
python -m pip install pyarrow

해결 실패 > torch 우회 > 실무에서는 필요 > 대안: scikit-learn <-- 이게 무엇인지 조사해보기

────────────────────────────────
torch란?
PyTorch (torch)는 AI 모델이 실제로 숫자 계산을 하는 엔진입니다.
텍스트 → 숫자 벡터 변환

Java로 비유
sentence-transformers = 서비스 코드
torch = 실제 연산하는 핵심 엔진 라이브러리

즉 모델을 “실행”하는 역할입니다.

from sentence_transformers import SentenceTransformer
코드에서 torch를 사용

임베딩 모델
all-MiniLM-L6-v2
bge-m3

이런 모델은 내부적으로 신경망 계산을 해야 하거든요.

흐름
문서 텍스트
→ sentence-transformers
→ torch
→ 벡터(숫자 배열)
→ FAISS 저장

torch는 embed.py 전용 엔진

그렇다면 해당 부분이 왜 중요한가?

RAG에서 가장 중요한 단계가: 질문 문장과 문서 문장을 같은 벡터 공간에 놓기 임

예를 들어
문서: 선도교사 자격조건은 경력 3년 이상
질문: 선도교사 경력 몇 년?

이 두 문장이 비슷하다는 걸 숫자로 만드는 게 임베딩이고,
그 계산 엔진이 torch입니다.
────────────────────────────────
＃실행 명령어 설명
python -m app.ingest_flow
지금 python -m app.ingest_flow 는 문서 적재 작업만 실행한 겁니다.
로그를 보면 flow가 Completed() 로 끝났고, 그 뒤에 Prefect 임시 서버도 종료됐습니다.
즉 작업은 끝났고 서버는 내려간 상태예요.
그래서 127.0.0.1:8901 로 접속하면 연결 거부가 뜨는 게 맞습니다.

python -m app.ingest_flow

PDF 읽기
chunk 생성
임베딩
FAISS 저장
parquet 저장
끝나면 종료

python -m uvicorn app.api:app --reload
웹 API 서버 실행
브라우저에서 접속 가능

정상
Uvicorn running on http://127.0.0.1:8000 이런 느낌으로 표시

브라우저 접근 주소
http://127.0.0.1:8000/docs
여기서 /health, /ask 테스트하면 됩니다.

추가로 확인하면 좋은 것:
data/processed/faiss.index
data/processed/chunks.parquet
────────────────────────────────
질문 할 프롬프트
담보주택 당 한도는 얼마야?
3128번호를 가진 사람의 이름이 뭐야?

{
"question": "sample1.pdf에서 담보주택 당 한도는 얼마야?",
"top_k": 4
}

{
"question": "sample2.pdf에서 3128번호를 가진 사람의 이름이 뭐야?",
"top_k": 4
}

sources = 검색된 chunk들
임베딩 된 문서 chunk 중 질문과 가장 유사한 top-k 결과
PDF
→ parse
→ chunk 분리
→ embedding
→ FAISS 저장
→ 질문 embedding
→ 유사한 chunk top-k 검색
→ sources 반환

text = chunk 본문
chunk의 실제 텍스트 내용
ex) 대출만기는 10년, 15년, 20년, 30년 source Text안에 포함 되는 부분

score = HashingVectorizer + FAISS IndexFlatIP 구조에서 코사인 유사도처럼 쓰는 점수
범위: 0 ~ 1

분류 기준
매우 높음: 0.7 ~ 1.0 (질문과 거의 같은 문장 / 키워드 포함)

적당히 관련
0.3 ~ 0.7 (관련은 있지만 정확 문장은 아님)

낮음
0.1 ~ 0.3 (주제는 비슷한데 정확도 떨어짐)

거의 무관
0 ~ 0.1 (노이즈 가능성 큼)

질문: 대출만기 및 거치기간은 몇년이 있어?
source: 대출만기는 10년, 15년, 20년, 30년

사람이 보기엔 정답인데 score가 낮은 이유는
지금 임베딩이 진짜 semantic embedding이 아니라 HashingVectorizer 기반 임시 방식이라서 그렇습니다.
즉 지금은: 단어 겹침 기반 lexical search에 가깝습니다. > 해결을 위해서 entence-transformers 붙이면 달라짐 SentenceTransformer("BAAI/bge-m3") > score가 확 올라감 예상 수준) 0.75~0.9

그래서 스코어가 낮으면 아예 답변을 안주는 것도 좋음  
if sources[0]["score"] < 0.15:
return "관련 문서를 찾지 못했습니다."

────────────────────────────────

실무형 응답포인트
{
"question": "sample1.pdf에서 대출만기 및 거치기간은 몇년이 있어?",
"answer": "대출만기는 10년, 15년, 20년, 30년이며, 거치기간은 1년 또는 비거치입니다.",
"matched_count": 4,
"sources": [
{
"rank": 1,
"source": "sample1.pdf",
"page": 13,
"text": "...",
"score": 0.1482
}
]
}

────────────────────────────────

청크파일 소스가 수정되면 ingest_flow 자체 새로 시작해야하고
FastApi도 재실행 해야
