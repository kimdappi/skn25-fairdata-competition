# agent기반 rag 설계
- metrics.json(평가셋)은 임의로 생성한 데이터셋이므로, 별도 검증할 필요는 없다.
### 코드 설계 조건

- 주석 단순하게 포함, fallback 없는 베이스라인으로 설계
- Langgraph의 stategraph를 일차적으로 사용한다
- search_raw_data와 from_uy에 있는 내용을 참고하고, 분석한다.

### 목표
- langgraph  기반 query에 대한 올바른 chunk_id 5개 검색
- from_uy/eda.py에 작성되어있는 분류 키워드를 프롬프트 엔지니어링으로 주입
- 메인 키워드를 라우팅 함수로 구분하여 retrieve에 있어 성능 향상
- search_raw_data에 있는 데이터의 임베딩 모델과 경로, 성능평가등 필요한 경로를 config.py에 분리한다.
- 성능평가까지 파이프라인 구축

### 설계 구조

- 법률 관련 질문을 물으면, 질문의 의도가 큰 도메인으로 무엇인지, [eda.py](http://eda.py) 코드 분류를 참고하는 라우팅 함수를 거치도록 한다.
- pydantic, description, 프롬프트 작성을 실천한다.
- query에 대한 답변 chunk_id가 5개가 나오도록 한다.

- splade도 도입해서, rrf 가중치를 splade bm25 densesearch 중에 splade를 제일 높게 한다. 

### 사용 모델

- data에 있는 데이터의 임베딩 모델과 경로, 성능평가등 필요한 경로를 config.py에 분리한다.
- 한국어 형태소 분리기, 임베딩 모델은 법률도메인에 맞는 모델로 너가 선정한다.
- retrieve는 기본적으로 faiss 사용한다,

### 성능평가

- metrics.json을 활용하여 query에 대한 chunk_id가 무엇이 나오는지 코드를 실행하고, Recall@5와 MRR 평가를 진행하여 특정 파일로 저장해둔다.
- metrics.json은 임의로 생성한 데이터셋이므로, 별도 검증할 필요는 없다.


상황 보고상.. bge가 이미 dense와 sparse를 둘 다 가능해서

splade 돌려보고, 비슷한 비율로 bge 돌려서 누가 더 나은지에 대한 논리를 작성할것.