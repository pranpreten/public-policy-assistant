# 공공정책 문서 기반 RAG 질의응답

한국어 공공정책 문서에서 관련 근거를 검색하고, 이를 바탕으로 질문에 답하는 RAG 질의응답 프로젝트다. 미세먼지·폭염 등 정책 문서를 대상으로 문서 처리, 임베딩 검색, 언어 모델 응답, 웹 화면을 연결한다.

## 처리 흐름

`정책 문서 → 청크 분리 → 한국어 임베딩 → ChromaDB 저장 → 질문 분류 → 문서 검색 → 근거 기반 응답`

정책 질문은 검색 문맥과 출처를 포함해 답변하고 일반 질문은 언어 모델이 직접 응답한다. 질문 분류와 검색 범위는 코드에 정의한 규칙 및 문서 집합에 영향을 받는다.

## 기술 구성

- 언어 모델: Qwen2.5-14B-Instruct, vLLM
- 임베딩: `jhgan/ko-sroberta-multitask`
- 벡터 저장소: ChromaDB
- 웹: Flask

## 주요 파일

| 경로 | 역할 |
|---|---|
| `run.py` | 초기 구축·CLI·웹 실행 모드 |
| `app.py` | 질문 입력 및 응답 API |
| `src/document_processor.py` | 문서 처리 및 청크 생성 |
| `src/vector_store.py` | 임베딩과 벡터 검색 |
| `src/llm_model.py` | vLLM 모델 로딩 및 생성 |
| `src/rag_pipeline.py` | 검색과 답변 생성 연결 |
| `scripts/collect_data.py` | 코드에 정의된 기본 정책 자료 저장 |
| `data/raw/`, `data/processed/`, `vectordb/` | 원문·청크·검색 인덱스 |

## 실행 준비

vLLM을 구동할 수 있는 Linux/WSL2·CUDA 환경과 모델을 적재할 충분한 GPU 메모리가 필요하다. 일반적인 Windows CPU 가상환경만으로 실행하는 GUI 프로젝트와 실행 조건이 다르다.

`requirements.txt`에는 vLLM이 포함되어 있지 않으므로 PyTorch·CUDA와 호환되는 vLLM을 별도로 설치해야 한다. 모델 가중치는 저장소에 포함되지 않으며 `models/Qwen2.5-14B-Instruct`에 준비하거나 코드의 Hugging Face 모델 로딩 경로를 이용한다. 최초 모델·임베딩 다운로드에는 인터넷 연결이 필요하다.

## 실행 방법

준비된 환경의 저장소 루트에서 실행한다.

```bash
python -m pip install -r requirements.txt
python -m pip install vllm
python run.py --mode setup
python run.py --mode web
```

`setup`은 기본 자료와 청크, 검색 저장소를 구축한다. 기존 자료를 유지하려면 재구축 전에 해당 파일을 확인한다. 웹은 `http://localhost:7865`에서 사용한다. CLI 실행은 다음과 같다.

```bash
python run.py --mode cli
```

## 데이터와 결과 해석

저장소에는 기본 정책 JSON·텍스트와 청크·검색 인덱스가 있다. 수집 스크립트의 추가 API 수집 부분은 실제 자동 갱신 기능으로 완성되어 있지 않으므로 최신 법령·정책이 자동 반영되는 시스템으로 설명하지 않는다.

확인할 결과는 답변 내용, 제시된 출처와 검색 문서의 일치 여부다. 저장 문서 밖의 정보나 최신 개정까지 보장하지 않으며, 답변 정확도의 정식 평가 수치를 이 README에서 별도로 주장하지 않는다.

---

[GitHub 저장소](https://github.com/pranpreten/public-policy-assistant) · [문서 작성 기준 코드](https://github.com/pranpreten/public-policy-assistant/tree/bd1e9972ea1ac70938ffed88ade7b6ea9f4f6f36)
