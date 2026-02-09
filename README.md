# 🏢 사내 규정 전문 RAG AI 챗봇 (Enterprise RAG Chatbot)

> **SBERT 임베딩과 LLM을 결합하여, 사내 규정(CSV)을 문맥 기반으로 검색하고 답변하는 AI 서비스입니다.**

---

## 📌 프로젝트 개요 (Overview)
- **개발 기간:** 202X.XX ~ 202X.XX (개인 프로젝트)
- **개발 의도:** 일반적인 키워드 검색(Keyword Search)의 한계를 극복하기 위해, **의미 기반 검색(Semantic Search)** 기술을 도입했습니다. 사용자가 "점심 언제 먹어?"라고 물어도 "12시 30분입니다"라는 규정을 찾아내도록 **RAG(Retrieval-Augmented Generation)** 파이프라인을 구축했습니다.

## 🛠 사용 기술 (Tech Stack)
- **Core:** Python 3.9
- **AI & Embedding:** Sentence-Transformers (`jhgan/ko-sbert-nli`), LLM (`Exaone 3.5` / `Gemma`)
- **Web Framework:** Streamlit (UI 구현 및 세션 관리)
- **Data:** CSV (Vector Database 대용으로 활용)
- **API:** REST API (Local LLM Server Communication)

## 💡 핵심 기능 및 구현 원리 (Key Features)

### 1. RAG (검색 증강 생성) 파이프라인 구축
- **문제 해결:** LLM이 없는 사실을 지어내는 '환각(Hallucination)' 현상을 방지.
- **구현:** 1. 사내 규정 데이터(`my_data.csv`)를 `SBERT` 모델을 통해 고차원 벡터로 임베딩.
  2. 사용자 질문과의 **코사인 유사도(Cosine Similarity)**를 계산하여 상위 3개의 관련 규정을 추출.
  3. 추출된 정보를 시스템 프롬프트(System Prompt)에 주입(Injection)하여 근거 있는 답변 생성.

### 2. 성능 최적화 (Optimization)
- **캐싱(Caching):** `@st.cache_resource`를 활용하여 무거운 임베딩 모델을 최초 1회만 로딩하도록 설계, 응답 속도 개선.
- **메모리 관리:** 대화가 길어질수록 프롬프트가 길어지는 문제를 방지하기 위해, 최근 대화 내역(Context Window)을 슬라이싱하여 관리.

### 3. 사용자 경험 (UX)
- **멀티턴 대화:** Streamlit의 `Session State`를 활용하여 이전 대화의 문맥을 유지하며 자연스러운 대화 가능.
- **예외 처리:** 검색된 정보가 없을 경우, 무리하게 답변하지 않고 "관련 정보를 찾을 수 없습니다"라고 응답하도록 프롬프트 제어.

## 📂 프로젝트 구조 (Structure)
```bash
📂 Enterprise-RAG-Chatbot
├── app.py           # Streamlit 웹 애플리케이션 메인 (UI & 로직)
├── chatbot.py       # RAG 검색 및 답변 생성 핵심 모듈 (Backend)
├── my_data.csv      # 사내 규정 데이터셋 (Knowledge Base)
└── config.toml      # 서버 환경 설정
