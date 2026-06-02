# 🏢 사내 규정 기반 하이브리드 AI 챗봇 (Enterprise RAG & Assistant)

SBERT 딥러닝 임베딩과 로컬 LLM을 결합하여, 사내 규정(CSV)을 기반으로 정확한 정보를 검색하고 상황에 따라 유연하게 대응하는 스마트 AI 서비스입니다. 웹(GUI)과 터미널(CLI) 환경을 모두 지원합니다.

## 📌 프로젝트 개요 (Overview)
* **개발 기간:** 2025.12 ~ 2025.12 (개인 프로젝트)
* **개발 의도:** 일반적인 키워드 검색(Keyword Search)의 한계를 극복하기 위해 문맥 중심의 의미 기반 검색(Semantic Search) 기술을 도입했습니다. 사용자가 규정 단어와 완전히 일치하지 않는 질문을 하더라도, 질문의 의도를 파악하여 정확한 사내 규정을 찾아 답변하는 RAG(Retrieval-Augmented Generation) 파이프라인을 구축했습니다.

## 🛠 사용 기술 (Tech Stack)
* **Core:** Python 3.9
* **AI & Embedding:** Sentence-Transformers (`jhgan/ko-sbert-nli`)
* **LLM:** EXAONE 3.5 (Ollama를 통한 로컬 환경 구동)
* **Interface:** Streamlit (웹 UI 및 세션 관리) / Python CLI (터미널 대화 환경)
* **Data:** CSV (Vector Database 대용으로 활용)
* **API:** REST API (Ollama Local Server Communication)

## 💡 핵심 기능 및 구현 원리 (Key Features)

### 1. 고효율 RAG 파이프라인 (가장 정확한 핵심 정보 추출)
* **의미 기반 매칭:** 사내 규정 데이터(`my_data.csv`)를 SBERT 모델을 통해 고차원 벡터로 변환하여 저장합니다.
* **타겟팅 검색:** 사용자 질문이 들어오면 **코사인 유사도(Cosine Similarity)**를 계산하여, 전체 지식 중 **가장 연관성이 높은 1개의 핵심 규정**을 정확하게 짚어내어 시스템 프롬프트에 주입(Injection)합니다.

### 2. 상황별 프롬프트 제어 및 유연한 Fallback (UX 최적화)
* **유연한 대화 전환:** 유사도 기준(0.4)을 넘는 사내 규정이 존재할 때는 제공된 근거 데이터를 바탕으로 정확한 답변을 생성합니다.
* **하이브리드 AI 비서:** 질문에 해당하는 규정 정보가 없을 경우, 답변을 단호하게 거부하는 대신 **친절한 일상 대화 모드로 자연스럽게 전환**되어 사용자의 대화 흐름이 끊기지 않도록 유연한 사용자 경험(UX)을 제공합니다.

### 3. 멀티 환경 및 대화 문맥 유지 (Multi-turn)
* **듀얼 인터페이스 지원:** 웹 브라우저 기반의 편리한 `app.py` 환경과 가볍고 빠르게 구동 가능한 터미널 기반의 `chatbot.py` 환경을 모두 제공합니다.
* **컨텍스트 윈도우 슬라이싱:** 대화가 길어져 대화 기록이 쌓이더라도, 시스템 메모리와 프롬프트 크기를 최적화하기 위해 인터페이스 환경에 따라 최근 10~20개의 대화 내역만 유동적으로 슬라이싱하여 안정적인 연속 대화(Multi-turn)를 유지합니다.

### 4. 캐싱 기반 성능 최적화 (Optimization)
* **리소르 로딩 최적화:** 무거운 SBERT 임베딩 모델 로딩과 대용량 데이터의 벡터화 작업을 최초 1회만 수행하도록 Streamlit의 `@st.cache_resource`를 적용하여 사용자의 대기 시간을 획기적으로 단축시켰습니다.

## 📂 프로젝트 구조 (Structure)
```bash
📂 Enterprise-RAG-Chatbot
├── app.py           # Streamlit 웹 애플리케이션 메인 (UI & 로직)
├── chatbot.py       # RAG 검색 및 답변 생성 핵심 모듈 (Backend)
├── my_data.csv      # 사내 규정 데이터셋 (Knowledge Base)
└── config.toml      # 서버 환경 설정
