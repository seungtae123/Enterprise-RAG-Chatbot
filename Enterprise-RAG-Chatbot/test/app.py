import streamlit as st # 웹 사이트 만드는 도구
import csv
import requests
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util

# --- [설정] ---
CSV_FILE_NAME = "my_data.csv"
MODEL_NAME = "exaone3.5"
API_URL = "http://localhost:11434/api/chat"

# --- [1. 모델 및 데이터 로딩 (캐싱)] ---
@st.cache_resource
def load_model():
    return SentenceTransformer("jhgan/ko-sbert-nli")

@st.cache_resource
def load_data():
    documents = []
    doc_embeddings = []

    if not os.path.exists(CSV_FILE_NAME):
        return [], None

    try:
        with open(CSV_FILE_NAME, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean_q = row.get('question', '').strip()
                clean_a = row.get('answer', '').strip()
                if clean_q and clean_a:
                    documents.append({"text": clean_q, "intent": clean_a})

        # 임베딩 생성
        model = load_model()
        doc_texts = [d["text"] + " " + d["intent"] for d in documents]
        doc_embeddings = model.encode(doc_texts, convert_to_tensor=True)

        return documents, doc_embeddings
    except Exception as e:
        st.error(f"데이터 로딩 중 오류 발생: {e}")
        return [], None

# 모델과 데이터 불러오기
embedder = load_model()
documents, doc_embeddings = load_data()

# --- [2. 검색 함수] ---
def retrieve_top_docs(query):
    if not documents: return []

    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]

    best_score = float(cos_scores.max())
    best_idx = int(cos_scores.argmax())

    if best_score > 0.4:  # 유사도 기준
        return [documents[best_idx]['intent']]
    else:
        return []

# --- [3. Ollama 호출 함수] ---
def ask_ollama(user_question, history):
    found_answers = retrieve_top_docs(user_question)

    if found_answers:
        context = found_answers[0]
        system_msg = (
            f"당신은 사내 안내 AI입니다. 아래 [정보]를 바탕으로 답변하세요.\n"
            f"[정보]: {context}\n"
            f"정보에 있는 내용은 정확히 전달하고, 없으면 모른다고 하세요."
        )
    else:
        system_msg = "관련 정보는 없지만, 친절한 AI 비서로서 자연스럽게 대화를 이어가세요."

    messages = [{"role": "system", "content": system_msg}]

    # 대화 기록 추가 (최근 10개만)
    for msg in history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_question})

    try:
        response = requests.post(API_URL, json={
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False
        })
        response.raise_for_status()
        return response.json()['message']['content']
    except Exception as e:
        return f"통신 오류: {e}"

# --- [4. 웹 화면 구성 (UI)] ---
st.title("🏢 우리 회사 AI 비서")
st.caption("궁금한 사내 규정이나 복지를 물어보세요!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# [수정] 대화 내용을 담을 '그릇(Container)'을 먼저 만듭니다.
# 이렇게 하면 매번 맨땅에 그리는 게 아니라, 이 상자 안 내용물만 교체하는 느낌을 줍니다.
chat_container = st.container()

# 그릇 안에다가 대화 내용을 차곡차곡 쌓습니다.
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("질문을 입력하세요..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("생각하는 중입니다..."):
            response = ask_ollama(prompt, st.session_state.messages[:-1])
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})