import csv
import requests
import numpy as np
import os
import sys
from sentence_transformers import SentenceTransformer, util # 딥러닝 검색 도구

# --- [설정] ---
CSV_FILE_NAME = "my_data.csv"
MODEL_NAME = "exaone3.5"       # LG EXAONE 모델
API_URL = "http://localhost:11434/api/chat"

# --- [1. 딥러닝 임베딩 모델 로딩] ---
print("▶ 딥러닝 검색 모델(SBERT)을 로딩 중입니다... (처음엔 시간 좀 걸려요)")
# 한국어 문장의 의미를 잘 파악하는 모델입니다.
embedder = SentenceTransformer("jhgan/ko-sbert-nli")

# --- [2. 기억 장치 초기화] ---
chat_history = []

# --- [3. CSV 파일 읽기 & 벡터화] ---
documents = []      # 원본 텍스트 저장
doc_embeddings = [] # 의미를 숫자로 바꾼 벡터 저장

if not os.path.exists(CSV_FILE_NAME):
    print(f"오류: '{CSV_FILE_NAME}' 파일이 없습니다.")
    exit()

try:
    with open(CSV_FILE_NAME, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_q = row.get('question', '').strip()
            clean_a = row.get('answer', '').strip()

            if clean_q and clean_a:
                documents.append({"text": clean_q, "intent": clean_a})

    print(f"▶ 데이터 로딩 완료: {len(documents)}개의 지식을 학습합니다.")

    # [핵심] 모든 질문+답변을 딥러닝 모델을 통해 '숫자 벡터'로 변환합니다.
    # TF-IDF와 달리 단어의 '의미'가 숫자로 저장됩니다.
    doc_texts = [d["text"] + " " + d["intent"] for d in documents]
    doc_embeddings = embedder.encode(doc_texts, convert_to_tensor=True)

    print("▶ 지식 학습(임베딩) 완료!")

except Exception as e:
    print(f"오류 발생: {e}")
    exit()

# --- [4. 의미 기반 검색 함수] ---
def retrieve_top_docs(query):
    # 1. 사용자 질문도 딥러닝 모델로 숫자로 변환
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # 2. 질문과 저장된 지식들 사이의 의미 유사도(Cos Sim) 계산
    cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]

    # 3. 가장 점수가 높은 것 찾기
    best_score = float(torch_max_val := cos_scores.max()) # 최고 점수
    best_idx = int(cos_scores.argmax())                   # 최고 점수의 위치

    # 점수가 0.4점 이상일 때만 답변 채택 (의미가 어느 정도 통해야 함)
    # TF-IDF보다 기준을 좀 더 높게 잡아도 됩니다.
    if best_score > 0.4:
        return [documents[best_idx]['intent']]
    else:
        return []

# --- [5. Ollama 호출 (기억력 포함)] ---
def ask_ollama(user_question):
    global chat_history

    found_answers = retrieve_top_docs(user_question)

    if found_answers:
        context = found_answers[0]
        system_msg = (
            f"당신은 스마트한 사내 안내 챗봇입니다. 아래 [정보]를 바탕으로 답변하세요.\n"
            f"[정보]: {context}\n"
            f"정보에 있는 내용은 정확히 전달하고, 정보에 없는 내용은 친절하게 모른다고 하세요."
        )
    else:
        system_msg = "관련된 사내 규정 정보는 없지만, 친절한 AI 비서로서 자연스럽게 대화를 이어가세요."

    messages = [{"role": "system", "content": system_msg}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_question})

    try:
        response = requests.post(API_URL, json={
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False
        })
        response.raise_for_status()
        bot_answer = response.json()['message']['content']

        chat_history.append({"role": "user", "content": user_question})
        chat_history.append({"role": "assistant", "content": bot_answer})

        if len(chat_history) > 20: chat_history = chat_history[-20:]

        return bot_answer

    except Exception as e:
        return f"통신 오류: {e}"

# --- [실행] ---
if __name__ == "__main__":
    print(f"\n=== {MODEL_NAME} 딥러닝(SBERT) 챗봇 ===")
    print("종료: quit / 기억 삭제: clear")

    while True:
        user = input("\n질문: ")

        if user.lower() in ["quit", "종료"]: break
        if user.lower() == "clear":
            chat_history = []
            print("▶ 대화 기억 초기화")
            continue
        if not user.strip(): continue

        sys.stdout.write("생각 중...")
        sys.stdout.flush()

        answer = ask_ollama(user)

        sys.stdout.write("\r" + " " * 20 + "\r")
        print(f"봇: {answer}")