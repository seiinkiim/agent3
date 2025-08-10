# app.py
# --------------------------- Imports ---------------------------
import os
import re
import random
from collections import deque

import pandas as pd
import streamlit as st

# LangChain / LLM
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

# RAG
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 너의 스트리밍 핸들러
from utils import StreamHandler


# --------------------------- Streamlit 기본 ---------------------------
st.set_page_config(page_title="운동화 쇼핑 에이전트")
st.title("운동화 쇼핑 에이전트")
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# 세션 상태
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "store" not in st.session_state:
    st.session_state["store"] = dict()
if "selected_question" not in st.session_state:
    st.session_state["selected_question"] = None
if "followup_step" not in st.session_state:
    st.session_state["followup_step"] = 0
if "seen_products" not in st.session_state:
    st.session_state["seen_products"] = set()  # "브랜드||제품명"
if "random_pool" not in st.session_state:
    st.session_state["random_pool"] = None  # 최초 로드 후 채움

SESSION_ID = "sneaker-chat"
CSV_PATH = "shoes_top12.csv"  # 파일명 맞게 변경 가능


# --------------------------- 후속질문(고정 6개) ---------------------------
followup_set_1 = {
    "Q1": "가벼운 운동화는 어떤 제품이 있나요?",
    "Q2": "무게감 있고 안정감 있는 제품은 무엇이 있나요?",
    "Q3": "통풍이 좋은 운동화 제품은 무엇이 있나요?",
}
followup_set_2 = {
    "Q4": "쿠션감이 좋은 제품은 무엇이 있나요?",
    "Q5": "평평한 운동화는 무엇이 있나요?",
    "Q6": "약간의 굽이 있는 제품은 무엇이 있나요?",
}


# --------------------------- 데이터/RAG 준비 ---------------------------
@st.cache_resource(show_spinner=True)
def load_product_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    # 구매링크 없는 행 제외
    df = df[df["구매링크"].notna() & (df["구매링크"].astype(str).str.strip() != "")].copy()
    df.reset_index(drop=True, inplace=True)
    return df

df_products = load_product_df(CSV_PATH)

# 세션 랜덤 풀 초기화
if st.session_state["random_pool"] is None:
    st.session_state["random_pool"] = deque(range(len(df_products)))
    random.shuffle(st.session_state["random_pool"])


@st.cache_resource(show_spinner=True)
def build_retriever(csv_path: str):
    loader = CSVLoader(csv_path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = splitter.split_documents(docs)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(splits, embedding=embedding)
    # 후보 폭을 넉넉히
    return vs.as_retriever(search_kwargs={"k": 12})

retriever = build_retriever(CSV_PATH)


# --------------------------- 유틸 함수 ---------------------------
def product_key(brand: str, name: str) -> str:
    return f"{str(brand).strip().lower()}||{str(name).strip().lower()}"


def draw_random_products(n: int = 3) -> str:
    """세션 내 중복 없이 무작위 n개 추출해서 표시하고, seen에 등록"""
    pool: deque = st.session_state["random_pool"]
    chosen_idx = []
    target = min(n, len(df_products))
    while len(chosen_idx) < target:
        if not pool:  # 전부 소진하면 다시 섞어서 순환
            pool.extend(range(len(df_products)))
            random.shuffle(pool)
        # 이미 본 제품은 건너뛰기
        idx = pool.popleft()
        row = df_products.iloc[idx]
        key = product_key(row["브랜드"], row["제품명"])
        if key in st.session_state["seen_products"]:
            continue
        chosen_idx.append(idx)

    lines = []
    for i, idx in enumerate(chosen_idx, start=1):
        row = df_products.iloc[idx]
        st.session_state["seen_products"].add(product_key(row["브랜드"], row["제품명"]))
        lines.append(
            f"{i}. {row['브랜드']} {row['제품명']} | {row['가격']} | {row['제품설명']} | {row['구매링크']}"
        )
    return "\n".join(lines)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state["store"]:
        st.session_state["store"][session_id] = ChatMessageHistory()
    return st.session_state["store"][session_id]


def build_query_from_history_and_input(
    history: BaseChatMessageHistory, user_input: str, max_turns: int = 4
) -> str:
    msgs = history.messages[-max_turns * 2 :] if hasattr(history, "messages") else []
    hist_text = []
    for m in msgs:
        role = getattr(m, "type", getattr(m, "role", ""))
        content = getattr(m, "content", "")
        if role in ("human", "user", "ai", "assistant"):
            hist_text.append(f"{role}: {content}")
    hist_blob = "\n".join(hist_text)
    return f"{hist_blob}\nuser: {user_input}\n\n요약 키워드: 운동 목적, 쿠션, 통풍, 경량/안정, 굽 높이, 브랜드 선호"


def filter_unseen_docs(docs):
    """세션에서 이미 본 상품 제거"""
    filtered = []
    for d in docs:
        t = d.page_content
        brand_m = re.search(r"브랜드\s*[:=]\s*(.+)", t)
        name_m = re.search(r"제품명\s*[:=]\s*(.+)", t)
        brand = brand_m.group(1).strip() if brand_m else ""
        name = name_m.group(1).strip() if name_m else ""
        if product_key(brand, name) not in st.session_state["seen_products"]:
            filtered.append(d)
    return filtered


def docs_to_rows(docs):
    """문서 -> 구조화 dict 리스트"""
    items = []
    for d in docs:
        t = d.page_content

        def grab(field):
            m = re.search(rf"{field}\s*[:=]\s*(.+)", t)
            return m.group(1).strip() if m else ""

        brand = grab("브랜드")
        name = grab("제품명")
        price = grab("가격")
        desc = grab("제품설명")
        url = grab("구매링크")
        if not url:
            continue
        items.append({"brand": brand, "name": name, "price": price, "desc": desc, "url": url})
    return items


def topup_with_unseen(rows, need: int):
    """rows가 3개 미만이면 CSV에서 아직 안 본 제품으로 채우고, 그래도 부족하면 '본 제품'로도 채워 반드시 3개 보장"""
    if need <= 0:
        return rows
    have_keys = {product_key(r["brand"], r["name"]) for r in rows}

    unseen = []
    seen = []
    for _, r in df_products.iterrows():
        key = product_key(r["브랜드"], r["제품명"])
        item = {
            "brand": r["브랜드"], "name": r["제품명"], "price": r["가격"],
            "desc": r["제품설명"], "url": r["구매링크"]
        }
        if key in have_keys:
            continue
        if key in st.session_state["seen_products"]:
            seen.append(item)
        else:
            unseen.append(item)

    random.shuffle(unseen)
    random.shuffle(seen)

    take = min(need, len(unseen))
    rows.extend(unseen[:take])
    need -= take

    if need > 0 and len(seen) > 0:
        rows.extend(seen[:need])

    return rows


def rows_to_context(rows):
    """dict 리스트 -> 컨텍스트 문자열"""
    return "\n".join(
        f"브랜드:{r['brand']} | 제품명:{r['name']} | 가격:{r['price']} | 설명:{r['desc']} | 구매링크:{r['url']}"
        for r in rows
    )


def build_banlist():
    """프롬프트에 전달할 재추천 금지 목록(가벼운 방어막)"""
    if not st.session_state["seen_products"]:
        return "없음"
    lines = []
    for key in st.session_state["seen_products"]:
        b, n = key.split("||", 1)
        lines.append(f"- {b.title()} {n}")
    return "\n".join(lines)


def force_three_list_output(rows, text: str) -> str:
    """모델 출력이 조건을 어기면, rows 기반으로 3줄을 강제로 생성해 반환"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    list_lines = [ln for ln in lines if ln.startswith("1.") or ln.startswith("2.") or ln.startswith("3.")]
    if len(list_lines) >= 3 and all("|" in ln for ln in list_lines[:3]):
        return "\n".join(list_lines[:3])
    rebuilt = []
    for i, r in enumerate(rows[:3], start=1):
        rebuilt.append(f"{i}. {r['brand']} {r['name']} | {r['price']} | {r['desc']} | {r['url']}")
    return "\n".join(rebuilt)


# --------------------------- 프롬프트 ---------------------------
SYSTEM_PROMPT = """당신은 운동화 쇼핑 에이전트입니다.
- 추천은 반드시 [컨텍스트]의 제품만 사용하세요. (컨텍스트 외 임의 추론 금지)
- '구매링크'가 없는 제품은 추천하지 마세요.
- 아래 금지 문구는 절대 출력하지 마세요: "정보가 없습니다", "컨텍스트에 따르면 제공되지 않습니다", "추천할 수 없습니다", "죄송합니다" 등 사과/부족안내/메타언급.

[출력 형식 — 정확히 3개]
1. [브랜드] [제품명] | [가격] | [설명] | [구매링크]
2. ...
3. ...
- [설명]에는 사용자가 언급한 조건(경량/안정, 통풍, 쿠션, 굽 높이 등)을 반영하세요.

[이미 추천된 제품 — 이번 세션에서 다시 추천 금지]
{banlist}

[컨텍스트]
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

llm = ChatOpenAI(model_name="gpt-4o", streaming=True)
chain = prompt | llm | StrOutputParser()
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


# --------------------------- 이전 대화 출력 ---------------------------
for role, msg in st.session_state["messages"]:
    st.chat_message(role).write(msg)


# --------------------------- 입력 ---------------------------
user_input = None
if st.session_state["selected_question"]:
    user_input = st.session_state["selected_question"]
    st.session_state["selected_question"] = None
else:
    tmp = st.chat_input("메시지를 입력해 주세요")
    if tmp:
        user_input = tmp


# --------------------------- 응답 처리 ---------------------------
if user_input:
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append(("user", user_input))

    # 첫 요청: 무작위 3개 (세션 중복 방지)
    if user_input.strip() == "운동화 추천해줘" and st.session_state["followup_step"] == 0:
        random_reco = draw_random_products(3)
        st.chat_message("assistant").write(random_reco)
        st.session_state["messages"].append(("assistant", random_reco))
        st.session_state["followup_step"] = 1  # 패널 1세트 노출

    # 이후: RAG 추천 (항상 3개 보장 + 형식 가드)
    else:
        history = get_session_history(SESSION_ID)
        query = build_query_from_history_and_input(history, user_input)
        rag_docs = retriever.get_relevant_documents(query)

        # 1) 이미 본 상품 제외
        rag_docs_filtered = filter_unseen_docs(rag_docs)

        # 2) 문서 -> 행 변환
        rows = docs_to_rows(rag_docs_filtered)

        # 3) 부족하면 CSV unseen/seen으로 보충(항상 3개)
        if len(rows) < 3:
            rows = topup_with_unseen(rows, 3 - len(rows))
        rows = rows[:3]  # 안전 가드

        # 4) 컨텍스트 생성
        context = rows_to_context(rows)

        # 5) seen 등록(모델 출력과 상관없이 중복 방지)
        for r in rows:
            st.session_state["seen_products"].add(product_key(r["brand"], r["name"]))

        # 6) 호출 + 형식 가드 적용
        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            raw_text = chain_with_memory.invoke(
                {"question": user_input, "context": context, "banlist": build_banlist()},
                config={"configurable": {"session_id": SESSION_ID}, "callbacks": [stream_handler]},
            )

        response_text = force_three_list_output(rows, raw_text)
        st.session_state["messages"].append(("assistant", response_text))

        # 패널 단계 진행(1세트 → 2세트 → 마지막 숨김 + 인증번호 안내)
        if st.session_state["followup_step"] == 1:
            st.session_state["followup_step"] = 2
        elif st.session_state["followup_step"] == 2:
            st.session_state["followup_step"] = 3
            code = f"{random.randint(0, 9999):04d}"
            end_msg = f"대화가 종료되었습니다. 인증번호는 {code} 입니다"
            st.chat_message("assistant").write(end_msg)
            st.session_state["messages"].append(("assistant", end_msg))


# --------------------------- 후속질문 패널 ---------------------------
def render_followup_panel(step: int):
    # step 1, 2일 때만 노출 (마지막엔 숨김)
    if step not in (1, 2):
        return
    st.markdown("### 이런 질문도 해보세요!")
    qset = followup_set_1 if step == 1 else followup_set_2
    for key, question in qset.items():
        col_q, col_btn = st.columns([8, 1])
        col_q.markdown(f"**{key}.** {question}")
        if col_btn.button("➕", key=f"btn_{key}"):
            st.session_state["selected_question"] = question
            st.rerun()

render_followup_panel(st.session_state["followup_step"])
