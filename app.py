import streamlit as st
import os
import json
import itertools
from sentence_transformers import CrossEncoder 

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

# ==========================================================
# [0] 페이지 및 경로 설정
# ==========================================================
st.set_page_config(page_title="건설 CM AI 검색 엔진", page_icon="🏗️", layout="wide")

if "GOOGLE_API_KEY" not in os.environ:
    pass # Streamlit Cloud Secrets 사용

# ⚠️ [핵심] 분할된 DB 경로 및 원본 데이터 경로
DB_PATH_1 = "./chroma_db_part1"
DB_PATH_2 = "./chroma_db_part2"
JSON_DATA_PATH = "./legal_data_total_vlm.json"

# 전역 변수 (Lookup용)
RAW_DATA = []

# ==========================================================
# [1] 시스템 로딩 (Dual DB + Lookup + Hybrid)
# ==========================================================
class SimpleHybridRetriever:
    def __init__(self, bm25, chroma1, chroma2, raw_data):
        self.bm25 = bm25
        self.chroma1 = chroma1
        self.chroma2 = chroma2
        self.raw_data = raw_data
        
    def invoke(self, query):
        # 1. BM25 검색 (텍스트 매칭)
        docs_bm25 = self.bm25.invoke(query)
        
        # 2. Chroma 검색 (두 DB 동시 검색)
        docs_c1 = self.chroma1.invoke(query)
        docs_c2 = self.chroma2.invoke(query)
        
        # 3. ID -> 텍스트 변환 (Lookup)
        real_docs_chroma = []
        for doc in (docs_c1 + docs_c2):
            try:
                idx = int(doc.page_content) 
                original_item = self.raw_data[idx] 
                
                content = original_item.get('content', '').strip()
                source = original_item.get('source', '').strip()
                article = original_item.get('article', '').strip()
                full_text = f"[{source}] {content}"
                
                new_doc = Document(page_content=full_text, metadata={"source": source, "article": article})
                real_docs_chroma.append(new_doc)
            except:
                continue

        # 4. 중복 제거 및 결합
        combined = []
        seen_ids = set()
        
        for d in itertools.chain(docs_bm25, real_docs_chroma):
            key = d.page_content[:30] 
            if key not in seen_ids:
                combined.append(d)
                seen_ids.add(key)
                
        return combined[:200]

@st.cache_resource
def load_search_system():
    global RAW_DATA
    
    # 1. JSON 로드
    if not os.path.exists(JSON_DATA_PATH):
        st.error("❌ JSON 파일이 없습니다. GitHub에 올려주세요.")
        st.stop()
    with open(JSON_DATA_PATH, 'r', encoding='utf-8') as f:
        RAW_DATA = json.load(f)

    # 2. Chroma DB 로드
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    if not os.path.exists(DB_PATH_1) or not os.path.exists(DB_PATH_2):
        st.error("❌ 분할된 DB 폴더가 없습니다.")
        st.stop()

    store1 = Chroma(persist_directory=DB_PATH_1, embedding_function=embeddings, collection_name="construction_laws")
    retriever1 = store1.as_retriever(search_kwargs={"k": 100})

    store2 = Chroma(persist_directory=DB_PATH_2, embedding_function=embeddings, collection_name="construction_laws")
    retriever2 = store2.as_retriever(search_kwargs={"k": 100})

    # 3. BM25 생성
    docs = []
    for item in RAW_DATA:
        content = item.get('content', '').strip()
        source = item.get('source', '').strip()
        if not content: continue
        doc = Document(page_content=f"[{source}] {content}", metadata={"source": source, "article": item.get('article', '')})
        docs.append(doc)
    
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 150

    # 4. 결합
    hybrid_retriever = SimpleHybridRetriever(bm25_retriever, retriever1, retriever2, RAW_DATA)
    
    # [수정] 메모리 절약을 위해 가벼운 모델(TinyBERT)로 교체
    reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", model_kwargs={"torch_dtype": "auto"})

    return hybrid_retriever, reranker

with st.spinner("🚀 AI 엔진(Dual DB) 시동 중..."):
    hybrid_retriever, reranker_model = load_search_system()

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# ==========================================================
# [2] RAG 로직
# ==========================================================
spacing_chain = ChatPromptTemplate.from_template("""
    당신은 **한국어 띄어쓰기 교정 전문가**입니다.
    입력: {question}
    규칙: 설명 없이 **교정된 문장만** 출력하세요.
    교정된 문장:""").pipe(llm).pipe(StrOutputParser())

hyde_chain = ChatPromptTemplate.from_template("""
    당신은 건설 분야 **검색 키워드 추출 전문가**입니다.
    입력: {question}
    규칙: 라벨/설명 금지. 질문의 의도를 확장한 **단어들만** 나열. 콤마(,) 구분.
    """).pipe(llm).pipe(StrOutputParser())

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 건설 기준을 찾아주는 **전문 엔지니어**입니다.
    [Context]를 바탕으로 질문에 대한 명확한 답변을 작성하십시오.
    [작성 원칙]
    1. **원문 존중**: 목록(①, ②...)은 요약하지 말고 **그대로 발췌**하십시오.
    2. **해석**: 서식(( )%)이 있는 경우에만 의미를 풀어서 설명하십시오.
    3. **출처**: 끝에 [문서명] 필수 표기.
    [Context]
    {context}
    """),
    ("human", "질문: {question}")
])

def retrieve_and_rerank(query):
    initial_docs = hybrid_retriever.invoke(query)
    if not initial_docs: return []
    
    pairs = [[query, doc.page_content] for doc in initial_docs]
    scores = []
    batch_size = 16 # 가벼운 모델이라 배치 사이즈를 늘려도 됨
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        batch_scores = reranker_model.predict(batch)
        scores.extend(batch_scores)
        
    scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:50]]

def format_docs(docs):
    return "\n\n".join([f"<출처: {d.metadata.get('source')} / {d.metadata.get('article')}>\n{d.page_content}" for d in docs])

rag_chain = (
    {
        "context": RunnableLambda(retrieve_and_rerank) | format_docs,
        "question": RunnablePassthrough() 
    }
    | answer_prompt
    | llm
    | StrOutputParser()
)

# ==========================================================
# [3] 웹 UI
# ==========================================================
st.title("🏗️ 건설 CM 전문 AI")
st.caption("🚀 1차 직구 검색(Direct) 후 → 원하면 HyDE 심층 검색(Expansion)으로 이어집니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # [Phase 1] 1차 검색
        with st.status("🔍 1차 정밀 검색 진행 중...", expanded=True) as status:
            st.write("🔧 띄어쓰기 교정 중...")
            corrected_query = spacing_chain.invoke({"question": prompt})
            st.write(f"-> 교정됨: `{corrected_query}`")
            
            st.write("📚 문서 통합 검색 중...")
            response_1 = rag_chain.invoke(corrected_query)
            status.update(label="✅ 1차 검색 완료!", state="complete", expanded=False)
        
        st.markdown("### 🤖 1차 답변")
        st.markdown(response_1)
        st.session_state.messages.append({"role": "assistant", "content": f"### 🤖 1차 답변\n{response_1}"})

        # [Phase 2] HyDE 확장 (로딩 UI 개선)
        with st.expander("🤔 답변이 부족한가요? (심층 검색)"):
            if st.button("🚀 HyDE 심층 검색 실행"):
                # [수정] 1차 검색처럼 'status' 박스를 사용해 진행상황 표시
                with st.status("🧠 전문가 모드(HyDE) 가동 중...", expanded=True) as status_2:
                    st.write("💡 질문의 의도를 분석하여 키워드를 확장합니다...")
                    hyde_keywords = hyde_chain.invoke({"question": corrected_query})
                    final_query = f"{corrected_query} {hyde_keywords}"
                    st.write(f"-> 확장된 검색어: `{final_query}`")
                    
                    st.write("🚀 확장된 범위로 재검색 및 정밀 심사 중...")
                    response_2 = rag_chain.invoke(final_query)
                    status_2.update(label="✅ 심층 검색 완료!", state="complete", expanded=False)

                st.success(f"확장된 키워드: {hyde_keywords}")
                st.markdown("---")
                st.markdown("### 🤖 2차 상세 답변 (HyDE)")
                st.markdown(response_2)
                
                st.session_state.messages.append({"role": "assistant", "content": f"### 🤖 2차 상세 답변 (HyDE)\n{response_2}"})