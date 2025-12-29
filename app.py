import streamlit as st
import os
import json
import itertools
import base64
import tempfile
import platform 
import time

# [필수 라이브러리]
from pdf2image import convert_from_path
from sentence_transformers import CrossEncoder 
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# ==========================================================
# [0] 기본 설정 및 상수 정의
# ==========================================================
st.set_page_config(page_title="건설 CM AI 파트너", page_icon="🏗️", layout="wide")

# 1. API 키 설정
if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
elif "GOOGLE_API_KEY" in os.environ:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
else:
    st.error("🚨 Google API Key가 없습니다.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# 2. Poppler 경로 (사용자 환경)
system_name = platform.system()
if system_name == "Windows":
    # 사용자의 로컬 환경 경로 (환경에 맞게 수정 필요)
    POPPLER_PATH = r"C:\Users\owner\myvenv\Release-25.12.0-0\poppler-25.12.0\Library\bin"
else:
    POPPLER_PATH = None 

# 3. 데이터 경로
DB_PATH_1 = "./chroma_db_part1"
DB_PATH_2 = "./chroma_db_part2"
JSON_DATA_PATH = "./legal_data_total_vlm.json"
RAW_DATA = []
MODEL_NAME = "models/gemini-2.5-flash" 

# ==========================================================
# [1] 검색 엔진 로딩 (Retriever & Reranker)
# ==========================================================
class SimpleHybridRetriever:
    """BM25 + ChromaDB 하이브리드 검색기"""
    def __init__(self, bm25, chroma1, chroma2, raw_data):
        self.bm25 = bm25
        self.chroma1 = chroma1
        self.chroma2 = chroma2
        self.raw_data = raw_data
        
    def invoke(self, query):
        docs_bm25 = self.bm25.invoke(query)
        docs_c1 = self.chroma1.invoke(query)
        docs_c2 = self.chroma2.invoke(query)
        
        real_docs_chroma = []
        for doc in (docs_c1 + docs_c2):
            try:
                idx = int(doc.page_content) 
                original_item = self.raw_data[idx] 
                full_text = f"[{original_item['source']}] {original_item['content']}"
                new_doc = Document(page_content=full_text, metadata={"source": original_item['source'], "article": original_item['article']})
                real_docs_chroma.append(new_doc)
            except:
                continue

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
    if not os.path.exists(JSON_DATA_PATH): return None, None
    with open(JSON_DATA_PATH, 'r', encoding='utf-8') as f: RAW_DATA = json.load(f)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    
    if not os.path.exists(DB_PATH_1): return None, None

    store1 = Chroma(persist_directory=DB_PATH_1, embedding_function=embeddings, collection_name="construction_laws")
    store2 = Chroma(persist_directory=DB_PATH_2, embedding_function=embeddings, collection_name="construction_laws")
    
    docs = [Document(page_content=f"[{i['source']}] {i['content']}", metadata={"source": i['source']}) for i in RAW_DATA if i['content']]
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 100
    
    hybrid = SimpleHybridRetriever(bm25, store1.as_retriever(), store2.as_retriever(), RAW_DATA)
    reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
    return hybrid, reranker

with st.spinner("🚀 AI 시스템 로딩 중..."):
    hybrid_retriever, reranker_model = load_search_system()
    if not hybrid_retriever:
        st.error("데이터 파일이나 DB를 찾을 수 없습니다.")
        st.stop()

# LLM 설정
safety = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
llm_text = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.1, google_api_key=GOOGLE_API_KEY, safety_settings=safety)
llm_vision = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, google_api_key=GOOGLE_API_KEY, safety_settings=safety)

# ==========================================================
# [2] RAG 로직 (항상 Deep RAG 사용)
# ==========================================================
# (1) 쿼리 확장
expansion_chain = ChatPromptTemplate.from_template(
    "건설 검색 최적화 AI입니다. '{question}'에 대한 검색어 3개를 '질문|키워드1,키워드2,키워드3' 형식으로 만드세요."
) | llm_text | StrOutputParser()

def get_expanded_queries(query):
    try:
        res = expansion_chain.invoke({"question": query})
        if "|" in res: return [res.split("|")[0]] + res.split("|")[1].split(",")
        return [query]
    except: return [query]

# (2) 문서 검색 및 순위 재조정 (Deep Only)
def retrieve_docs_deep(query):
    # 항상 5단계 심층 검색 수행 (확장 + Top-50 + Rerank)
    queries = get_expanded_queries(query)
    top_k = 50
    
    all_docs = []
    seen = set()
    for q in queries:
        for doc in hybrid_retriever.invoke(q):
            if doc.page_content not in seen:
                all_docs.append(doc)
                seen.add(doc.page_content)
                
    if not all_docs: return []
    
    # Rerank 수행 (배치 처리)
    pairs = [[query, d.page_content] for d in all_docs]
    scores = []
    for i in range(0, len(pairs), 32):
        batch = pairs[i : i+32]
        scores.extend(reranker_model.predict(batch))
    scored = sorted(zip(all_docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, s in scored[:top_k]]

# (3) 답변 생성
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 건설 CM 기술사입니다.
    [Context]를 근거로 답변하되, 정보가 없으면 일반적인 공학적 지식과 시방서 기준을 활용하여 구체적으로 조언하세요.
    절대 "모릅니다"라고 끝내지 마세요.
    \n[Context]\n{context}
    """),
    ("human", "{question}")
])
def format_docs(docs): return "\n\n".join([d.page_content for d in docs])

# ==========================================================
# [3] Vision AI & 보고서 로직
# ==========================================================
def analyze_page_detail(image_base64, query):
    msg = HumanMessage(content=[
        {"type": "text", "text": f"건설 전문가로서 도면을 보고 '{query}'에 대해 분석하세요. 문제점을 짧고 명확하게(1~2문장) 지적하세요."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
    ])
    try: return llm_vision.invoke([msg]).content
    except: return "분석 불가"

def generate_consolidated_report(filename, page_results):
    # 페이지별 내용을 합쳐서 LLM에게 줌
    raw_text = "\n".join([f"Page {p['page']}: {p['content']}" for p in page_results])
    
    prompt = f"""
    당신은 CM단장입니다. '{filename}' 도면 분석 결과를 바탕으로 **종합 리포트**를 작성하세요.
    1. 전체적인 시공 리스크 총평
    2. 공종별 주요 관리 포인트
    (※ 개별 페이지 내용은 여기서 나열하지 말고, 전체적인 흐름만 요약하세요.)
    
    [분석 데이터]
    {raw_text}
    """
    summary = llm_text.invoke(prompt).content
    
    # [핵심] 페이지별 리스트는 강제로 붙임
    appendix = "\n\n---\n### 🔎 [부록] 페이지별 진단 요약\n"
    for p in page_results:
        clean_msg = p['content'].replace('\n', ' ').strip()
        appendix += f"- **Page {p['page']}**: {clean_msg}\n"
        
    return summary + appendix

# ==========================================================
# [4] UI 구성
# ==========================================================
# 세션 상태
if "messages" not in st.session_state: st.session_state.messages = []
if "processed_files" not in st.session_state: st.session_state.processed_files = set()
if "current_image_base64" not in st.session_state: st.session_state.current_image_base64 = None

# --- 사이드바: 동적 모드 선택 ---
with st.sidebar:
    st.header("📂 도면 투입구")
    uploaded_files = st.file_uploader("PDF 도면 업로드", type=["pdf"], accept_multiple_files=True)
    st.markdown("---")
    
    # [수정된 로직]
    if uploaded_files:
        st.subheader("🤖 질문 설정")
        # Case A: 도면 있음 -> '도면 보기' vs '법규 찾기' 선택
        search_mode = st.radio(
            "모드 선택", 
            ["📂 도면 관련 질문", "⚖️ 건축 법규 검색"],
            help="📂 도면 관련: 보고 있는 도면 내용 분석\n⚖️ 법규 검색: 도면 무시하고 법규 DB 심층 검색"
        )
    else:
        # Case B: 도면 없음 -> 버튼 숨김 + 심층 모드 고정
        search_mode = "⚖️ 건축 법규 검색"
        st.info("💡 **심층 법규 검색 모드**가 활성화되었습니다.\n(Query Expansion + Rerank 자동 적용)")

# --- 도면 분석 프로세스 ---
if uploaded_files:
    for f in uploaded_files:
        if f.name not in st.session_state.processed_files:
            with st.status(f"📄 '{f.name}' 정밀 분석 중...", expanded=True) as status:
                # 1. 변환
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.read())
                    path = tmp.name
                
                try:
                    pages = convert_from_path(path, poppler_path=POPPLER_PATH)
                except:
                    st.error("PDF 변환 실패"); continue
                
                # 2. Vision Loop
                page_results = []
                prog_bar = st.progress(0)
                
                for i, p in enumerate(pages):
                    prog_bar.progress((i+1)/len(pages), text=f"🔍 Page {i+1} 결함 탐지 중...")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img:
                        p.save(img.name, "JPEG")
                        with open(img.name, "rb") as r:
                            b64 = base64.b64encode(r.read()).decode("utf-8")
                    
                    st.session_state.current_image_base64 = b64 # 마지막 페이지 저장
                    
                    # 분석
                    res = analyze_page_detail(b64, "시공 품질 및 안전 위험 요소")
                    page_results.append({"page": i+1, "content": res})
                    time.sleep(0.1)

                # 3. 종합 보고서 (+부록 강제 병합)
                status.write("📝 보고서 생성 중...")
                final_report = generate_consolidated_report(f.name, page_results)
                
                st.session_state.processed_files.add(f.name)
                st.session_state.messages.append({"role": "assistant", "content": final_report})
                
                prog_bar.empty()
                status.update(label="분석 완료", state="complete")

# --- 메인 채팅창 ---
st.title("🏗️ 건설 CM AI 파트너")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        response = ""
        
        # [로직 분기]
        # 1. Vision Mode (도면 관련 질문 선택 시)
        if search_mode == "📂 도면 관련 질문" and st.session_state.current_image_base64:
            with st.spinner("👁️ 도면을 검토하고 있습니다..."):
                msg = HumanMessage(content=[
                    {"type": "text", "text": f"질문: {prompt}\n(이전 맥락과 도면을 참고하여 기술적으로 답변하세요)"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{st.session_state.current_image_base64}"}}
                ])
                response = llm_vision.invoke([msg]).content

        # 2. Deep RAG Mode (그 외 모든 경우: 도면 없을 때 or 법규 검색 모드)
        else:
            # UI 상 메시지 표시 (심층 검색 중임을 명시)
            with st.status("🧠 심층 검색 중 (Query Expansion + Rerank)...", expanded=True):
                context_docs = retrieve_docs_deep(prompt)
                context_text = format_docs(context_docs)
                response = answer_prompt.pipe(llm_text).pipe(StrOutputParser()).invoke({"context": context_text, "question": prompt})

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})