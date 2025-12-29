import streamlit as st
import os
import json
import itertools
import base64
import tempfile
import platform 
from pdf2image import convert_from_path
from sentence_transformers import CrossEncoder 

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# ==========================================================
# [0] 기본 설정
# ==========================================================
st.set_page_config(page_title="건설 CM AI 통합 솔루션", page_icon="🏗️", layout="wide")

# 1. API 키 가져오기 (가장 안전한 방법)
# Secrets에서 가져오되, 없으면 환경변수 확인, 그래도 없으면 에러 처리
if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
elif "GOOGLE_API_KEY" in os.environ:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
else:
    st.error("🚨 치명적 오류: Google API Key를 찾을 수 없습니다. Streamlit Secrets 설정을 확인하세요.")
    st.stop()

# 2. Poppler 경로 (자동 감지)
system_name = platform.system()
if system_name == "Windows":
    POPPLER_PATH = r"C:\Users\owner\myvenv\Release-25.12.0-0\poppler-25.12.0\Library\bin"
else:
    POPPLER_PATH = None 

# 3. 데이터 경로
DB_PATH_1 = "./chroma_db_part1"
DB_PATH_2 = "./chroma_db_part2"
JSON_DATA_PATH = "./legal_data_total_vlm.json"
RAW_DATA = []

# ==========================================================
# [1] 시스템 로딩
# ==========================================================
class SimpleHybridRetriever:
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
                content = original_item.get('content', '').strip()
                source = original_item.get('source', '').strip()
                article = original_item.get('article', '').strip()
                full_text = f"[{source}] {content}"
                new_doc = Document(page_content=full_text, metadata={"source": source, "article": article})
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
    if not os.path.exists(JSON_DATA_PATH):
        st.error("❌ JSON 데이터 파일이 없습니다.")
        st.stop()
    with open(JSON_DATA_PATH, 'r', encoding='utf-8') as f:
        RAW_DATA = json.load(f)

    # [핵심 수정] API 키를 직접 전달합니다.
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=GOOGLE_API_KEY
    )
    
    if not os.path.exists(DB_PATH_1) or not os.path.exists(DB_PATH_2):
        st.error("❌ DB 폴더가 없습니다.")
        st.stop()

    store1 = Chroma(persist_directory=DB_PATH_1, embedding_function=embeddings, collection_name="construction_laws")
    retriever1 = store1.as_retriever(search_kwargs={"k": 100})
    store2 = Chroma(persist_directory=DB_PATH_2, embedding_function=embeddings, collection_name="construction_laws")
    retriever2 = store2.as_retriever(search_kwargs={"k": 100})

    docs = []
    for item in RAW_DATA:
        content = item.get('content', '').strip()
        source = item.get('source', '').strip()
        if not content: continue
        doc = Document(page_content=f"[{source}] {content}", metadata={"source": source, "article": item.get('article', '')})
        docs.append(doc)
    
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 150

    hybrid_retriever = SimpleHybridRetriever(bm25_retriever, retriever1, retriever2, RAW_DATA)
    reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", model_kwargs={"torch_dtype": "auto"})

    return hybrid_retriever, reranker

with st.spinner("🚀 AI 통합 엔진(Text+Vision) 시동 중..."):
    try:
        hybrid_retriever, reranker_model = load_search_system()
    except Exception as e:
        st.error(f"시스템 로딩 실패: {e}")
        st.stop()

# [핵심 수정] LLM 초기화 시에도 API 키 직접 전달
llm_text = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0, 
    google_api_key=GOOGLE_API_KEY
)
llm_vision = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0, 
    google_api_key=GOOGLE_API_KEY
)

# ==========================================================
# [2] 로직 체인
# ==========================================================
spacing_chain = ChatPromptTemplate.from_template("교정된 한국어 문장만 출력(설명X): {question}").pipe(llm_text).pipe(StrOutputParser())
hyde_chain = ChatPromptTemplate.from_template("건설 전문 검색 키워드 5개 나열(콤마 구분, 설명X): {question}").pipe(llm_text).pipe(StrOutputParser())

def retrieve_and_rerank(query, top_k=5):
    initial_docs = hybrid_retriever.invoke(query)
    if not initial_docs: return []
    pairs = [[query, doc.page_content] for doc in initial_docs]
    scores = []
    batch_size = 16
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        batch_scores = reranker_model.predict(batch)
        scores.extend(batch_scores)
    scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_k]]

def analyze_drawing_deep(image_base64, query, retrieved_docs):
    laws_text = "\n".join([f"- {d.page_content}" for d in retrieved_docs])
    prompt_text = f"""
    당신은 건축 도면 검토 및 법규 분석 전문가입니다.
    
    [분석 요청 사항]
    {query}
    
    [관련 법규 데이터베이스]
    {laws_text}
    
    [지시사항]
    1. **자동 정밀 분석:** 첨부된 도면을 보고 위 법규 데이터베이스를 참조하여, 법규 위반 가능성이 있는 모든 요소를 찾아내세요.
    2. **공간 및 치수 확인:** 방, 거실, 복도, 계단 등의 치수와 배치가 기준에 적합한지 확인하세요.
    3. **결과 보고:** - ✅ 적합한 항목
       - ⚠️ 위반 의심 항목 (구체적인 위치와 이유 설명)
       - ❓ 판단 불가 항목 (이유 설명)
    """
    message = HumanMessage(content=[
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
    ])
    return llm_vision.invoke([message]).content

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "건설 기준 엔지니어입니다. [Context]를 보고 답변하세요. 출처 표기 필수.\n[Context]\n{context}"),
    ("human", "질문: {question}")
])
def format_docs(docs):
    return "\n\n".join([f"<출처: {d.metadata.get('source')} / {d.metadata.get('article')}>\n{d.page_content}" for d in docs])

rag_chain = (
    {"context": RunnableLambda(lambda x: retrieve_and_rerank(x, top_k=20)) | format_docs, "question": RunnablePassthrough()}
    | answer_prompt | llm_text | StrOutputParser()
)

# ==========================================================
# [3] 웹 UI 구성
# ==========================================================
st.title("🏗️ 건설 CM 전문 AI (도면 + 법규)")

# --- [A] 사이드바: PDF 업로드 및 상태 관리 ---
if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

with st.sidebar:
    st.header("📂 도면 투입구")
    st.info("💡 PDF를 넣으면 **즉시 심층 분석**이 시작됩니다.")
    uploaded_files = st.file_uploader("검토할 도면 PDF를 선택하세요", type=["pdf"], accept_multiple_files=True)
    
    current_image_base64 = None
    
    if uploaded_files:
        target_file = uploaded_files[0]
        # 파일이 새로 바뀌었는지 확인
        if st.session_state.last_processed_file != target_file.name:
             st.session_state.analysis_result = None # 결과 초기화
             
        st.write(f"📄 파일: **{target_file.name}**")
        
        with st.spinner("이미지 변환 중..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(target_file.read())
                tmp_path = tmp_file.name
            try:
                images = convert_from_path(tmp_path, poppler_path=POPPLER_PATH, first_page=1, last_page=1)
                if images:
                    st.image(images[0], caption="검토 대상 도면", use_container_width=True)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                        images[0].save(tmp_img.name, "JPEG")
                        with open(tmp_img.name, "rb") as f:
                            current_image_base64 = base64.b64encode(f.read()).decode("utf-8")
                else:
                    st.error("이미지 변환 실패")
            except Exception as e:
                st.error(f"오류: {e}")

# --- [B] 자동 심층 분석 트리거 ---
if current_image_base64 and st.session_state.analysis_result is None:
    target_file_name = uploaded_files[0].name
    
    # 1. 자동 분석 시작
    with st.status(f"🚀 '{target_file_name}' 도면 자동 심층 분석 중...", expanded=True) as status:
        st.write("🔧 도면의 주요 공간 및 법규 검토 항목 식별 중...")
        # 자동 질문 생성
        auto_query = "건축 도면의 주요 치수(복도, 계단, 거실 등)와 소방/피난 설비가 건축 법규에 적합한지 포괄적으로 검토해줘."
        
        st.write("📚 관련 법규(복도, 계단, 소방 등) 검색 및 매핑 중...")
        retrieved_docs = retrieve_and_rerank(auto_query, top_k=7) 
        
        st.write("👀 Vision AI가 도면 정밀 계측 및 법규 대조 수행 중...")
        vision_result = analyze_drawing_deep(current_image_base64, auto_query, retrieved_docs)
        
        # 결과 저장
        final_report = f"### 🏗️ 도면 자동 심층 분석 결과\n**분석 대상:** {target_file_name}\n\n{vision_result}\n\n---\n**[참고한 법규]**\n" + "\n".join([f"- {d.metadata.get('source')} ({d.metadata.get('article')})" for d in retrieved_docs])
        
        st.session_state.analysis_result = final_report
        st.session_state.last_processed_file = target_file_name
        
        status.update(label="✅ 분석 완료! 아래 결과를 확인하세요.", state="complete", expanded=False)

# --- [C] 결과 표시 및 채팅 인터페이스 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 자동 분석 결과가 있으면 채팅창에 가장 먼저 박제
if st.session_state.analysis_result:
    if not st.session_state.messages or st.session_state.messages[-1]["content"] != st.session_state.analysis_result:
        st.session_state.messages.append({"role": "assistant", "content": st.session_state.analysis_result})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- [D] 추가 질문 입력 ---
if prompt := st.chat_input("추가 질문이 있으신가요? (예: 계단 치수만 다시 확인해줘)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if current_image_base64:
            with st.status("🔍 추가 질의 분석 중...", expanded=True) as status:
                st.write("📚 관련 법규 재검색...")
                retrieved_docs = retrieve_and_rerank(prompt, top_k=5)
                st.write("👀 도면 재확인...")
                vision_res = analyze_drawing_deep(current_image_base64, prompt, retrieved_docs)
                status.update(label="✅ 답변 완료", state="complete")
            
            final_res = f"{vision_res}\n\n[참고 법규]: " + ", ".join([d.metadata.get('article') for d in retrieved_docs])
            st.markdown(final_res)
            st.session_state.messages.append({"role": "assistant", "content": final_res})
        
        else:
            corrected = spacing_chain.invoke({"question": prompt})
            response = rag_chain.invoke(corrected)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})