import streamlit as st
import os
import json
import itertools
import base64
import tempfile
import platform  # [핵심] 운영체제 감지용 라이브러리
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
# [0] 기본 설정 (경로 및 API)
# ==========================================================
st.set_page_config(page_title="건설 CM AI 통합 솔루션", page_icon="🏗️", layout="wide")

# 1. API 키 (환경변수 또는 직접 입력)
if "GOOGLE_API_KEY" not in os.environ:
    # os.environ["GOOGLE_API_KEY"] = "여기에_키를_넣으세요" # Streamlit Secrets를 쓴다면 주석 유지
    pass

# 2. Poppler 경로 설정 (자동 감지 로직)
# [중요] 윈도우와 리눅스(서버)를 구분하여 경로를 설정합니다.
system_name = platform.system()

if system_name == "Windows":
    # 사용자 로컬 컴퓨터용 경로
    POPPLER_PATH = r"C:\Users\owner\myvenv\Release-25.12.0-0\poppler-25.12.0\Library\bin"
else:
    # Streamlit Cloud (Linux) 서버용 
    # packages.txt를 통해 설치된 poppler-utils는 시스템 PATH에 등록되므로 경로 지정이 필요 없습니다(None).
    POPPLER_PATH = None 

# 3. 데이터 경로
DB_PATH_1 = "./chroma_db_part1"
DB_PATH_2 = "./chroma_db_part2"
JSON_DATA_PATH = "./legal_data_total_vlm.json"

# 전역 변수
RAW_DATA = []

# ==========================================================
# [1] 시스템 로딩 (DB + Hybrid Search + Vision)
# ==========================================================
class SimpleHybridRetriever:
    def __init__(self, bm25, chroma1, chroma2, raw_data):
        self.bm25 = bm25
        self.chroma1 = chroma1
        self.chroma2 = chroma2
        self.raw_data = raw_data
        
    def invoke(self, query):
        # 1. BM25 & Chroma 검색
        docs_bm25 = self.bm25.invoke(query)
        docs_c1 = self.chroma1.invoke(query)
        docs_c2 = self.chroma2.invoke(query)
        
        # 2. ID -> 원본 텍스트 복원
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

        # 3. 중복 제거
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
    
    # JSON 로드
    if not os.path.exists(JSON_DATA_PATH):
        st.error("❌ JSON 데이터 파일이 없습니다.")
        st.stop()
    with open(JSON_DATA_PATH, 'r', encoding='utf-8') as f:
        RAW_DATA = json.load(f)

    # 임베딩 & DB 로드
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    if not os.path.exists(DB_PATH_1) or not os.path.exists(DB_PATH_2):
        st.error("❌ DB 폴더가 없습니다.")
        st.stop()

    store1 = Chroma(persist_directory=DB_PATH_1, embedding_function=embeddings, collection_name="construction_laws")
    retriever1 = store1.as_retriever(search_kwargs={"k": 100})
    store2 = Chroma(persist_directory=DB_PATH_2, embedding_function=embeddings, collection_name="construction_laws")
    retriever2 = store2.as_retriever(search_kwargs={"k": 100})

    # BM25 생성
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

# 시스템 초기화
with st.spinner("🚀 AI 통합 엔진(Text+Vision) 시동 중..."):
    hybrid_retriever, reranker_model = load_search_system()

# 모델 설정
llm_text = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_vision = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0) # 비전 특화 모델

# ==========================================================
# [2] 로직 체인 (Logic Chains)
# ==========================================================

# 1. 텍스트 교정 및 확장 체인
spacing_chain = ChatPromptTemplate.from_template("교정된 한국어 문장만 출력(설명X): {question}").pipe(llm_text).pipe(StrOutputParser())
hyde_chain = ChatPromptTemplate.from_template("건설 전문 검색 키워드 5개 나열(콤마 구분, 설명X): {question}").pipe(llm_text).pipe(StrOutputParser())

# 2. 검색 및 리랭킹 함수
def retrieve_and_rerank(query, top_k=5):
    initial_docs = hybrid_retriever.invoke(query)
    if not initial_docs: return []
    
    pairs = [[query, doc.page_content] for doc in initial_docs]
    
    # 배치 처리로 속도 향상
    scores = []
    batch_size = 16
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        batch_scores = reranker_model.predict(batch)
        scores.extend(batch_scores)
        
    scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_k]]

# 3. Vision 분석 함수 (PDF 심층 분석용)
def analyze_drawing_deep(image_base64, query, retrieved_docs):
    laws_text = "\n".join([f"- {d.page_content}" for d in retrieved_docs])
    
    prompt_text = f"""
    당신은 건축 도면 검토 및 법규 분석 전문가입니다.
    
    [사용자 질문]
    {query}
    
    [관련 법규 데이터베이스]
    {laws_text}
    
    [분석 지시사항]
    1. **법규 매핑:** 사용자 질문과 관련된 법규를 위 데이터베이스에서 찾아내세요.
    2. **도면 인식:** 첨부된 도면 이미지를 정밀 분석하여 벽체, 공간, 치수, 심볼을 식별하세요.
    3. **위반 검토:** 도면의 내용이 법규 기준을 만족하는지 엄격하게 대조하세요.
    4. **결과 출력:** - 위반 여부 (적합/부적합/판단불가)
       - 구체적인 근거 (도면의 어느 부분, 어떤 치수 때문인지)
       - 개선 제안 (필요시)
    """
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
        ]
    )
    return llm_vision.invoke([message]).content

# 4. 일반 텍스트 답변 체인
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

# --- [A] 사이드바: PDF 폴더/파일 선택 창 ---
with st.sidebar:
    st.header("📂 도면 투입구 (PDF)")
    st.info("💡 PDF를 넣으면 자동으로 '심층 도면 분석' 모드로 전환됩니다.")
    
    # 파일 업로더 (여러 파일 가능, 폴더 선택과 유사한 효과)
    uploaded_files = st.file_uploader("검토할 도면 PDF를 선택하세요", type=["pdf"], accept_multiple_files=True)
    
    current_image_base64 = None
    
    if uploaded_files:
        # 편의상 첫 번째 파일만 처리 (추후 리스트로 확장 가능)
        target_file = uploaded_files[0]
        st.write(f"📄 선택된 파일: **{target_file.name}**")
        
        with st.spinner("이미지 변환 중..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(target_file.read())
                tmp_path = tmp_file.name
            
            try:
                # 1페이지만 변환 (속도 최적화)
                # [중요] 여기서 위에서 설정한 POPPLER_PATH 변수를 사용합니다.
                images = convert_from_path(tmp_path, poppler_path=POPPLER_PATH, first_page=1, last_page=1)
                if images:
                    st.image(images[0], caption="검토 대상 도면", use_container_width=True)
                    
                    # Base64 변환
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                        images[0].save(tmp_img.name, "JPEG")
                        with open(tmp_img.name, "rb") as f:
                            current_image_base64 = base64.b64encode(f.read()).decode("utf-8")
                else:
                    st.error("이미지 변환 실패")
            except Exception as e:
                st.error(f"오류: {e}")

# --- [B] 채팅 인터페이스 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- [C] 질문 처리 및 분기 로직 ---
if prompt := st.chat_input("질문 또는 도면 검토 요청을 입력하세요..."):
    # 사용자 질문 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 답변 생성
    with st.chat_message("assistant"):
        
        # [Case 1: PDF가 있는 경우 -> 무조건 심층 분석 (HyDE + Vision)]
        if current_image_base64:
            with st.status("🧠 도면 심층 분석 모드 가동...", expanded=True) as status:
                # 1. HyDE로 검색 키워드 확장
                st.write("🔧 질문 의도 파악 및 법규 검색어 확장 중...")
                hyde_keywords = hyde_chain.invoke({"question": prompt})
                expanded_query = f"{prompt} {hyde_keywords}"
                
                # 2. 관련 법규 검색 (RAG)
                st.write("📚 관련 건설 법규/기준 정밀 검색 중...")
                retrieved_docs = retrieve_and_rerank(expanded_query, top_k=5)
                
                # 3. Vision AI 분석
                st.write("👀 도면 시각적 분석 및 법규 대조 중...")
                vision_response = analyze_drawing_deep(current_image_base64, prompt, retrieved_docs)
                
                status.update(label="✅ 심층 분석 완료!", state="complete", expanded=False)
            
            # 결과 출력
            final_res = f"### 🏗️ 도면 심층 분석 결과\n\n{vision_response}\n\n---\n**[참고한 법규 및 키워드]**\n- 확장 키워드: `{hyde_keywords}`\n" + "\n".join([f"- {d.metadata.get('source')} ({d.metadata.get('article')})" for d in retrieved_docs])
            st.markdown(final_res)
            st.session_state.messages.append({"role": "assistant", "content": final_res})
            
        # [Case 2: PDF가 없는 경우 -> 기존 텍스트 모드 (1차 -> 2차)]
        else:
            with st.status("🔍 1차 검색 진행 중...", expanded=True) as status:
                corrected_query = spacing_chain.invoke({"question": prompt})
                response_1 = rag_chain.invoke(corrected_query)
                status.update(label="✅ 1차 검색 완료", state="complete", expanded=False)
            
            msg_content = f"### 🤖 1차 답변\n{response_1}"
            st.markdown(msg_content)
            st.session_state.messages.append({"role": "assistant", "content": msg_content})
            st.rerun() # 버튼 표시를 위해 리런

# --- [D] 텍스트 모드일 때만 심층 검색 버튼 표시 ---
last_msg = st.session_state.messages[-1] if st.session_state.messages else None
if last_msg and last_msg["role"] == "assistant" and "1차 답변" in last_msg["content"] and "2차" not in last_msg["content"]:
    with st.expander("🤔 답변이 부족한가요? (HyDE 심층 검색)"):
        if st.button("🚀 심층 검색 실행"):
            prev_question = st.session_state.messages[-2]["content"]
            
            with st.status("🧠 전문가 모드(HyDE) 가동 중...", expanded=True) as status:
                hyde_keywords = hyde_chain.invoke({"question": prev_question})
                final_query = f"{prev_question} {hyde_keywords}"
                response_2 = rag_chain.invoke(final_query)
                status.update(label="✅ 완료", state="complete")
            
            final_res = f"### 🤖 2차 상세 답변 (HyDE)\n**확장된 검색어:** `{hyde_keywords}`\n\n{response_2}"
            st.session_state.messages.append({"role": "assistant", "content": final_res})
            st.rerun()