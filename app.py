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
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# ==========================================================
# [0] 기본 설정 및 상수 정의
# ==========================================================
st.set_page_config(page_title="건설 CM AI 통합 솔루션 (시공/품질)", page_icon="🏗️", layout="wide")

# 1. API 키 가져오기 (Secrets 우선 -> 환경변수)
if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
elif "GOOGLE_API_KEY" in os.environ:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
else:
    st.error("🚨 치명적 오류: Google API Key가 없습니다. Streamlit Secrets 설정을 확인하세요.")
    st.stop()

# 환경 변수 동기화
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# 2. Poppler 경로 설정 (Windows 로컬 / 리눅스 서버 자동 구분)
system_name = platform.system()
if system_name == "Windows":
    # 사용자님 로컬 경로 (수정 금지)
    POPPLER_PATH = r"C:\Users\owner\myvenv\Release-25.12.0-0\poppler-25.12.0\Library\bin"
else:
    # Streamlit Cloud 서버용 (자동 설치됨)
    POPPLER_PATH = None 

# 3. 데이터 경로
DB_PATH_1 = "./chroma_db_part1"
DB_PATH_2 = "./chroma_db_part2"
JSON_DATA_PATH = "./legal_data_total_vlm.json"
RAW_DATA = []

# 4. 모델 설정 (사용자 로그 기반 생존 모델)
MODEL_NAME = "models/gemini-2.5-flash" 

# ==========================================================
# [1] 시스템 로딩 (검색엔진 + AI모델)
# ==========================================================
class SimpleHybridRetriever:
    def __init__(self, bm25, chroma1, chroma2, raw_data):
        self.bm25 = bm25
        self.chroma1 = chroma1
        self.chroma2 = chroma2
        self.raw_data = raw_data
        
    def invoke(self, query):
        # 3가지 검색기 동시 가동
        docs_bm25 = self.bm25.invoke(query)
        docs_c1 = self.chroma1.invoke(query)
        docs_c2 = self.chroma2.invoke(query)
        
        # Chroma 결과(ID)를 실제 텍스트로 변환
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

        # 중복 제거 및 결합
        combined = []
        seen_ids = set()
        for d in itertools.chain(docs_bm25, real_docs_chroma):
            key = d.page_content[:30] # 내용 앞부분으로 중복 체크
            if key not in seen_ids:
                combined.append(d)
                seen_ids.add(key)
        return combined[:200]

@st.cache_resource
def load_search_system():
    global RAW_DATA
    if not os.path.exists(JSON_DATA_PATH):
        st.error("❌ JSON 데이터 파일이 없습니다. (legal_data_total_vlm.json)")
        st.stop()
        
    with open(JSON_DATA_PATH, 'r', encoding='utf-8') as f:
        RAW_DATA = json.load(f)

    # 임베딩 모델 (API 키 직접 주입)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=GOOGLE_API_KEY
    )
    
    if not os.path.exists(DB_PATH_1) or not os.path.exists(DB_PATH_2):
        st.error("❌ DB 폴더가 없습니다. (chroma_db_part1, part2)")
        st.stop()

    # Chroma DB 로드
    store1 = Chroma(persist_directory=DB_PATH_1, embedding_function=embeddings, collection_name="construction_laws")
    retriever1 = store1.as_retriever(search_kwargs={"k": 100})
    store2 = Chroma(persist_directory=DB_PATH_2, embedding_function=embeddings, collection_name="construction_laws")
    retriever2 = store2.as_retriever(search_kwargs={"k": 100})

    # BM25(키워드 검색) 로드
    docs = []
    for item in RAW_DATA:
        content = item.get('content', '').strip()
        source = item.get('source', '').strip()
        if not content: continue
        doc = Document(page_content=f"[{source}] {content}", metadata={"source": source, "article": item.get('article', '')})
        docs.append(doc)
    
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 150

    # 하이브리드 검색기 생성
    hybrid_retriever = SimpleHybridRetriever(bm25_retriever, retriever1, retriever2, RAW_DATA)
    
    # Reranker (정확도 향상)
    reranker = CrossEncoder(
        "cross-encoder/ms-marco-TinyBERT-L-2-v2", 
        model_kwargs={"dtype": "auto"}
    )

    return hybrid_retriever, reranker

# 시스템 초기화
with st.spinner("🚀 AI 건설 통합 엔진(Text+Vision) 시동 중..."):
    try:
        hybrid_retriever, reranker_model = load_search_system()
    except Exception as e:
        st.error(f"시스템 로딩 실패: {e}")
        st.stop()

# LLM 초기화 (API 키 직접 주입 & 2.5 모델 사용)
llm_text = ChatGoogleGenerativeAI(
    model=MODEL_NAME, 
    temperature=0, 
    google_api_key=GOOGLE_API_KEY
)
llm_vision = ChatGoogleGenerativeAI(
    model=MODEL_NAME, 
    temperature=0, 
    google_api_key=GOOGLE_API_KEY
)

# ==========================================================
# [2] 로직 체인 (RAG & Vision)
# ==========================================================
spacing_chain = ChatPromptTemplate.from_template("교정된 한국어 문장만 출력(설명X): {question}").pipe(llm_text).pipe(StrOutputParser())

def retrieve_and_rerank(query, top_k=5):
    # 1차 검색
    initial_docs = hybrid_retriever.invoke(query)
    if not initial_docs: return []
    
    # 2차 재순위(Reranking)
    pairs = [[query, doc.page_content] for doc in initial_docs]
    scores = []
    batch_size = 16
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        batch_scores = reranker_model.predict(batch)
        scores.extend(batch_scores)
    
    scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_k]]

# [핵심] 도면 분석 함수: 시공/품질/안전 관리 관점
def analyze_drawing_deep(image_base64, query, retrieved_docs):
    # DB에서 찾아온 시방서/지침 내용 합치기
    laws_text = "\n".join([f"- {d.page_content}" for d in retrieved_docs])
    
    prompt_text = f"""
    당신은 건설 현장의 **시공 품질 및 안전 관리 전문가(Construction CM Expert)**입니다.
    
    [검토 요청 사항]
    {query}
    
    [참고 기준: 건설관리 지침 및 시방서 DB]
    {laws_text}
    
    [지시사항]
    1. **시공 관점 분석:** 도면의 부재(벽체, 슬래브, 창호, 마감 등)를 보고, 실제 시공 시 발생할 수 있는 문제점을 찾으세요.
    2. **DB 기반 대조:** 위 [참고 기준]에 있는 시방서나 지침 내용과 대조하여, 준수해야 할 품질 기준이나 안전 수칙을 제시하세요.
    3. **중점 검토 항목:**
       - **품질 관리:** 균열(Crack), 누수, 단열 결손, 재료 분리, 양생 관리 등.
       - **안전 관리:** 추락 위험, 장비 전도, 가설재 설치 등 현장 안전 요소.
       - **하자 예방:** 결로, 층간 소음, 마감 불량 등 주요 하자 포인트.
    4. **결과 보고 양식:**
       - 🛠️ **주요 시공 관리 포인트:** (도면의 특정 부위에 대해 시방서 기준 제시)
       - ⚠️ **예상되는 품질/안전 리스크:** (LH 지침 등에 근거한 위험 요소 경고)
       - 💡 **개선 제안:** (참고 문헌에 기반한 기술적 조언)
    """
    
    message = HumanMessage(content=[
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
    ])
    
    return llm_vision.invoke([message]).content

# RAG용 프롬프트
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "건설 기준 엔지니어입니다. [Context]를 보고 답변하세요. 출처(Source/Article) 표기 필수.\n[Context]\n{context}"),
    ("human", "질문: {question}")
])

def format_docs(docs):
    return "\n\n".join([f"<출처: {d.metadata.get('source')} / {d.metadata.get('article')}>\n{d.page_content}" for d in docs])

rag_chain = (
    {"context": RunnableLambda(lambda x: retrieve_and_rerank(x, top_k=10)) | format_docs, "question": RunnablePassthrough()}
    | answer_prompt | llm_text | StrOutputParser()
)

# ==========================================================
# [3] 웹 UI 구성
# ==========================================================
st.title("🏗️ 건설 CM 전문 AI (시공 품질/안전)")

# --- [A] 사이드바: 파일 업로드 ---
if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "current_image_base64" not in st.session_state: 
    st.session_state.current_image_base64 = None

with st.sidebar:
    st.header("📂 도면 투입구")
    st.info("💡 PDF를 넣으면 **전체 페이지**를 시공/품질 관점으로 분석합니다.")
    uploaded_files = st.file_uploader("검토할 도면 PDF를 선택하세요", type=["pdf"], accept_multiple_files=True)

# --- [B] 자동 분석 (다중 페이지 전체 분석) ---
if uploaded_files:
    target_file = uploaded_files[0]
    
    # 새 파일이 들어오면 분석 시작
    if st.session_state.last_processed_file != target_file.name:
        st.session_state.analysis_result = "" # 초기화
        st.session_state.last_processed_file = target_file.name
        
        # 1. PDF -> 이미지 변환 (전체 페이지)
        with st.status("📄 PDF 변환 및 분석 준비 중...", expanded=True) as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(target_file.read())
                tmp_path = tmp_file.name
            try:
                # [핵심] 옵션 없이 호출하면 전체 페이지 변환됨
                all_pages = convert_from_path(tmp_path, poppler_path=POPPLER_PATH)
                status.write(f"✅ 총 {len(all_pages)}장의 도면을 확인했습니다. 전체 분석을 시작합니다.")
            except Exception as e:
                st.error(f"이미지 변환 오류: {e}")
                st.stop()
        
        # 2. 페이지별 순차 반복 분석 (Loop)
        full_report = f"### 🏗️ 도면 시공 품질/안전 분석 결과 (총 {len(all_pages)}장)\n**분석 대상:** {target_file.name}\n\n"
        progress_bar = st.progress(0)
        
        # 결과 표시용 컨테이너
        result_container = st.container()

        for i, page_img in enumerate(all_pages):
            page_num = i + 1
            progress_msg = f"🚀 {page_num}/{len(all_pages)} 페이지 분석 중... (시공 기준 대조 중)"
            progress_bar.progress((i + 1) / len(all_pages), text=progress_msg)
            
            # 이미지 Base64 변환
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                page_img.save(tmp_img.name, "JPEG")
                with open(tmp_img.name, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode("utf-8")
            
            # 마지막 페이지 이미지는 질문용으로 세션에 저장
            st.session_state.current_image_base64 = img_base64
            
            # [수정] 자동 질문: 품질/안전/하자 관점
            auto_query = """
            이 도면을 실제 시공할 때 유의해야 할 '품질 관리', '안전 사고 예방', '하자 방지' 사항을 검토해줘.
            특히 콘크리트 타설, 마감 공사, 단열/방수 시공 시 LH 전문 시방서 및 건설관리 지침에 따라 주의할 점을 알려줘.
            """
            
            # 검색 및 분석 수행
            retrieved_docs = retrieve_and_rerank(auto_query, top_k=5)
            vision_result = analyze_drawing_deep(img_base64, auto_query, retrieved_docs)
            
            # 결과 누적
            page_content = f"""
#### 📄 Page {page_num} 분석
{vision_result}
"""
            full_report += page_content
            
            # 중간 결과 화면 표시 (Expander로 이미지 보여주기)
            with result_container:
                with st.expander(f"🔎 제 {page_num}페이지 도면 & 요약 보기", expanded=False):
                    st.image(page_img, caption=f"Page {page_num}", width="stretch")
                    st.markdown(vision_result)

        progress_bar.empty()
        st.session_state.analysis_result = full_report
        st.success("✅ 모든 페이지 분석이 완료되었습니다!")

# --- [C] 채팅 인터페이스 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 분석 결과가 있으면 채팅창 맨 위에 표시
if st.session_state.analysis_result:
    if not st.session_state.messages or st.session_state.messages[0]["content"] != st.session_state.analysis_result:
        st.session_state.messages = [{"role": "assistant", "content": st.session_state.analysis_result}]

# 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- [D] 사용자 질문 입력 ---
if prompt := st.chat_input("추가 질문이 있으신가요? (예: 창호 주위 결로 방지 대책은?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 이미지가 있는 경우
        if st.session_state.current_image_base64:
            with st.status("🔍 도면과 시방서를 복합 분석 중...", expanded=True) as status:
                st.write("📚 관련 시방서/지침 검색 중...")
                retrieved_docs = retrieve_and_rerank(prompt, top_k=5)
                
                st.write("👀 마지막 도면 페이지 재확인 중...")
                # 질문에 대해 다시 Vision AI 호출
                vision_res = analyze_drawing_deep(st.session_state.current_image_base64, prompt, retrieved_docs)
                status.update(label="✅ 답변 완료", state="complete")
            
            final_res = f"{vision_res}\n\n[참고 자료]: " + ", ".join([d.metadata.get('article', '출처미상') for d in retrieved_docs])
            st.markdown(final_res)
            st.session_state.messages.append({"role": "assistant", "content": final_res})
        
        # 이미지가 없는 경우
        else:
            corrected = spacing_chain.invoke({"question": prompt})
            response = rag_chain.invoke(corrected)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})