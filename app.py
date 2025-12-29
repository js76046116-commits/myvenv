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
st.set_page_config(page_title="건설 CM AI 통합 솔루션 (종합보고서)", page_icon="🏗️", layout="wide")

# 1. API 키 가져오기
if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
elif "GOOGLE_API_KEY" in os.environ:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
else:
    st.error("🚨 치명적 오류: Google API Key가 없습니다.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# 2. Poppler 경로
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

# 4. 모델 설정
MODEL_NAME = "models/gemini-2.5-flash" 

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

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    
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
    reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", model_kwargs={"dtype": "auto"})

    return hybrid_retriever, reranker

with st.spinner("🚀 AI 엔진 시동 중..."):
    try:
        hybrid_retriever, reranker_model = load_search_system()
    except Exception as e:
        st.error(f"시스템 로딩 실패: {e}")
        st.stop()

# 안전 설정 (건설 현장 사진 차단 방지)
safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

llm_text = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, google_api_key=GOOGLE_API_KEY, safety_settings=safety_settings)
llm_vision = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, google_api_key=GOOGLE_API_KEY, safety_settings=safety_settings)

# ==========================================================
# [2] 분석 로직 (페이지 분석 -> 종합 리포트)
# ==========================================================
spacing_chain = ChatPromptTemplate.from_template("교정된 한국어 문장만 출력(설명X): {question}").pipe(llm_text).pipe(StrOutputParser())

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

# [A] 페이지별 정밀 진단 (Vision)
def analyze_page_detail(image_base64, query, retrieved_docs):
    laws_text = "\n".join([f"- {d.page_content}" for d in retrieved_docs])
    if not laws_text.strip():
        laws_text = "(검색된 관련 시방서가 없습니다. 일반적인 시공 지식을 바탕으로 분석합니다.)"

    prompt_text = f"""
    당신은 건설 현장의 **시공 품질 및 안전 관리 전문가(Construction CM Expert)**입니다.
    
    [검토 요청] {query}
    [참고 기준] {laws_text}
    
    [지시사항]
    1. 도면을 정밀하게 보고 시공 시 발생 가능한 **품질 문제(균열, 누수, 결로)**와 **안전 위험(추락, 전도)**을 찾아내세요.
    2. 반드시 위 [참고 기준]의 시방서 내용을 근거로 지적하세요.
    3. **메모 형식**으로 핵심만 간단히 작성하세요. (나중에 종합할 것입니다.)
    """
    
    message = HumanMessage(content=[
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
    ])
    
    try:
        response = llm_vision.invoke([message])
        return response.content if response.content else "특이사항 없음."
    except Exception as e:
        return f"분석 오류: {str(e)}"

# [B] 종합 보고서 작성 (Text summarization)
def generate_final_report(file_name, page_results):
    # 페이지별 결과를 하나의 텍스트로 합침
    raw_data = ""
    for item in page_results:
        raw_data += f"\n[Page {item['page']} 진단내용]:\n{item['content']}\n"
    
    prompt = f"""
    당신은 건설사업관리단장(CM단장)입니다.
    각 페이지별 담당자가 보고한 내용을 바탕으로 **'{file_name}'에 대한 최종 시공 품질/안전 검토 보고서**를 작성하세요.

    [담당자 보고 내용 합본]
    {raw_data}

    [보고서 작성 규칙]
    1. **중복 통합:** 여러 페이지에서 반복되는 지적 사항은 하나로 합쳐서 강력하게 권고하세요.
    2. **구조화된 목차:**
       # 🏗️ [종합] 시공 품질 및 안전 검토 보고서
       ## 1. 총평 (Executive Summary)
       ## 2. 주요 시공 관리 포인트 (LH 시방서 기준)
          - 품질 관리 (균열, 방수, 단열 등)
          - 안전 관리 (추락, 낙하, 장비 등)
       ## 3. 페이지별 특이사항 (Issues by Page)
          - (문제가 발견된 페이지만 요약하여 기재)
    3. **톤앤매너:** 전문적이고 단호한 어조를 사용하세요.
    """
    return llm_text.invoke(prompt).content

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "건설 기준 엔지니어입니다. [Context]를 보고 답변하세요.\n[Context]\n{context}"),
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
st.title("🏗️ 건설 CM 전문 AI (시공 품질/안전 종합분석)")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set() # 처리한 파일 기억장소
if "current_image_base64" not in st.session_state:
    st.session_state.current_image_base64 = None

# --- [A] 사이드바 ---
with st.sidebar:
    st.header("📂 도면 투입구")
    st.info("💡 PDF를 업로드하면 **전체 페이지를 분석하여 하나의 종합 보고서**를 만듭니다.")
    uploaded_files = st.file_uploader("검토할 도면 PDF", type=["pdf"], accept_multiple_files=True)

# --- [B] 자동 분석 로직 (순차 처리 + 종합) ---
if uploaded_files:
    for target_file in uploaded_files:
        # 이미 처리한 파일은 건너뜀 (중복 분석 방지)
        if target_file.name not in st.session_state.processed_files:
            
            # 1. 파일 변환 알림
            with st.status(f"📄 '{target_file.name}' 도면 스캔 중...", expanded=True) as status:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(target_file.read())
                    tmp_path = tmp_file.name
                
                try:
                    all_pages = convert_from_path(tmp_path, poppler_path=POPPLER_PATH)
                    status.write(f"✅ 총 {len(all_pages)}페이지 변환 완료. 정밀 진단 시작...")
                except Exception as e:
                    st.error(f"변환 실패: {e}")
                    continue

                # 2. 페이지별 루프 (Vision Analysis)
                page_results = []
                progress_bar = st.progress(0)
                
                for i, page_img in enumerate(all_pages):
                    page_num = i + 1
                    progress_text = f"🔍 Page {page_num}/{len(all_pages)} 정밀 분석 중... (시방서 대조)"
                    progress_bar.progress((i + 1) / len(all_pages), text=progress_text)
                    
                    # 이미지 인코딩
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                        page_img.save(tmp_img.name, "JPEG")
                        with open(tmp_img.name, "rb") as f:
                            img_base64 = base64.b64encode(f.read()).decode("utf-8")
                    
                    # 가장 최근 본 이미지 저장 (추가 질문용)
                    st.session_state.current_image_base64 = img_base64
                    
                    # 개별 페이지 분석
                    query = "이 도면의 시공 품질 및 안전 위험 요소를 찾아줘."
                    retrieved_docs = retrieve_and_rerank(query, top_k=3)
                    result = analyze_page_detail(img_base64, query, retrieved_docs)
                    
                    # 결과 메모
                    page_results.append({"page": page_num, "content": result})

                # 3. 종합 보고서 작성 (Consolidation)
                status.write("📝 페이지별 진단 완료. 종합 보고서 작성 중...")
                final_report = generate_final_report(target_file.name, page_results)
                
                # 4. 결과 저장 및 출력
                st.session_state.processed_files.add(target_file.name)
                st.session_state.messages.append({"role": "assistant", "content": final_report})
                
                progress_bar.empty()
                status.update(label=f"✅ '{target_file.name}' 분석 완료!", state="complete")

# --- [C] 채팅창 표시 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- [D] 사용자 질문 ---
if prompt := st.chat_input("보고서 내용에 대해 궁금한 점이 있나요?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 이미지가 있으면 (방금 분석한 도면 기준)
        if st.session_state.current_image_base64:
            with st.status("🔍 도면 재검토 및 답변 중...", expanded=True):
                retrieved_docs = retrieve_and_rerank(prompt, top_k=5)
                # Vision AI에게 다시 물어봄
                prompt_text = f"사용자 질문: {prompt}\n(이전 분석 맥락을 참고하여 답변하세요.)"
                message = HumanMessage(content=[
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{st.session_state.current_image_base64}"}},
                ])
                response = llm_vision.invoke([message]).content
            
            # 근거 자료 표시
            refs = "\n\n[관련 근거]: " + ", ".join([d.metadata.get('article', '출처미상') for d in retrieved_docs])
            final_res = response + refs
            st.markdown(final_res)
            st.session_state.messages.append({"role": "assistant", "content": final_res})
        
        # 이미지가 없으면 (일반 텍스트 질문)
        else:
            corrected = spacing_chain.invoke({"question": prompt})
            response = rag_chain.invoke(corrected)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})