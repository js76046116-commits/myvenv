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
st.set_page_config(page_title="건설 CM AI 통합 솔루션 (Deep RAG)", page_icon="🏗️", layout="wide")

# 1. API 키 가져오기
if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
elif "GOOGLE_API_KEY" in os.environ:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
else:
    st.error("🚨 치명적 오류: Google API Key가 없습니다.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# 2. Poppler 경로 (Windows 환경 대응)
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

# 4. 모델 설정
MODEL_NAME = "models/gemini-2.5-flash" 

# ==========================================================
# [1] 시스템 로딩 (검색 엔진 & 모델)
# ==========================================================
class SimpleHybridRetriever:
    """BM25(키워드) + Chroma(벡터) 결합 검색기"""
    def __init__(self, bm25, chroma1, chroma2, raw_data):
        self.bm25 = bm25
        self.chroma1 = chroma1
        self.chroma2 = chroma2
        self.raw_data = raw_data
        
    def invoke(self, query):
        # 1. 병렬 검색 수행
        docs_bm25 = self.bm25.invoke(query)
        docs_c1 = self.chroma1.invoke(query)
        docs_c2 = self.chroma2.invoke(query)
        
        # 2. Chroma 결과 복원 (인덱스 -> 원본 텍스트)
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

        # 3. 결과 통합 및 중복 제거
        combined = []
        seen_ids = set()
        for d in itertools.chain(docs_bm25, real_docs_chroma):
            key = d.page_content[:30] # 내용 앞부분으로 중복 체크
            if key not in seen_ids:
                combined.append(d)
                seen_ids.add(key)
        return combined[:200] # 1차적으로 넉넉하게 반환

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
    
    # Cross-Encoder (Reranker) 로드
    reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", model_kwargs={"dtype": "auto"})

    return hybrid_retriever, reranker

with st.spinner("🚀 AI 5단계 심층 검색 엔진 시동 중..."):
    try:
        hybrid_retriever, reranker_model = load_search_system()
    except Exception as e:
        st.error(f"시스템 로딩 실패: {e}")
        st.stop()

# LLM 초기화
safety_settings = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
llm_text = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.1, google_api_key=GOOGLE_API_KEY, safety_settings=safety_settings)
llm_vision = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, google_api_key=GOOGLE_API_KEY, safety_settings=safety_settings)

# ==========================================================
# [2] Deep RAG 파이프라인 (5단계 로직 구현)
# ==========================================================

# (1) 쿼리 확장 (Query Expansion)
expansion_prompt = ChatPromptTemplate.from_template("""
당신은 건설/건축 검색 최적화 AI입니다.
사용자 질문을 분석하여 검색 정확도를 높일 수 있는 **'확장 검색어'** 3개를 생성하세요.
건설 표준 시방서, 법규 용어, 동의어를 포함해야 합니다.

[사용자 질문]: {question}

[출력 형식]: 질문 | 키워드1, 키워드2, 키워드3
(설명 없이 위 형식으로만 출력하세요)
""")
expansion_chain = expansion_prompt | llm_text | StrOutputParser()

def get_expanded_queries(original_query):
    """(1단계) 사용자 질문을 확장하여 리스트로 반환"""
    try:
        expanded_str = expansion_chain.invoke({"question": original_query})
        if "|" in expanded_str:
            base, keywords = expanded_str.split("|", 1)
            queries = [base.strip()] + [k.strip() for k in keywords.split(",")]
        else:
            queries = [original_query]
        return queries[:4] # 최대 4개까지만 사용 (속도 조절)
    except:
        return [original_query]

# (2)~(4) 하이브리드 검색 + 재순위화 + Top-K 필터링
def retrieve_and_rerank(query, top_k=50):
    # Step 1: 쿼리 확장
    expanded_queries = get_expanded_queries(query)
    
    # Step 2: 하이브리드 검색 (확장된 쿼리 각각 수행)
    all_docs = []
    seen_contents = set()
    
    for q in expanded_queries:
        docs = hybrid_retriever.invoke(q)
        for doc in docs:
            if doc.page_content not in seen_contents:
                all_docs.append(doc)
                seen_contents.add(doc.page_content)
    
    if not all_docs: return []

    # Step 3: 정밀 재순위화 (Cross-Encoder)
    pairs = [[query, doc.page_content] for doc in all_docs]
    scores = []
    batch_size = 32
    
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        batch_scores = reranker_model.predict(batch)
        scores.extend(batch_scores)
    
    scored_docs = sorted(zip(all_docs, scores), key=lambda x: x[1], reverse=True)
    
    # Step 4: Top-K 필터링 (Top-50)
    final_top_k = [doc for doc, score in scored_docs[:top_k]]
    return final_top_k

# (5) 답변 생성 (유연한 프롬프트)
spacing_chain = ChatPromptTemplate.from_template("교정된 한국어 문장만 출력(설명X): {question}").pipe(llm_text).pipe(StrOutputParser())

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 베테랑 건설 사업 관리자(CM)이자 시공 기술사입니다.
    사용자의 질문에 대해 아래 [Context](검색된 법규/시방서)를 참고하여 답변해야 합니다.

    [답변 규칙]
    1. **우선 순위:** [Context]에 구체적인 절차나 기준이 있다면 반드시 그것을 근거로 답변하세요.
    2. **일반 지식 활용:** 만약 [Context]에 '해결 방안'이나 '구체적 공법'이 부족하다면, 
       **"제공된 법규 데이터에는 구체적 방법이 명시되지 않았으나, 일반적인 시공 기준에 따르면..."** 이라고 언급한 뒤, 당신이 알고 있는 **표준 시방서 및 공학적 지식**을 동원해 해결책을 제시하세요.
    3. 절대 "모른다"고 끝내지 말고, 실무적인 조언을 제공하세요.
    4. 출처가 있다면 [출처: ...] 형태로 명시하세요.

    [Context]
    {context}
    """),
    ("human", "질문: {question}")
])

def format_docs(docs):
    return "\n\n".join([f"<출처: {d.metadata.get('source')} / {d.metadata.get('article')}>\n{d.page_content}" for d in docs])

# 최종 RAG 체인 (Top-50 적용)
rag_chain = (
    {"context": RunnableLambda(lambda x: retrieve_and_rerank(x, top_k=50)) | format_docs, "question": RunnablePassthrough()}
    | answer_prompt | llm_text | StrOutputParser()
)

# ==========================================================
# [3] Vision AI (도면 분석용)
# ==========================================================
def analyze_page_detail(image_base64, query, retrieved_docs):
    laws_text = "\n".join([f"- {d.page_content[:200]}..." for d in retrieved_docs])
    if not laws_text.strip():
        laws_text = "(일반 시공 지식 기반)"

    prompt_text = f"""
    당신은 건설 시공 품질/안전 전문가입니다.
    [검토 요청] {query}
    [참고 기준] {laws_text}
    
    도면 이미지를 정밀 분석하여 품질 문제(균열, 누수 등)와 안전 위험을 찾아내세요.
    """
    
    message = HumanMessage(content=[
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
    ])
    try:
        response = llm_vision.invoke([message])
        return response.content
    except:
        return "분석 불가"

def generate_final_report(file_name, page_results):
    raw_data = ""
    for item in page_results:
        raw_data += f"\n[Page {item['page']}]: {item['content']}\n"
    
    prompt = f"""
    당신은 건설사업관리단장(CM단장)입니다.
    '{file_name}' 도면의 페이지별 분석 내용을 종합하여 **최종 시공 품질/안전 검토 보고서**를 작성하세요.
    중복된 내용은 통합하고, 핵심 이슈 위주로 요약하세요.
    
    [분석 데이터]
    {raw_data}
    """
    return llm_text.invoke(prompt).content

# ==========================================================
# [4] 웹 UI (Streamlit)
# ==========================================================
st.title("🏗️ 건설 CM 전문 AI (Deep RAG + Vision)")

# 세션 상태 관리
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "current_image_base64" not in st.session_state:
    st.session_state.current_image_base64 = None

# --- [사이드바] 파일 업로드 및 모드 설정 ---
with st.sidebar:
    st.header("📂 도면 투입구")
    uploaded_files = st.file_uploader("PDF 도면 업로드", type=["pdf"], accept_multiple_files=True)
    
    st.markdown("---")
    
    # [핵심] 파일 업로드 여부에 따라 UI 즉시 변경
    if uploaded_files:
        st.subheader("🤖 질문 모드")
        search_mode = st.radio(
            "모드 선택",
            ["📂 도면 기반 질문", "⚖️ 일반 법규 검색"],
            index=0,
            help="📂 도면 기반: 현재 보는 도면 내용 참고\n⚖️ 일반 법규: 도면 무시, 법규 DB 심층 검색"
        )
    else:
        # 파일이 없을 때 기본값
        search_mode = "⚖️ 일반 법규 검색"
        st.info("💡 도면이 없습니다.\n**'심층 법규 검색 모드'**가 활성화되었습니다.\n(Query Expansion + Rerank)")

# --- [메인] 도면 처리 로직 ---
if uploaded_files:
    for target_file in uploaded_files:
        if target_file.name not in st.session_state.processed_files:
            with st.status(f"📄 '{target_file.name}' 분석 중...", expanded=True) as status:
                # 1. PDF 변환
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(target_file.read())
                    tmp_path = tmp_file.name
                
                try:
                    all_pages = convert_from_path(tmp_path, poppler_path=POPPLER_PATH)
                except Exception as e:
                    st.error(f"변환 오류: {e}")
                    continue

                # 2. Vision 분석 루프
                page_results = []
                progress = st.progress(0)
                for i, page_img in enumerate(all_pages):
                    progress.progress((i+1)/len(all_pages), text=f"🔍 Page {i+1} 정밀 진단 중...")
                    
                    # 이미지 base64 변환
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                        page_img.save(tmp_img.name, "JPEG")
                        with open(tmp_img.name, "rb") as f:
                            img_base64 = base64.b64encode(f.read()).decode("utf-8")
                    
                    st.session_state.current_image_base64 = img_base64 # 최신 이미지 저장
                    
                    # 분석 실행
                    res = analyze_page_detail(img_base64, "위험 요소 식별", [])
                    page_results.append({"page": i+1, "content": res})
                
                # 3. 종합 보고서
                status.write("📝 종합 보고서 작성 중...")
                report = generate_final_report(target_file.name, page_results)
                
                st.session_state.processed_files.add(target_file.name)
                st.session_state.messages.append({"role": "assistant", "content": report})
                progress.empty()
                status.update(label="분석 완료", state="complete")

# --- [채팅창] ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # [분기 로직] 도면 모드 vs 법규 모드
        
        # Case 1: 도면 모드이고 + 이미지가 있을 때 -> Vision AI
        if search_mode == "📂 도면 기반 질문" and st.session_state.current_image_base64:
            with st.status("🔍 도면 재검토 및 Vision 분석 중...", expanded=True):
                msg = HumanMessage(content=[
                    {"type": "text", "text": f"질문: {prompt}\n(이전 분석 맥락 참고하여 답변)"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{st.session_state.current_image_base64}"}}
                ])
                response = llm_vision.invoke([msg]).content
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Case 2: 일반 법규 모드이거나 OR 도면이 없을 때 -> Deep RAG
        else:
            with st.status("🧠 5단계 심층 검색 중 (확장-검색-재순위화)...", expanded=True):
                # 쿼리 교정 및 확장 -> 검색 -> 답변
                corrected_query = spacing_chain.invoke({"question": prompt})
                response = rag_chain.invoke(corrected_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})