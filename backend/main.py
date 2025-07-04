from fastapi import FastAPI

# from utils import load_vectorstore, get_answer
from embeded import load_vectorstore, get_answer
import os
from dotenv import load_dotenv
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from langchain.memory import ConversationBufferMemory
import deepl

app = FastAPI()

origins = [
    "http://localhost:3000",  # Next.js 개발서버 주소
    "https://navisation.vercel.app",  # 배포 후 프론트 도메인 추가
    "https://navisation-phi.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 허용할 출처 리스트
    allow_credentials=True,  # 쿠키 같은 인증정보 허용 여부
    allow_methods=["*"],  # 허용할 HTTP 메서드
    allow_headers=["*"],  # 허용할 헤더
)

# .env 파일 로드
load_dotenv()

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
vectorstore = load_vectorstore(UPSTAGE_API_KEY)

DEEPL_AUTH_KEY = os.getenv("DEEPL_AUTH_KEY")
translator = deepl.Translator(DEEPL_AUTH_KEY)

session_memories = {}


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/chat-request")
async def chat_request(
    req: str = Query(..., description="질문 내용"),
    lang: str = Query(..., description="언어 코드"),
    session_id: str = Query(..., description="세션 ID"),
):
    print(f"[session: {session_id}] req: {req}, lang: {lang}")

    print(session_memories)
    # 세션별 memory 객체 재사용 또는 생성
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(return_messages=False)

    memory = session_memories[session_id]
    # req를 한국어로 번역 (DeepL 언어코드 기준으로)
    lang_map = {
        "영어": "EN",
        "중국어": "ZH",
        "일본어": "JA",
        "한국어": "KO",
        "베트남어": "VI",
    }
    """
    question으로 context를 먼저 찾은 후 LLM으로 답을 생성. 
    중국어나 베트남어는 question 그대로 전달하면 제대로 retriever 못함
    question을 한국어로 번역한 후 전달 
    답변은 그 나라 언어 그대로 나옴 
    
    """

    if lang in lang_map and lang_map[lang] != "KO":
        translated_req = translator.translate_text(
            req, source_lang=lang_map[lang], target_lang="KO"
        ).text
        print(f"🔁 Translated '{req}' ({lang}) → '{translated_req}' (한국어)")
    else:
        translated_req = req  # 이미 한국어이면 그대로 사용

    ans = get_answer(vectorstore, translated_req, lang, memory)
    return {"question": req, "lang": lang, "session_id": session_id, "answer": ans}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app", host="0.0.0.0", port=8000)
