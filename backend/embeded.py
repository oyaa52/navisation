# from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re
import json
import time
import tiktoken
from tqdm import tqdm
from langchain_upstage import ChatUpstage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings
from ragas import evaluate
from langchain.memory import ConversationBufferMemory
from rapidfuzz import fuzz
from datasets import Dataset
from langchain.embeddings import OpenAIEmbeddings
from ragas.metrics import context_precision, context_recall, faithfulness


UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")


# ---------------------------------------
def remove_markdown(text: str) -> str:
    # 굵은 텍스트 제거
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    # 기울임 텍스트 제거
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    # 헤더 제거
    text = re.sub(r"#+ ", "", text)
    # 리스트 기호 제거
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    # 인라인 코드 제거
    text = re.sub(r"`(.*?)`", r"\1", text)
    # 코드블럭 제거
    text = re.sub(r"```[\s\S]*?```", "", text)
    return text.strip()


def load_vectorstore(UPSTAGE_API_KEY):
    embedding_model = UpstageEmbeddings(
        api_key=UPSTAGE_API_KEY, model="solar-embedding-1-large-passage"
    )

    vectorstore = FAISS.load_local(
        "faiss_vector_store", embedding_model, allow_dangerous_deserialization=True
    )
    return vectorstore


def get_answer(vectorstore, question, lang, memory):
    llm = ChatUpstage(
        model="solar-pro2-preview", temperature=0.3, reasoning_effort="high"
    )
    language = lang

    # 2. 프롬프트 정의
    prompt_template = PromptTemplate.from_template(
        # """
        # 당신은 대한민국 출입국 관련 매뉴얼을 바탕으로 질문에 답변하는 전문가이자 통역가입니다.
        # ---
        # 다음은 참고할 정보입니다.
        # 컨텍스트:
        # {context}
        # 이전 대화 기록:
        # {history}
        # 질문:
        # {question}
        # ---
        # #Strong Rules:
        # #AVOID ALL markdown elements: **bold**, *italic*, #headers, -lists, ` code blocks, etc.
        # 질문과 컨텍스트를 잘 읽고, 아래 지침에 따라 사용자가 사용한 언어인 {language}으로 가장 적절한 답변을 해 주세요. 적절한 주제와 내용, 이모지로 구성하여 읽기 쉽게 작성해 주세요.:
        # - 반드시 사용자가 사용한 언어인 {language}를 사용하여 답변하세요!!!
        # - 만약 이전 대화 기록이 있다면 반드시 **이전에** 사용자가 선택한 **세부 비자 유형**에 맞는 답변을 해 주세요.
        # - 각 비자에는 세부 유형(sub-type)이 있을 수 있습니다 (예: D-8-1, D-8-4).
        #   → 각 세부 비자별 요건 및 제출 서류가 다르므로, 반드시 정확한 세부 유형을 구분하여 사용자가 사용한 언어인 {language}으로 답변해 주세요.
        # - 각 비자에 대한 제출 서류, 대상자, 자격요건 등은 모든 조건을 만족해야 하는지, 아니면 일부만 만족해도 되는지 명확히 구분하여 답변해 주세요.
        # - 점수제에도 여러 종류가 있습니다. 비자에 따라 해당하는 점수제가 달라지니 유의해서 답변해 주세요.
        # - 컨텍스트에 답변에 필요한 정보가 부족하다면, 절대로 추측하지 말고 주어진 정보만 바탕으로 답변해 주세요..
        # - 이전 대화에 포함된 질문이라면 간결하게 다시 설명하고, 새 정보가 있다면 그 위주로 답변하세요.
        # - 답변에는 질문 내용을 반복하지 마시고, 바로 답변부터 시작하세요.
        # ---
        # 답변:
        # """
        """
        You are an expert and interpreter who answers questions based on the Korean immigration manual. 
                                                   
        ---
        Here is the information to refer to. 

        Context:
        {context}
                                                   
        History of previous conversations: 
        {history}

        Question:
        {question}

        ---
        #Strong Rules:
        #AVOID ALL markdown elements: **bold**, *italic*, #headers, -lists, ` code blocks, etc. 
                                                   
        Read the questions and context carefully and follow the instructions below to give the most appropriate answer in the language you used: {language}. Please make it easy to read by organizing appropriate topics, contents, and emojis:
        - Be sure to answer using {language}, the language you used!!!
        - If you have a history of previous conversations, please be sure to give an answer that matches the **Detailed Visa type ** you have chosen before ****.
        - Each visa may have a sub-type (e.g., D-8-1, D-8-4).  
          → Since each detailed visa requirement and submission document are different, be sure to distinguish the exact detailed type and answer in {language}, the language you used.
        - Please answer the submission documents, subjects, qualification requirements, etc. for each visa by clearly separating whether they should meet all conditions or only a part of them.
        - There are many types of scoring systems. Please answer carefully because the corresponding scoring system varies depending on the visa.
        - If the context lacks the information you need to answer, please never guess and answer based only on the information given..
        - If the questions are included in the previous conversation, please briefly explain them again, and if you have any new information, answer them mainly.
        - Don't repeat the question in your answer, but start with the answer right away.
                                                   
        --- 

        Answer:
        """
    )

    chain = prompt_template | llm | StrOutputParser()

    # 3. Retriever 정의
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.9},
    )

    history_str = memory.load_memory_variables({}).get("history", "")

    #  히스토리를 검색 단계에서 반영
    # search_query = (history_str + "\n" if history_str else "") + question
    docs = retriever.invoke(question)

    # context는 docs에서 추출
    context = "\n\n".join([doc.page_content for doc in docs])
    # context_str = (history_str + "\n\n" if history_str else "") + context

    # 6. LLM 호출 (필수 프롬프트 변수 포함)
    answer = chain.invoke(
        {
            "context": context,
            "question": question,
            "language": language,
            "history": history_str,
        }
    )

    answer = remove_markdown(answer)

    memory.save_context({"input": question}, {"output": answer})

    return answer


# pdf_paths = ["stay.pdf", "visa.pdf"]  # 여기에 두 PDF 경로 입력

# all_text_blocks = []
# all_tables = []

# for pdf_path in pdf_paths:
#     with pdfplumber.open(pdf_path) as pdf:
#         for i, page in enumerate(pdf.pages):
#             text = page.extract_text()
#             tables = page.extract_tables()

#             # 텍스트 문단 구조화
#             if text:
#                 paragraphs = text.split('\n\n')  # 줄바꿈 기준 나누기
#                 for para in paragraphs:
#                     all_text_blocks.append({
#                         "file": os.path.basename(pdf_path),
#                         "page": i + 1,
#                         "type": "text",
#                         "content": para.strip()
#                     })

#             # 표 구조화
#             for table in tables:
#                 if table and len(table) > 1:
#                     headers = table[0]
#                     rows = table[1:]
#                     all_tables.append({
#                         "file": os.path.basename(pdf_path),
#                         "page": i + 1,
#                         "type": "table",
#                         "headers": headers,
#                         "rows": rows
#                     })


# # 확인
# # print(all_text_blocks[:1000])  # 처음 일부 텍스트 출력
# # print(all_tables[1])


# """### 문맥 별로 청크하기
# - 평균 토큰 수: 2000개
# - 최대 토큰 수 : 약 4800개
# - 최소 토큰 수: 약 30개
# """

# # 문맥 별로 청크하기
# def table_to_markdown(headers, rows):
#     # None → 빈 문자열 처리
#     headers = [str(h) if h is not None else "" for h in headers]
#     header_line = "| " + " | ".join(headers) + " |"
#     separator = "| " + " | ".join(["---"] * len(headers)) + " |"

#     row_lines = []
#     for row in rows:
#         cleaned_row = [str(cell) if cell is not None else "" for cell in row]
#         row_lines.append("| " + " | ".join(cleaned_row) + " |")

#     return "\n".join([header_line, separator] + row_lines)


# def build_semantic_chunks_by_proximity(all_chunks, window=1):
#     semantic_chunks = []

#     # 정렬
#     all_chunks.sort(key=lambda x: (x['file'], x['page']))

#     for i, chunk in enumerate(all_chunks):
#         if chunk["type"] == "table":
#             # 주변 텍스트 찾기
#             related_texts = []
#             for offset in range(-window, window + 1):
#                 j = i + offset
#                 if 0 <= j < len(all_chunks) and all_chunks[j]["type"] == "text":
#                     related_texts.append(all_chunks[j]["content"])

#             # 표 마크다운 변환
#             table_md = table_to_markdown(chunk["headers"], chunk["rows"])

#             combined = "\n\n".join(related_texts + [table_md])

#             # 토큰 수 측정
#             # token_count = count_tokens(combined)

#             semantic_chunks.append({
#                 "content": combined,
#                 "metadata": {
#                     "file": chunk["file"],
#                     "page": chunk["page"]
#                 }
#             })

#     return semantic_chunks

# semantic_chunks = build_semantic_chunks_by_proximity(all_text_blocks + all_tables, window=1)

# # 예시 출력
# for i, chunk in enumerate(semantic_chunks[:3]):
#     print(f"\n--- Chunk {i+1} ---\n")
#     print(chunk["content"][:1000])  # 내용 일부만 출력

# import json

# # 1. Tokenizer 설정 (GPT-3.5 기준)
# encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

# def count_tokens(text):
#     return len(encoding.encode(text))

# def split_by_token_window(text, size=500, overlap=100):
#     tokens = encoding.encode(text)
#     chunks = []

#     start = 0
#     while start < len(tokens):
#         end = start + size
#         chunk_tokens = tokens[start:end]
#         chunk_text = encoding.decode(chunk_tokens)
#         chunks.append(chunk_text)

#         start += size - overlap

#     return chunks

# final_chunks = []

# for chunk in semantic_chunks:
#     split_contents = split_by_token_window(chunk["content"], size=500, overlap=100)
#     for split_text in split_contents:
#         final_chunks.append({
#             "content": split_text,
#             "metadata": chunk["metadata"]
#         })

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# # all_text_blocks.json 불러오기
# with open('all_text_blocks.json', 'r', encoding='utf-8') as f:
#     raw_blocks = json.load(f)

# # all_tables.json 불러오기
# # with open('all_tables.json', 'r', encoding='utf-8') as f:
# #     all_tables = json.load(f)

# all_text_blocks = [
#     Document(page_content=block["content"], metadata={"file": block["file"], "page": block["page"]})
#     for block in raw_blocks
# ]

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=100
# )

# all_text_blocks
# splits = text_splitter.split_documents(all_text_blocks)

# print("splits", len(splits))
# print(splits[:3])

# """### chunk 개수, 평균 토큰 확인"""

# # 모델 이름에 따라 tokenizer 설정
# encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

# def count_tokens(text):
#     return len(encoding.encode(text))

# # 전체 토큰 수와 통계 구하기
# token_counts = [count_tokens(chunk['content']) for chunk in final_chunks]

# avg_tokens = sum(token_counts) / len(token_counts)
# max_tokens = max(token_counts)
# min_tokens = min(token_counts)

# print(f"✅ 총 chunk 수: {len(final_chunks)}")
# print(f"📊 평균 토큰 수: {avg_tokens:.2f}")
# print(f"🔺 최대 토큰 수: {max_tokens}")
# print(f"🔻 최소 토큰 수: {min_tokens}")

# """## 임베딩"""

# # 텍스트 + 문서 청크 시 랭체인 document로 변환해주는 코드

# splits = [
#     Document(page_content=chunk["content"], metadata=chunk["metadata"])
#     for chunk in final_chunks
# ]

# # 임베딩
# embedding_model = UpstageEmbeddings(model="solar-embedding-1-large-passage")

# vectors = []
# documents = []

# for i, doc in enumerate(splits):
#     try:
#         vectorstore = FAISS.from_documents([doc], embedding=embedding_model)
#         if i == 0:
#             main_vectorstore = vectorstore
#         else:
#             main_vectorstore.merge_from(vectorstore)
#         time.sleep(1)  # 요청 속도 제한 방지
#     except Exception as e:
#         print(f"❗ Doc {i} 임베딩 실패: {e}")


# # 첫 번째 문서 확인
# doc = vectorstore.docstore.search("0")  # 또는 list(vectorstore.docstore._dict.keys())[0]
# print("👉 샘플 문서:", doc)


# from langchain_core.prompts import ChatPromptTemplate

# query ="D-8 비자에 대해 알려줘"

# # 4. Dense Retriever 생성
# retriever = vectorstore.as_retriever(
#     search_type="mmr",
#     search_kwargs={"k": 3},
# )

# # 5. ChatPromptTemplate 정의
# result_docs = retriever.invoke(query)

# # 1. 프롬프트 템플릿 정의
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             # system context
#             """
#             당신은 대한민국 비자/체류 자격 관련 전문가입니다. 다음 정보를 바탕으로 사용자의 질문에 친절하고 구체적으로 답해주세요.
#             답을 모르겠으면 대답하지 않아도 됩니다.
#             문서: {context}
#             """,
#         ),
#         ("human", "{input}"),
#     ]
# )


# # 2. LLM 모델 설정 (Upstage 예시)
# llm = ChatUpstage(
#     model="solar-1-mini-chat",  # 또는 "solar-1-mini-32k" 등 사용 가능
#     temperature=0.3
# )

# # 3. 프롬프트에 값 삽입 후 실행
# chain = prompt | llm | StrOutputParser()

# docs = retriever.invoke(query)

# # context는 docs에서 추출
# context_text = "\n\n".join([doc.page_content for doc in docs])
# response = chain.invoke({"input": query, "context": context_text})
# # response = chain.invoke({"question": query})

# print(response)

# """RAGAS 평가"""

# data_sets = [
#     {
#         "question": "D-8-4 비자란 무엇인가요?",
#         "answer": "D-8-4 비자는 기술 창업자를 위한 비자로, 국내외 학위 보유자나 정부 추천을 받은 사람이 신청할 수 있습니다.",
#         "contexts": [
#             "D-8-4 비자는 국내에서 전문학사 이상의 학위를 취득했거나 국외에서 학사 이상의 학위를 취득했거나, 관계 중앙행정기관의 장의 추천을 받은 자로서 지식재산권 또는 이에 준하는 기술력을 보유한 기술 창업자를 위한 비자입니다."
#         ],
#         "ground_truth": "D-8-4 비자는 국내에서 전문학사 이상의 학위를 취득했거나 국외에서 학사 이상의 학위를 취득했거나, 관계 중앙행정기관의 장의 추천을 받은 자로서 지식재산권 또는 이에 준하는 기술력을 보유한 기술 창업자를 위한 비자입니다."
#     },
#     {
#         "question": "D-8-4 비자에 필요한 서류는 무엇인가요?",
#         "answer": "사증신청서, 여권, 사진, 학위증명서, 지식재산권 관련 서류, OASIS 수료증 등이 필요합니다.",
#         "contexts": [
#             "D-8-4 비자에 필요한 서류로는 사증발급인정신청서(별지 제17호 서식), 여권, 규격 사진, 수수료가 포함되며, 법인설립신고확인서와 사업자등록증 사본, 학위증명서 또는 관계 중앙행정기관장의 추천서 사본이 포함됩니다. 또한, 포인트제 관련 항목(및 점수)을 증명하는 서류가 필요하며, 지식재산권 보유자는 특허증, 실용신안등록증, 디자인등록증 사본을 제출해야 합니다. 참고로, 특허청의 '특허정보넷 키프리스'(www.kipris.or.kr)에서 지식재산권 보유 여부를 확인할 수 있습니다. 특허 출원자는 특허청장이 발급한 출원사실증명서를 제출해야 하며, 법무부 장관이 지정한 이민창업지원 프로그램(OASIS) 이수(수료, 졸업) 증명서, 수상 증서, 공고문 등 관련 서류도 제출해야 합니다. 법무부의 출입국 정책에 따르면, OASIS 수료증의 유효기간은 발급일로부터 2년입니다. 포인트제 항목을 입증하는 기타 서류도 함께 제출해야 하며, 초청 목적, 초청 진정성, 초청자 및 피초청자의 자격 확인 등을 위해 외국 공공기관장은 첨부서류를 일부 조정할 수 있습니다."
#         ],
#         "ground_truth": "D-8-4 비자에 필요한 서류로는 사증발급인정신청서(별지 제17호 서식), 여권, 규격 사진, 수수료가 포함되며, 법인설립신고확인서와 사업자등록증 사본, 학위증명서 또는 관계 중앙행정기관장의 추천서 사본이 포함됩니다. 또한, 포인트제 관련 항목(및 점수)을 증명하는 서류가 필요하며, 지식재산권 보유자는 특허증, 실용신안등록증, 디자인등록증 사본을 제출해야 합니다. 참고로, 특허청의 '특허정보넷 키프리스'(www.kipris.or.kr)에서 지식재산권 보유 여부를 확인할 수 있습니다. 특허 출원자는 특허청장이 발급한 출원사실증명서를 제출해야 하며, 법무부 장관이 지정한 이민창업지원 프로그램(OASIS) 이수(수료, 졸업) 증명서, 수상 증서, 공고문 등 관련 서류도 제출해야 합니다. 법무부의 출입국 정책에 따르면, OASIS 수료증의 유효기간은 발급일로부터 2년입니다. 포인트제 항목을 입증하는 기타 서류도 함께 제출해야 하며, 초청 목적, 초청 진정성, 초청자 및 피초청자의 자격 확인 등을 위해 외국 공공기관장은 첨부서류를 일부 조정할 수 있습니다."
#     },
#     {
#         "question": "D-8 비자에는 어떤 하위 비자가 있나요?",
#         "answer": "D-8 비자는 총 5개 하위 유형으로 구성됩니다. D-8-1, D-8-2, D-8-3, D-8-4, D-8-4S입니다.",
#         "contexts": [
#             "기업투자 비자에는 외국인투자촉진법에 따른 외국인 투자기업의 대표자 대상 D-8-1 비자, 벤처기업육성에관한특별조치법에 따라 설립된 벤처기업의 창업자 대상 D-8-2 비자(중국 등 특정국가 제외), 내국인이 경영하는 기업에 투자하는 외국인을 위한 D-8-3 비자, 관련 학위를 보유하거나 중앙행정기관의 추천을 받은 기술창업자를 위한 D-8-4 비자, 스타트업 코리아 특별비자 민간평가위원회의 평가와 중소벤처기업부 장관의 추천을 받은 기술창업자를 위한 D-8-4S 비자가 포함됩니다."
#         ],
#         "ground_truth": "기업투자 비자에는 외국인투자촉진법에 따른 외국인 투자기업의 대표자 대상 D-8-1 비자, 벤처기업육성에관한특별조치법에 따라 설립된 벤처기업의 창업자 대상 D-8-2 비자(중국 등 특정국가 제외), 내국인이 경영하는 기업에 투자하는 외국인을 위한 D-8-3 비자, 관련 학위를 보유하거나 중앙행정기관의 추천을 받은 기술창업자를 위한 D-8-4 비자, 스타트업 코리아 특별비자 민간평가위원회의 평가와 중소벤처기업부 장관의 추천을 받은 기술창업자를 위한 D-8-4S 비자가 포함됩니다."
#     }
# ]


# # # 1. 질문 리스트
# # questions = [
# #     "D-8-4 비자는 누구를 위한 비자인가요?",
# #     "D-8-4 비자에 제출해야 하는 서류는 무엇인가요?",
# #     "D-8 비자의 하위 유형에는 무엇이 있나요?"
# # ]

# # 2. 빈 딕셔너리
# dense_data = {
#    "question": [],
#     "answer": [],
#     "contexts": [],
#     "ground_truth": [],
# }

# retriever = vectorstore.as_retriever(
#     search_type="mmr",
#     # lambda_mult (1에 가까울 수록 유사도, 0.1은 다양성 중심)
#     search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.9}
# )

# # 3. context 채우는 함수
# def fill_data(data, question, answer, retriever, ground_truth):
#     results = retriever.invoke(question)
#     context = [doc.page_content for doc in results]

#     data["question"].append(question)
#     data["answer"].append(answer)           # LLM 또는 수동 입력
#     data["contexts"].append(context)
#     data["ground_truth"].append(ground_truth)          # 기준 정답 입력

# # 4. fill_data로 데이터 채우기
# for data in enumerate(data_sets):
#     fill_data(dense_data, data["question"], data["answer"], retriever, data["ground_truth"])

# # 5. Dataset 생성
# dense_dataset = Dataset.from_dict(dense_data)

# # 6. 평가
# result = evaluate(
#     dataset=dense_dataset,
#     metrics=[context_precision, context_recall, faithfulness],
#     llm=llm,                # faithfulness 등 LLM 기반 메트릭에 필요
#     embeddings= UpstageEmbeddings(model="embedding-query")
#     # retriever = retriever
# )

# print(result)
