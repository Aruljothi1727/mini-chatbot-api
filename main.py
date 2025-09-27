import os
import re
import uuid
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from sqlalchemy import create_engine
from langchain.schema import BaseRetriever

load_dotenv()
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)



DATABASE_URL = os.getenv("DB_CONNECTION")  
engine = create_engine(DATABASE_URL)

# ------------------- FastAPI Setup -------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global dictionary to store session memories
memory_sessions = {}

def get_memory_for_session_chain(session_id: str, k: int = 20):
    """Create or get sliding window memory for a session."""
    if session_id not in memory_sessions:
        memory_sessions[session_id] = ConversationBufferWindowMemory(
             memory_key="history",
            input_key="input",
            output_key="output",
            k=k,                     # keep only last k Q/A
            return_messages=False    # return history as string
        )
    memory = memory_sessions[session_id]
  
    chain = ConversationalRetrievalChain.from_llm(
       llm=llm,
    retriever=my_retriever,
    memory=memory
    )

    return memory,chain

# ------------------- Vector Service -------------------
class VectorService:
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.vector_store = PGVector(
            embeddings=self.embedding_model,
            collection_name="doc_vector",
            connection=engine,  
            use_jsonb=True,
        )
        self.item_code_pattern = re.compile(r'\b[A-Z]\d{4}[A-Z]?\b')

    def search(self, query: str, k: int = 7) -> List[Any]:
        return self.vector_store.similarity_search(query, k=k)

vector_service = VectorService()
my_retriever = vector_service.vector_store.as_retriever(search_kwargs={"k": 7})


# ------------------- QA System -------------------
class QASystem:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.item_code_pattern = re.compile(r'\b[A-Z]\d{4}[A-Z]?\b')

    def normalize_query(self, question: str) -> str:
        question = question.strip()
        if not question.endswith("?"):
            question += "?"
        return question

    def clean_and_rank_context(self, retrieved_chunks, query):
        query_item_codes = self.item_code_pattern.findall(query.upper())
        ranked_chunks = []

        for chunk in retrieved_chunks:
            score = 0
            chunk_item_codes = chunk.metadata.get('item_codes', [])

            for query_code in query_item_codes:
                if query_code in chunk_item_codes:
                    score += 10

            chunk_type = chunk.metadata.get('chunk_type', '')
            if chunk_type == 'item':
                score += 5
            elif chunk_type == 'section':
                score += 3

            query_words = query.lower().split()
            content_lower = chunk.page_content.lower()
            for word in query_words:
                if len(word) > 3 and word in content_lower:
                    score += 1

            ranked_chunks.append((score, chunk))

        ranked_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in ranked_chunks]

    async def answer_query(self, question: str, k: int = 7, sessionId: str = None):
        normalized_question = self.normalize_query(question)

        # Step 1: Get conversation memory
        memory,chain = get_memory_for_session_chain(sessionId)
        history = memory.load_memory_variables({}).get("history", "")
        # print(f"\n[DEBUG] Loaded history for session {sessionId}:\n{history}\n")

        # Step 2: Search top chunks
        retrieved_chunks_with_scores = vector_service.vector_store.similarity_search_with_score(normalized_question, k=k)
        retrieved_chunks = [doc for doc, score in retrieved_chunks_with_scores if score >= 0.1]
        if not retrieved_chunks:
          return {
        "question": question,
        "answer": "No relevant information found in documentation.",
        "retrieved_chunks": [],
        "sessionId": sessionId
        }

        # Step 3: Rank and clean
        ranked_chunks = self.clean_and_rank_context(retrieved_chunks, normalized_question)

        # Step 4: Build context text including conversation history
        context_parts = []
        if history:
            context_parts.append(f"Conversation History:\n{history}")

        for i, chunk in enumerate(ranked_chunks, 1):
            meta = chunk.metadata
            header = f"\n--- Context {i} ---\n"
            if meta.get('item_code'):
                header += f"Item Code: {meta.get('item_code')}\n"
            if meta.get('section'):
                header += f"Section: {meta.get('section')}\n"
            if meta.get('chapter'):
                header += f"Chapter: {meta.get('chapter')}\n"
            header += f"Type: {meta.get('chunk_type','unknown')}\nContent:\n"
            context_parts.append(header + chunk.page_content)
        context_text = "\n".join(context_parts)

        system_prompt = """You are an expert OASIS documentation consultant.
- Only answer questions that are related to OASIS documentation.
- If the question is not related, respond: 'This question is outside the scope of OASIS documentation.'
- Provide concise answers in your own words using the context.
- Cite the **correct OASIS-E item code(s)** only if they are directly relevant to the question (e.g., GG0170, M1630). 
  • Do not cite unrelated codes from other sections.  
  • If uncertain about a code, state that clearly instead of guessing.
- Include instructions and examples only if in context.
- Always maintain context continuity across questions in a session.
- When a question relates to a previous topic, anchor your answer to that topic unless the user explicitly changes subjects. 
- Before answering, check previous responses in the session for relevant context to ensure consistency and avoid contradictions.
- Ensure explanations remain consistent with prior answers and maintain coherence throughout the conversation.  
- Keep answers within 500-700 tokens.
- If information is missing, state that clearly."""

        user_prompt = f"Question: {normalized_question}\n\nContext:\n{context_text}"

        # print(f"\n[DEBUG] Sending to LLM (session={sessionId}):")
        # print(f"System Prompt:\n{system_prompt}\n")
        # print(f"User Prompt (first 1000 chars):\n{user_prompt[:1000]}...\n")
 

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,
                max_tokens=700
            )
            answer = response.choices[0].message.content

            # Step 5: Save current question/answer to session memory
            memory.save_context({"input": question}, {"output": answer})

            return {
                "question": question,
                "answer": answer,
                "sessionId": sessionId,
              # "retrieved_chunks": [c.page_content for c in ranked_chunks],
            }
        except Exception as e:
            return {
                "question": question,
                "answer": f"Error generating response: {str(e)}",
                "retrieved_chunks": [c.page_content for c in ranked_chunks],
                "sessionId": sessionId
            }

qa_system = QASystem()

# ------------------- FastAPI Endpoint -------------------
class QueryRequest(BaseModel):
    question: str
    k: int = 7
    sessionId: Optional[str] = None

def generate_session_id():
    return str(uuid.uuid4())

def valid_format(session_id):
    return bool(re.match(
        r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$',
        session_id or ""
    ))

@app.post("/query")
async def query_docs(request: Request, body: QueryRequest):
    session_id = request.headers.get("session-id")
    if not session_id or not valid_format(session_id):
        session_id = generate_session_id()
        # print("New session id generated:", session_id)

    return await qa_system.answer_query(body.question, body.k, session_id)
