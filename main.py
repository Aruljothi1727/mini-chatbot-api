import os
import re
from dotenv import load_dotenv
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

# ------------------- FastAPI Setup -------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            connection=os.getenv("DB_CONNECTION"),
            use_jsonb=True,
        )
        self.item_code_pattern = re.compile(r'\b[A-Z]\d{4}[A-Z]?\b')

    def search(self, query: str, k: int = 7) -> List[Any]:
        """Search vectors and return top chunks"""
        return self.vector_store.similarity_search(query, k=k)

vector_service = VectorService()

# ------------------- QA System -------------------
class QASystem:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.item_code_pattern = re.compile(r'\b[A-Z]\d{4}[A-Z]?\b')

    def normalize_query(self, question: str) -> str:
        """Normalize query to improve retrieval (add ? if missing)."""
        question = question.strip()
        if not question.endswith("?"):
            question += "?"
        return question

    def clean_and_rank_context(self, retrieved_chunks, query):
        """Clean and rank retrieved chunks for better context."""
        query_item_codes = self.item_code_pattern.findall(query.upper())
        ranked_chunks = []

        for chunk in retrieved_chunks:
            score = 0
            chunk_item_codes = chunk.metadata.get('item_codes', [])

            # Boost score for exact item code matches
            for query_code in query_item_codes:
                if query_code in chunk_item_codes:
                    score += 10

            # Boost score for chunk type relevance
            chunk_type = chunk.metadata.get('chunk_type', '')
            if chunk_type == 'item':
                score += 5
            elif chunk_type == 'section':
                score += 3

            # Boost score for content relevance (keyword match)
            query_words = query.lower().split()
            content_lower = chunk.page_content.lower()
            for word in query_words:
                if len(word) > 3 and word in content_lower:
                    score += 1

            ranked_chunks.append((score, chunk))

        ranked_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in ranked_chunks]

    async def answer_query(self, question: str, k: int = 7):
        """Search vectors, rank, and generate GPT answer with context."""
        normalized_question = self.normalize_query(question)

        # Step 1: search top chunks
        retrieved_chunks = vector_service.search(normalized_question, k=k)
        if not retrieved_chunks:
            return {
                "question": question,
                "answer": "No relevant information found in documentation.",
                "retrieved_chunks": [],
            }

        # Step 2: rank and clean
        ranked_chunks = self.clean_and_rank_context(retrieved_chunks, normalized_question)

        # Step 3: build context text
        context_parts = []
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

        # Step 4: build GPT prompt
        system_prompt = """You are an expert OASIS (Outcome and Assessment Information Set) documentation consultant.
- Provide accurate, concise answers in your own words using the context.
- Cite relevant item codes if applicable.
- Include coding instructions, time points, and examples only if in context.
- Make answers crisp, clear, and summarized.
- Respond in 500-700 tokens max.
- If information is missing, state that clearly.
- Avoid directly quoting manuals; explain professionally and clearly."""

        user_prompt = f"Question: {normalized_question}\n\nContext:\n{context_text}\n\nAnswer concisely and within 500-700 tokens."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Step 5: call OpenAI
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,
                max_tokens=700
            )
            answer = response.choices[0].message.content
            return {
                "question": question,
                "answer": answer,
                "retrieved_chunks": [c.page_content for c in ranked_chunks],
            }
        except Exception as e:
            return {
                "question": question,
                "answer": f"Error generating response: {str(e)}",
                "retrieved_chunks": [c.page_content for c in ranked_chunks],
            }

# Init QA system
qa_system = QASystem()

# ------------------- FastAPI Endpoint -------------------
class QueryRequest(BaseModel):
    question: str
    k: int = 7

@app.post("/query")
async def query_docs(request: QueryRequest):
    return await qa_system.answer_query(request.question, request.k)
