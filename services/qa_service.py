import os
from openai import OpenAI
from services.vector_service import enhanced_vector_service
from dotenv import load_dotenv
import re

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class QASystem:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.item_code_pattern = re.compile(r'\b[A-Z]\d{4}[A-Z]?\b')
    
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
            
            # Boost score for content relevance (simple keyword matching)
            query_words = query.lower().split()
            content_lower = chunk.page_content.lower()
            for word in query_words:
                if len(word) > 3 and word in content_lower:
                    score += 1
            
            ranked_chunks.append((score, chunk))
        
        # Sort by score (descending) and return chunks
        ranked_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in ranked_chunks]
    
    async def answer_query(self, question: str, k: int = 7):
        """Enhanced query answering with specific logic."""
        
        # Get enhanced search results
        search_results = enhanced_vector_service.get_related_chunks(question, k)
        retrieved_chunks = search_results['results']
        
        if not retrieved_chunks:
            return {
                "question": question,
                "answer": "I couldn't find relevant information in the OASIS documentation to answer your question.",
                "retrieved_chunks": [],
                "debug_info": search_results['debug_info']
            }
        
        # Clean and rank context
        ranked_chunks = self.clean_and_rank_context(retrieved_chunks, question)
        
        # Build enhanced context with metadata
        context_parts = []
        for i, chunk in enumerate(ranked_chunks, 1):
            metadata = chunk.metadata
            
            # Add context header
            context_header = f"\n--- Context {i} ---\n"
            if metadata.get('item_code'):
                context_header += f"Item Code: {metadata.get('item_code')}\n"
            if metadata.get('section'):
                context_header += f"Section: {metadata.get('section')}\n"
            if metadata.get('chapter'):
                context_header += f"Chapter: {metadata.get('chapter')}\n"
            context_header += f"Type: {metadata.get('chunk_type', 'unknown')}\n"
            context_header += "Content:\n"
            
            context_parts.append(context_header + chunk.page_content)
        
        context_text = "\n".join(context_parts)
        
        # Enhanced system prompt for OASIS
        system_prompt = """You are an expert OASIS (Outcome and Assessment Information Set) documentation assistant. 

Key guidelines:
1. When users ask about specific OASIS item codes (like M1845, A1110B, C1310C), provide the exact item information including coding instructions
2. Always cite specific item codes when they are relevant to the answer
3. If the context contains coding instructions, include them in your response
4. Explain OASIS concepts clearly, including time points, response options, and clinical scenarios
5. If you don't find specific information in the context, say so clearly
6. When discussing functional assessments, include the specific response scales and coding guidance

Use the provided context documents to answer questions accurately and comprehensively."""
        
        # Enhanced user prompt
        user_prompt = f"""Based on the OASIS documentation context provided below, please answer the following question:

Question: {question}

Context from OASIS Documentation:
{context_text}

Please provide a comprehensive answer that includes:
- Specific OASIS item codes if relevant
- Coding instructions if applicable  
- Time points when assessments are completed
- Any relevant examples or scenarios
- Clear explanations of OASIS concepts

If the question asks about a specific item code that's not found in the context, please mention that explicitly."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,  # Lower temperature for more consistent responses
            )
            
            answer = response.choices[0].message.content
            
            return {
                "question": question,
                "answer": answer,
                "retrieved_chunks": [chunk.page_content for chunk in ranked_chunks],
                "debug_info": {
                    **search_results['debug_info'],
                    "context_length": len(context_text),
                    "num_chunks_used": len(ranked_chunks)
                }
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"Error generating response: {str(e)}",
                "retrieved_chunks": [chunk.page_content for chunk in ranked_chunks],
                "debug_info": search_results['debug_info']
            }

# Initialize the QA system
qa_system = QASystem()

# Wrapper function for backward compatibility
async def answer_query(question: str, k: int = 7):
    """Enhanced query answering function."""
    return await qa_system.answer_query(question, k)


