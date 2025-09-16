from langchain_postgres import PGVector
import os
import re
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings

load_dotenv()

class EnhancedVectorService:
    def __init__(self):
        # Initialize HuggingFace embedding model
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small"  
        )

        # PostgreSQL connection
        connection = os.getenv("DB_CONNECTION")
        collection_name = "doc_vector"  
        
        # Vector store with enhanced settings
        self.vector_store = PGVector(
            embeddings=self.embedding_model,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )
        
        # OASIS item code pattern for enhanced search
        self.item_code_pattern = re.compile(r'\b[A-Z]\d{4}[A-Z]?\b')
    
    def add_to_vectorstore(self, chunks: List[Any], batch_size: int = 32):
        """Store documents in PGVector with batch embeddings."""
        if not chunks:
            print("No chunks provided")
            return

        # Extract texts and metadata
        texts = [chunk.page_content for chunk in chunks]
        metadatas = []

        # Enhance metadata for each chunk
        for chunk in chunks:
            item_codes = self.item_code_pattern.findall(chunk.page_content)
            enhanced_metadata = chunk.metadata.copy()
            enhanced_metadata['item_codes'] = item_codes
            enhanced_metadata['has_item_codes'] = len(item_codes) > 0
            enhanced_metadata['content_length'] = len(chunk.page_content)
            metadatas.append(enhanced_metadata)

        # Compute embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)

        # Add documents to vector store
        for chunk, embedding, metadata in zip(chunks, all_embeddings, metadatas):
            self.vector_store.add_documents([
                type(chunk)(page_content=chunk.page_content, metadata=metadata)
            ])

        print(f"Added {len(chunks)} chunks to vector store with batch embeddings")
    
    def search_with_item_codes(self, query: str, k: int = 5) -> List[Any]:
        """Enhanced search that prioritizes item code matches."""
        results = []
        query_item_codes = self.item_code_pattern.findall(query.upper())
        
        if query_item_codes:
            print(f"Found item codes in query: {query_item_codes}")
            for item_code in query_item_codes:
                item_specific_query = f"item code {item_code}"
                item_results = self.vector_store.similarity_search(
                    item_specific_query, k=k, filter={"has_item_codes": True}
                )
                for result in item_results:
                    if item_code in result.metadata.get('item_codes', []):
                        if result not in results:
                            results.append(result)
            if results:
                general_results = self.vector_store.similarity_search(
                    query, k=max(2, k-len(results))
                )
                for result in general_results:
                    if result not in results:
                        results.append(result)
                return results[:k]
        return self.vector_store.similarity_search(query, k=k)
    
    def search_vectorstore(self, query: str, k: int = 5) -> List[Any]:
        """Main search function with multiple strategies."""
        results = self.search_with_item_codes(query, k)
        if len(results) < k // 2:
            broader_results = self.vector_store.similarity_search(query, k=k*2)
            seen_content = set()
            final_results = []
            for result in results:
                content_hash = hash(result.page_content[:100])
                if content_hash not in seen_content:
                    final_results.append(result)
                    seen_content.add(content_hash)
            for result in broader_results:
                content_hash = hash(result.page_content[:100])
                if content_hash not in seen_content and len(final_results) < k:
                    final_results.append(result)
                    seen_content.add(content_hash)
            return final_results[:k]
        return results
    
    def get_related_chunks(self, query: str, k: int = 7) -> Dict[str, Any]:
        """Get related chunks with enhanced context and debugging info."""
        results = self.search_vectorstore(query, k)
        query_item_codes = self.item_code_pattern.findall(query.upper())
        return {
            'query': query,
            'query_item_codes': query_item_codes,
            'num_results': len(results),
            'results': results,
            'debug_info': {
                'result_types': [r.metadata.get('chunk_type', 'unknown') for r in results],
                'result_item_codes': [r.metadata.get('item_codes', []) for r in results],
                'result_sources': [r.metadata.get('section', 'unknown') for r in results]
            }
        }

# Initialize the enhanced service
enhanced_vector_service = EnhancedVectorService()

# Wrapper functions for backward compatibility
def add_to_vectorstore(chunks):
    return enhanced_vector_service.add_to_vectorstore(chunks)

def search_vectorstore(query: str, k: int = 7):
    return enhanced_vector_service.search_vectorstore(query, k)
