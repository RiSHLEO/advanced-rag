import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from hybrid_search import load_and_chunk_document, build_vector_store, build_bm25_index
from query_rewriter import search_with_rewrites
from reranker import rerank_chunks

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def build_advanced_pipeline(pdf_path: str, chunk_size: int = 500,
                             chunk_overlap: int = 100):
    """
    Build the complete advanced RAG pipeline.
    Returns a callable that takes a question and returns answer + retrieved chunks.
    """
    
    print("Loading and chunking document...")
    chunks = load_and_chunk_document(pdf_path, chunk_size, chunk_overlap)
    
    print("Building vector store...")
    vectorstore = build_vector_store(chunks, api_key)
    
    print("Building BM25 index...")
    bm25_index, _ = build_bm25_index(chunks)
    
    # LLM for generation
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)
    
    prompt = PromptTemplate(
        template="""You are a precise assistant. Answer the question using ONLY 
the information provided in the context below.

STRICT RULES:
1. Only use information explicitly stated in the context
2. Quote relevant numbers or facts directly from the context
3. Do not add any information from your general knowledge
4. If the context does not contain enough information, say exactly:
   "The context does not contain enough information to answer this question."
5. Keep your answer concise and directly focused on what was asked

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )
    
    def run_query(question: str) -> Tuple[str, List[Document]]:
        """Run the full advanced RAG pipeline for one question."""
        
        print(f"\nProcessing: {question}")
        
        # Step 1 — Query rewriting + hybrid search
        print("Step 1: Query rewriting and hybrid search...")
        candidates = search_with_rewrites(
            vectorstore, bm25_index, chunks, question, k=10
        )
        print(f"  Found {len(candidates)} candidate chunks")
        
        # Step 2 — Re-rank candidates
        print("Step 2: Re-ranking with cross-encoder...")
        top_chunks = rerank_chunks(question, candidates, top_k=3)
        print(f"  Re-ranked to top {len(top_chunks)} chunks")
        
        # Step 3 — Generate answer
        context = "\n\n".join([chunk.page_content for chunk in top_chunks])
        
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        
        return answer, top_chunks
    
    return run_query