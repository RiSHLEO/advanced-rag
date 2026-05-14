import os
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import re

def load_and_chunk_document(pdf_path: str, chunk_size: int = 500, 
                             chunk_overlap: int = 100) -> List[Document]:
    """Load PDF and split into chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    print(f"Document split into {len(chunks)} chunks")
    return chunks

def build_vector_store(chunks: List[Document], api_key: str) -> Chroma:
    """Build ChromaDB vector store from chunks."""
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    return vectorstore

def build_bm25_index(chunks: List[Document]) -> Tuple[BM25Okapi, List[Document]]:
    """Build BM25 keyword search index from chunks."""
    
    def tokenize(text: str) -> List[str]:
        # Lowercase and split on non-alphanumeric characters
        text = text.lower()
        tokens = re.split(r'[^a-z0-9]', text)
        # Remove empty strings
        return [t for t in tokens if t]
    
    tokenized_chunks = [tokenize(chunk.page_content) for chunk in chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    
    return bm25_index, chunks

def semantic_search(vectorstore: Chroma, query: str, k: int = 10) -> List[Document]:
    """Search using semantic similarity."""
    return vectorstore.similarity_search(query, k=k)

def bm25_search(bm25_index: BM25Okapi, chunks: List[Document], 
                query: str, k: int = 10) -> List[Document]:
    """Search using BM25 keyword matching."""
    
    def tokenize(text: str) -> List[str]:
        text = text.lower()
        tokens = re.split(r'[^a-z0-9]', text)
        return [t for t in tokens if t]
    
    tokenized_query = tokenize(query)
    scores = bm25_index.get_scores(tokenized_query)
    
    # Get top k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    
    return [chunks[i] for i in top_indices]

def hybrid_search(vectorstore: Chroma, bm25_index: BM25Okapi, 
                  chunks: List[Document], query: str, k: int = 10) -> List[Document]:
    """
    Combine semantic and BM25 search results.
    Merges results from both, removes duplicates, returns top k.
    """
    
    # Get results from both
    semantic_results = semantic_search(vectorstore, query, k=k)
    bm25_results = bm25_search(bm25_index, chunks, query, k=k)
    
    # Merge and deduplicate by content
    seen_content = set()
    merged_results = []
    
    # Interleave results — take one from each alternately
    # This gives equal weight to both search methods
    max_len = max(len(semantic_results), len(bm25_results))
    
    for i in range(max_len):
        if i < len(semantic_results):
            content = semantic_results[i].page_content
            if content not in seen_content:
                seen_content.add(content)
                merged_results.append(semantic_results[i])
        
        if i < len(bm25_results):
            content = bm25_results[i].page_content
            if content not in seen_content:
                seen_content.add(content)
                merged_results.append(bm25_results[i])
    
    return merged_results[:k]