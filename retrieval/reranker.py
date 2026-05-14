from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# Load the cross-encoder model
# This downloads once and caches locally
print("Loading cross-encoder model...")
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Cross-encoder loaded")

def rerank_chunks(query: str, chunks: List[Document], top_k: int = 3) -> List[Document]:
    """
    Re-rank chunks using a cross-encoder model.
    Takes query + each chunk together and scores their relevance.
    Returns top_k most relevant chunks.
    """
    
    if not chunks:
        return []
    
    # Create pairs of [query, chunk_text] for the cross-encoder
    pairs = [[query, chunk.page_content] for chunk in chunks]
    
    # Score each pair
    scores = reranker_model.predict(pairs)
    
    # Sort chunks by score descending
    scored_chunks = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Return top k chunks
    return [chunk for chunk, score in scored_chunks[:top_k]]