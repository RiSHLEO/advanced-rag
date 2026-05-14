import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def rewrite_query(query: str, num_rewrites: int = 3) -> List[str]:
    """
    Generate multiple alternative phrasings of the query.
    More diverse phrasings = better chance of finding relevant chunks.
    """
    
    prompt = f"""Generate {num_rewrites} different ways to ask the following question.
Each rewrite should approach the question from a slightly different angle or use different terminology.
Return only the rewritten questions, one per line, no numbering or extra text.

Original question: {query}

Rewritten questions:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    rewrites = response.choices[0].message.content.strip().split("\n")
    rewrites = [r.strip() for r in rewrites if r.strip()]
    
    # Always include the original query
    all_queries = [query] + rewrites
    
    return all_queries

def search_with_rewrites(vectorstore, bm25_index, chunks, query: str, k: int = 10):
    """
    Search using original query plus all rewrites.
    Merge and deduplicate results.
    """
    from hybrid_search import hybrid_search
    
    # Get all query versions
    all_queries = rewrite_query(query)
    
    print(f"Searching with {len(all_queries)} query versions:")
    for q in all_queries:
        print(f"  - {q}")
    
    # Search with each query version
    seen_content = set()
    all_results = []
    
    for q in all_queries:
        results = hybrid_search(vectorstore, bm25_index, chunks, q, k=5)
        
        for chunk in results:
            if chunk.page_content not in seen_content:
                seen_content.add(chunk.page_content)
                all_results.append(chunk)
    
    return all_results[:k]