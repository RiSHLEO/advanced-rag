# Advanced RAG System

A production-grade Retrieval Augmented Generation system implementing four 
advanced techniques — hybrid search, cross-encoder re-ranking, query rewriting, 
and configurable chunking — built to address specific weaknesses identified 
through systematic RAG evaluation.

**Live App:** [Click here to view the app](https://advanced-rag-metuk48d5i2tgww2bdyjuy.streamlit.app/)

---

## Background — Why Advanced RAG?

This project was built as a direct response to findings from my 
[RAG Evaluation Dashboard](https://github.com/RiSHLEO/rag-evaluation).

Evaluation of a naive RAG pipeline revealed:
- **Context Recall: 0.62** — two questions completely failed retrieval
- **Faithfulness: 0.59** — model hallucinating from training data
- Operating expenses question returned "context does not contain enough 
  information" despite the answer existing in the document

Each advanced technique in this project directly targets one of those failures.

---

## The Four Advanced Techniques

### 1. Hybrid Search
Combines semantic similarity search with BM25 keyword search.

**Why:** Semantic search understands meaning but misses specific financial 
figures and exact terms. BM25 finds exact keyword matches but has no 
semantic understanding. Together they cover each other's blind spots.

**Result:** Questions that previously returned no results — like operating 
expenses — are now found through BM25 keyword matching.

### 2. Cross-Encoder Re-ranking
After hybrid search retrieves 10 candidate chunks, a cross-encoder model 
scores each question-chunk pair together and selects the top 3.

**Why:** Embedding similarity scores each piece of text independently. 
A cross-encoder reads the question and chunk together, giving a more 
accurate relevance score. Slower but significantly more precise.

**Model used:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — runs locally, 
no API cost.

### 3. Query Rewriting
Before searching, GPT generates 3 alternative phrasings of the question. 
All versions are searched and results are merged.

**Why:** The user's exact phrasing might not match how information is 
expressed in the document. Alternative phrasings increase the chance of 
finding relevant chunks.

**Example:**

Original: "What were Tesla's total operating expenses in 2023?"
Rewrite 1: "How much money did Tesla spend on operations in 2023?"
Rewrite 2: "What was the total cost of running Tesla in 2023?"
Rewrite 3: "Sum of Tesla's operating expenses for the year 2023"

### 4. Configurable Chunking
Chunk size is adjustable via the sidebar. Financial documents specifically 
require larger chunks (1000+) to avoid splitting tables mid-row.

**Finding:** chunk_size=500 caused the income statement table to split, 
hiding operating expense figures. chunk_size=1000 kept the table intact 
and the question was answered correctly.

---

## Evaluation Results

Both naive and advanced pipelines were evaluated using RAGAS metrics on 
Tesla's 2023 Annual Report with 8 test questions.

| Metric | Naive RAG (cs=500) | Advanced RAG (cs=500) | Change |
|---|---|---|---|
| Faithfulness | 0.81 | 0.50 | Complex tradeoff |
| Answer Relevancy | 0.81 | 0.78 | Comparable |
| Context Precision | 1.00 | 0.93 | Slight decrease |
| Context Recall | 0.62 | 0.88 | **+0.26 improvement** |
| Overall | 0.81 | 0.77 | Tradeoff |

**Key finding:** Advanced RAG significantly improved context recall from 
0.62 to 0.88 — recovering questions that naive retrieval completely missed. 
However query rewriting introduced noise in some configurations, reducing 
faithfulness. This demonstrates that RAG improvements involve genuine 
engineering tradeoffs — the right configuration depends on which metric 
matters most for the specific use case.

---

## Architecture

User question
↓
Query Rewriting — GPT generates 3 alternative phrasings
↓
Hybrid Search — semantic + BM25 on all query versions
↓
Merge & deduplicate — 10 candidate chunks
↓
Cross-encoder Re-ranking — score each question-chunk pair
↓
Top 3 chunks selected
↓
GPT generates answer from top 3 chunks
↓
Answer + source chunks returned to user

---

## Technical Stack

- **LLM:** GPT-3.5-turbo via OpenAI API
- **Embeddings:** OpenAI text-embedding-ada-002
- **Vector Database:** ChromaDB
- **Keyword Search:** BM25 via rank-bm25
- **Re-ranking:** cross-encoder/ms-marco-MiniLM-L-6-v2 via sentence-transformers
- **RAG Framework:** LangChain with LCEL
- **Frontend:** Streamlit

---

## How to Run Locally

```bash
git clone https://github.com/RiSHLEO/advanced-rag
cd advanced-rag
pip install -r requirements.txt
```

Create a `.env` file: OPENAI_API_KEY=your-key-here

Then run:
```bash
cd app
streamlit run app.py
```

Upload any PDF and ask questions. Retrieved source chunks are shown 
for every answer so you can verify what information was used.

---

## Related Projects

This project is part of a three-project RAG series:

1. [RAG Document Chatbot](https://github.com/RiSHLEO/rag-document-chatbot) — Naive RAG implementation
2. [RAG Evaluation Dashboard](https://github.com/RiSHLEO/rag-evaluation) — Systematic evaluation using RAGAS
3. **Advanced RAG System** (this project) — Targeted improvements based on evaluation findings

---

## What I Would Improve With More Time

- **Parent-child chunking** — store small chunks for retrieval but 
  return larger parent chunks to the LLM for richer context
- **Semantic chunking** — split on meaning boundaries rather than 
  character count, keeping related sentences together
- **Adaptive retrieval** — automatically select between naive and 
  advanced retrieval based on query complexity
- **Caching** — store embeddings and BM25 index so the pipeline 
  doesn't rebuild on every upload
- **Streaming** — show the answer token by token as GPT generates it
- **Multi-document support** — index multiple PDFs simultaneously 
  with document-level metadata filtering