# Codebase RAG Assistant

A Retrieval-Augmented Generation (RAG) system for querying GitHub repositories using natural language. The system is designed with a focus on retrieval quality, performance optimization, and iterative evaluation.

---

## Versions

### v1 – Vector RAG
- FAISS-based semantic retrieval
- Repository ingestion → chunking → embeddings → vector search → LLM response
- Evaluation using precision, recall, and MRR

---

### v2 – Hybrid RAG
- Combined BM25 keyword search with vector similarity search
- Used Reciprocal Rank Fusion (RRF) to merge retrieval results

#### System Improvements
- Implemented FAISS index caching (save/load index)
- Eliminated repeated embedding computation on restart
- Significantly reduced application startup time

#### Retrieval Improvements
- Handled filename-based queries (e.g., "day-02")
- Token normalization for structured identifiers (day-02, day_02, day 02)
- Included file paths in chunk embeddings
- Case normalization for consistent matching
- Filename-aware boosting (token-level and exact match)
- Deduplication of retrieved chunks
- Increased top_k to improve context coverage
- Filtered irrelevant directories (e.g., .git, __pycache__) during ingestion

---

## Evaluation
- Experiment tracking using MLflow
- Metrics:
  - Precision@k
  - Recall@k
  - MRR (Mean Reciprocal Rank)
  - Context Precision
  - Faithfulness

- Compared vector vs hybrid retrieval across different repository types
- Evaluated both retrieval quality and LLM response reliability

---

## Ongoing Work (v3)
- Re-ranking models for improved result ordering
- Graph-based retrieval for code relationships
- Deployment and latency monitoring

---

## Tech Stack
Python, LangChain, OpenAI API, FAISS, BM25, MLflow, Streamlit