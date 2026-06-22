# Latency Breakdown — Production RAG Pipeline

| Bước | Thời gian |
|------|-----------|
| M1 Chunking | 0.54s |
| M5 Enrichment (LLM) | 419.14s |
| M2 Indexing (BM25 + Dense embed) | 118.26s |
| M3 Reranker load (warm-up) | 0.00s |
| Avg / query (search + rerank + LLM answer) | 9171ms |
| **Tổng end-to-end** | **848.8s** |

_Ghi chú: enrichment tốn nhất vì gọi LLM 1 lần/chunk (combined mode); indexing gồm embedding toàn bộ chunk qua Gemini text-embedding (768-dim) + upsert Qdrant Cloud._