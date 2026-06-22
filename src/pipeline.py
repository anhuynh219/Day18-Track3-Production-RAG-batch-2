from __future__ import annotations

"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K

# Thu thập latency từng bước để xuất bảng breakdown (bonus).
STAGE_TIMINGS: dict[str, float] = {}


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    STAGE_TIMINGS["chunking"] = time.time() - t0
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({STAGE_TIMINGS['chunking']:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)
    STAGE_TIMINGS["enrichment"] = time.time() - t0

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    STAGE_TIMINGS["indexing"] = time.time() - t0
    print(f"  ✓ Indexed ({STAGE_TIMINGS['indexing']:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    reranker._load_model()   # warm-up để loại thời gian load khỏi latency truy vấn
    STAGE_TIMINGS["reranker_load"] = time.time() - t0
    print(f"  ✓ Reranker ready ({STAGE_TIMINGS['reranker_load']:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    from config import get_llm_client, LLM_CHAT_MODEL
    client = get_llm_client()
    if client and contexts:
        try:
            context_str = "\n\n".join(contexts)
            system_prompt = (
                "Bạn trả lời câu hỏi CHỈ dựa trên CONTEXT. Tuân thủ nghiêm:\n"
                "1. Mọi con số, mốc thời gian, điều khoản phải lấy NGUYÊN VĂN từ context.\n"
                "2. Nếu cần tính toán, viết RÕ phép tính bằng các số có trong context "
                "(vd: '85% × 20.000.000 = 17.000.000').\n"
                "3. Trả lời NGẮN GỌN, KHÔNG thêm thông tin/kiến thức ngoài context.\n"
                "4. Nếu context không chứa thông tin → trả lời đúng câu: "
                "'Không tìm thấy thông tin trong tài liệu.'"
            )
            resp = client.chat.completions.create(
                model=LLM_CHAT_MODEL, temperature=0.0, messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
                ])
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []
    query_times = []

    for i, item in enumerate(test_set):
        tq = time.time()
        answer, contexts = run_query(item["question"], search, reranker)
        query_times.append(time.time() - tq)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    STAGE_TIMINGS["avg_query_s"] = sum(query_times) / len(query_times) if query_times else 0.0

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    os.makedirs("reports", exist_ok=True)
    save_report(results, failures, path="reports/ragas_report.json")
    return results


def write_latency_report(total_s: float, path: str = "reports/latency_breakdown.md"):
    """Xuất bảng latency breakdown từng bước (bonus +2)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# Latency Breakdown — Production RAG Pipeline\n",
        "| Bước | Thời gian |",
        "|------|-----------|",
        f"| M1 Chunking | {STAGE_TIMINGS.get('chunking', 0):.2f}s |",
        f"| M5 Enrichment (LLM) | {STAGE_TIMINGS.get('enrichment', 0):.2f}s |",
        f"| M2 Indexing (BM25 + Dense embed) | {STAGE_TIMINGS.get('indexing', 0):.2f}s |",
        f"| M3 Reranker load (warm-up) | {STAGE_TIMINGS.get('reranker_load', 0):.2f}s |",
        f"| Avg / query (search + rerank + LLM answer) | {STAGE_TIMINGS.get('avg_query_s', 0)*1000:.0f}ms |",
        f"| **Tổng end-to-end** | **{total_s:.1f}s** |",
        "",
        "_Ghi chú: enrichment tốn nhất vì gọi LLM 1 lần/chunk (combined mode); "
        "indexing gồm embedding toàn bộ chunk qua Gemini text-embedding (768-dim) + upsert Qdrant Cloud._",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Latency report saved to {path}")


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    total = time.time() - start
    write_latency_report(total)
    print(f"\nTotal: {total:.1f}s")
