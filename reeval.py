"""
Re-eval nhanh — tái dùng index production CÓ SẴN trên Qdrant (không re-enrich, không re-embed).
Chỉ chạy lại: hybrid search + rerank + answer (prompt mới) + RAGAS.
Dùng để thử cải thiện faithfulness mà không tốn ~14 phút chạy full pipeline.

Chạy: .\.venv\Scripts\python.exe reeval.py
"""
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import COLLECTION_NAME, RERANK_TOP_K, get_qdrant_client
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.pipeline import run_query


def load_existing_chunks(collection: str) -> list[dict]:
    """Scroll toàn bộ điểm đã index → list chunks (để dựng BM25 in-memory)."""
    client = get_qdrant_client()
    chunks, offset = [], None
    while True:
        points, offset = client.scroll(collection, limit=256, offset=offset,
                                       with_payload=True, with_vectors=False)
        for p in points:
            payload = p.payload or {}
            if payload.get("text"):
                chunks.append({"text": payload["text"],
                               "metadata": {k: v for k, v in payload.items() if k != "text"}})
        if offset is None:
            break
    return chunks


def main():
    start = time.time()
    print("=" * 60)
    print("RE-EVAL (tái dùng index production có sẵn)")
    print("=" * 60, flush=True)

    chunks = load_existing_chunks(COLLECTION_NAME)
    print(f"  ✓ Nạp {len(chunks)} chunks từ '{COLLECTION_NAME}'", flush=True)

    # HybridSearch: BM25 dựng lại in-memory; Dense truy vấn collection có sẵn (không re-embed).
    search = HybridSearch()
    search.bm25.index(chunks)
    reranker = CrossEncoderReranker()

    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []
    print(f"\n[Eval] {len(test_set)} câu hỏi (prompt grounded mới)...", flush=True)
    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"]); answers.append(answer)
        all_contexts.append(contexts); ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    print("\n[Eval] RAGAS...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES (re-eval)")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures, path="reports/ragas_report.json")
    print(f"\nTotal: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
