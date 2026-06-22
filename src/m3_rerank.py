from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K, RERANKER_BACKEND, LLM_CHAT_MODEL, get_llm_client


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    """Reranker với 2 backend (chọn qua config.RERANKER_BACKEND):

    - "api"   : LLM-as-reranker — Gemini chấm điểm liên quan query↔doc trong 1 API call.
                Không cần tải model nặng (~2.2GB bge-reranker).
    - "local" : CrossEncoder bge-reranker-v2-m3 qua sentence_transformers.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3",
                 backend: str | None = None):
        self.model_name = model_name
        self.backend = backend or RERANKER_BACKEND
        self._model = None

    def _load_model(self):
        if self.backend == "api":
            return None   # API backend không cần load model cục bộ
        if self._model is None:
            # Dùng sentence_transformers.CrossEncoder, KHÔNG dùng FlagEmbedding
            # (FlagReranker crash với transformers>=5.0).
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def _llm_scores(self, query: str, documents: list[dict]) -> list[float]:
        """Gọi Gemini 1 lần để chấm điểm 0-10 độ liên quan cho từng doc."""
        client = get_llm_client()
        numbered = "\n".join(f"[{i}] {d['text']}" for i, d in enumerate(documents))
        prompt = (
            f"Câu hỏi: {query}\n\nCác đoạn văn:\n{numbered}\n\n"
            "Chấm điểm độ liên quan của TỪNG đoạn với câu hỏi, thang 0-10 "
            "(10 = trả lời trực tiếp). Chỉ trả về JSON dạng "
            '{"scores": [đoạn0, đoạn1, ...]} đúng thứ tự, không giải thích.'
        )
        resp = client.chat.completions.create(
            model=LLM_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        import json, re
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        scores = json.loads(m.group(0) if m else raw)["scores"]
        # Đảm bảo đủ độ dài.
        scores = [float(s) for s in scores][:len(documents)]
        scores += [0.0] * (len(documents) - len(scores))
        return scores

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        if self.backend == "api":
            try:
                scores = self._llm_scores(query, documents)
            except Exception as e:
                print(f"  ⚠️  LLM rerank failed ({e}); giữ nguyên thứ tự gốc.")
                scores = [doc.get("score", 0.0) for doc in documents]
        else:
            model = self._load_model()
            pairs = [(query, doc["text"]) for doc in documents]
            scores = model.predict(pairs)
            if isinstance(scores, (int, float)):
                scores = [scores]

        scored = sorted(zip(scores, documents), key=lambda x: float(x[0]), reverse=True)
        return [
            RerankResult(
                text=doc["text"],
                original_score=doc.get("score", 0.0),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        # TODO (optional): from flashrank import Ranker, RerankRequest
        # model = Ranker(); passages = [{"text": d["text"]} for d in documents]
        # results = model.rerank(RerankRequest(query=query, passages=passages))
        return []


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
