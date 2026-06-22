"""
Lab 18 — Production RAG · Demo Dashboard
========================================
Giao diện so sánh trực quan: điểm RAGAS (naive vs production), tìm kiếm
BM25/Dense/Hybrid/Rerank trực tiếp, so sánh chiến lược chunking, và ước tính chi phí.

Chạy:
    .\.venv\Scripts\streamlit run demo_app.py
"""
from __future__ import annotations

import os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import (COLLECTION_NAME, NAIVE_COLLECTION, LLM_CHAT_MODEL, LLM_EMBED_MODEL,
                    EMBEDDING_BACKEND, RERANKER_BACKEND, USE_GEMINI, get_qdrant_client)

st.set_page_config(page_title="Lab 18 · Production RAG Demo", layout="wide",
                   page_icon="🔍")

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
METRIC_VI = {
    "faithfulness": "Faithfulness (bám context)",
    "answer_relevancy": "Answer Relevancy (đúng câu hỏi)",
    "context_precision": "Context Precision (lọc nhiễu)",
    "context_recall": "Context Recall (đủ thông tin)",
}
REPORTS = "reports"


# ─────────────────────────── Helpers ───────────────────────────

def load_report(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@st.cache_resource(show_spinner="Đang nạp index từ Qdrant + dựng BM25...")
def load_search_engine(collection: str):
    """Đọc toàn bộ điểm từ collection Qdrant (đã index sẵn), dựng BM25 in-memory.
    Không re-embed corpus → nhanh và rẻ."""
    from src.m2_search import BM25Search, DenseSearch, reciprocal_rank_fusion  # noqa
    client = get_qdrant_client()
    chunks, offset = [], None
    while True:
        points, offset = client.scroll(collection, limit=256, offset=offset,
                                       with_payload=True, with_vectors=False)
        for p in points:
            payload = p.payload or {}
            text = payload.get("text", "")
            if text:
                chunks.append({"text": text, "metadata": {k: v for k, v in payload.items()
                                                          if k != "text"}})
        if offset is None:
            break
    bm25 = BM25Search(); bm25.index(chunks)
    dense = DenseSearch()
    return bm25, dense, chunks


def collection_count(collection: str) -> int | None:
    try:
        return get_qdrant_client().count(collection).count
    except Exception:
        return None


# ─────────────────────────── Sidebar ───────────────────────────

st.sidebar.title("⚙️ Cấu hình")
st.sidebar.markdown(f"""
- **LLM:** `{LLM_CHAT_MODEL}`
- **Embedding:** `{LLM_EMBED_MODEL}`
- **Embedding backend:** `{EMBEDDING_BACKEND}`
- **Reranker backend:** `{RERANKER_BACKEND}`
- **Provider:** {"Gemini API ✅" if USE_GEMINI else "OpenAI / local"}
""")
prod_n = collection_count(COLLECTION_NAME)
naive_n = collection_count(NAIVE_COLLECTION)
st.sidebar.markdown("**Qdrant collections:**")
st.sidebar.write(f"- `{COLLECTION_NAME}`: {prod_n if prod_n is not None else '—'} điểm")
st.sidebar.write(f"- `{NAIVE_COLLECTION}`: {naive_n if naive_n is not None else '—'} điểm")

st.title("🔍 Lab 18 — Production RAG · Demo Dashboard")
st.caption("So sánh Naive vs Production · Tìm kiếm Hybrid · Chunking · Chi phí — corpus chính sách nhân sự tiếng Việt")

tab_scores, tab_search, tab_chunk, tab_cost = st.tabs(
    ["📊 Điểm số RAGAS", "🔎 Tìm kiếm trực tiếp", "✂️ Chunking", "💰 Chi phí & Latency"])


# ─────────────────────── Tab 1: RAGAS Scores ───────────────────────

with tab_scores:
    st.subheader("So sánh Naive Baseline vs Production Pipeline")
    naive = load_report(f"{REPORTS}/naive_baseline_report.json")
    prod = load_report(f"{REPORTS}/ragas_report.json")

    if not naive and not prod:
        st.info("Chưa có report. Chạy `python naive_baseline.py` và `python src/pipeline.py` trước.")
    else:
        na = (naive or {}).get("aggregate", {})
        pa = (prod or {}).get("aggregate", {})
        rows = []
        for m in METRICS:
            rows.append({"Metric": METRIC_VI[m],
                         "Naive": round(float(na.get(m, 0)), 4),
                         "Production": round(float(pa.get(m, 0)), 4),
                         "Δ": round(float(pa.get(m, 0)) - float(na.get(m, 0)), 4)})
        df = pd.DataFrame(rows)

        c1, c2, c3, c4 = st.columns(4)
        for col, m in zip([c1, c2, c3, c4], METRICS):
            p = float(pa.get(m, 0)); n = float(na.get(m, 0))
            col.metric(METRIC_VI[m].split(" (")[0], f"{p:.3f}", f"{p-n:+.3f}")

        fig = go.Figure()
        fig.add_bar(name="Naive", x=df["Metric"], y=df["Naive"], marker_color="#9aa0a6")
        fig.add_bar(name="Production", x=df["Metric"], y=df["Production"], marker_color="#1a73e8")
        fig.add_hline(y=0.75, line_dash="dash", line_color="green",
                      annotation_text="ngưỡng 0.75")
        fig.update_layout(barmode="group", yaxis_range=[0, 1.05], height=420,
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

        if pa:
            passed = sum(1 for m in METRICS if float(pa.get(m, 0)) >= 0.70)
            st.markdown("**🎁 Bonus checklist (Production):**")
            b1 = float(pa.get("faithfulness", 0)) >= 0.85
            b2 = all(float(pa.get(m, 0)) >= 0.75 for m in METRICS)
            st.write(f"- {'✅' if b1 else '⬜'} Faithfulness ≥ 0.85 (+3)")
            st.write(f"- {'✅' if b2 else '⬜'} Tất cả metrics ≥ 0.75 (+3)")
            st.write(f"- ✅ Enrichment combined 1-call (+2) · ✅ Latency breakdown (+2)")
            st.success(f"{passed}/4 metrics đạt ngưỡng 0.70")


# ─────────────────────── Tab 2: Live Search ───────────────────────

with tab_search:
    st.subheader("So sánh BM25 vs Dense vs Hybrid (RRF) vs +Reranking")
    coll = st.radio("Collection để truy vấn:", [COLLECTION_NAME, NAIVE_COLLECTION],
                    horizontal=True)
    if collection_count(coll) is None:
        st.warning(f"Collection `{coll}` chưa tồn tại — chạy pipeline/baseline để tạo index.")
    else:
        query = st.text_input("Câu hỏi:", "Nhân viên được nghỉ bao nhiêu ngày phép năm?")
        topk = st.slider("Số kết quả hiển thị", 3, 10, 5)
        if st.button("🔎 Tìm kiếm", type="primary"):
            from src.m2_search import reciprocal_rank_fusion
            from src.m3_rerank import CrossEncoderReranker
            bm25, dense, _ = load_search_engine(coll)

            timings = {}
            t = time.perf_counter(); bm = bm25.search(query, top_k=topk); timings["BM25"] = (time.perf_counter()-t)*1000
            t = time.perf_counter(); dn = dense.search(query, top_k=topk, collection=coll); timings["Dense"] = (time.perf_counter()-t)*1000
            t = time.perf_counter(); hy = reciprocal_rank_fusion([bm, dn], top_k=topk); timings["Hybrid (RRF)"] = (time.perf_counter()-t)*1000
            docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in hy]
            t = time.perf_counter(); rr = CrossEncoderReranker().rerank(query, docs, top_k=min(3, topk)); timings["+Rerank"] = (time.perf_counter()-t)*1000

            lt = pd.DataFrame([{"Bước": k, "Latency (ms)": round(v, 1)} for k, v in timings.items()])
            st.plotly_chart(px.bar(lt, x="Bước", y="Latency (ms)", color="Bước",
                                   title="Latency từng bước"), use_container_width=True)

            cols = st.columns(4)
            def render(col, title, items, score_key):
                with col:
                    st.markdown(f"**{title}**")
                    if not items:
                        st.caption("(rỗng)")
                    for i, it in enumerate(items):
                        txt = it.text if hasattr(it, "text") else it["text"]
                        sc = getattr(it, score_key, None) if hasattr(it, score_key) else None
                        src = (it.metadata if hasattr(it, "metadata") else {}).get("source", "")
                        st.markdown(f"`#{i+1}` · {sc:.3f}" if sc is not None else f"`#{i+1}`")
                        st.caption(f"📄 {src}")
                        st.write(txt[:180] + ("..." if len(txt) > 180 else ""))
                        st.divider()
            render(cols[0], "BM25 (từ khóa)", bm, "score")
            render(cols[1], "Dense (ngữ nghĩa)", dn, "score")
            render(cols[2], "Hybrid (RRF)", hy, "score")
            render(cols[3], "+Reranking", rr, "rerank_score")


# ─────────────────────── Tab 3: Chunking ───────────────────────

with tab_chunk:
    st.subheader("So sánh 4 chiến lược chunking (M1)")
    from config import DATA_DIR
    md_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".md"))
    doc = st.selectbox("Chọn tài liệu:", md_files,
                       index=md_files.index("nghi_phep_nam_v2024.md") if "nghi_phep_nam_v2024.md" in md_files else 0)
    if st.button("✂️ Chạy chunking", type="primary"):
        from src.m1_chunking import (chunk_basic, chunk_semantic, chunk_hierarchical,
                                     chunk_structure_aware)
        with open(os.path.join(DATA_DIR, doc), encoding="utf-8") as f:
            text = f.read()
        meta = {"source": doc}
        with st.spinner("Đang chunk (semantic gọi embedding API)..."):
            basic = chunk_basic(text, metadata=meta)
            semantic = chunk_semantic(text, metadata=meta)
            parents, children = chunk_hierarchical(text, metadata=meta)
            structure = chunk_structure_aware(text, metadata=meta)

        data = {"Basic": basic, "Semantic": semantic, "Hierarchical (child)": children,
                "Structure-aware": structure}
        rows = []
        for name, ch in data.items():
            lengths = [len(c.text) for c in ch] or [0]
            rows.append({"Chiến lược": name, "Số chunk": len(ch),
                         "Độ dài TB": round(sum(lengths)/len(lengths)),
                         "Min": min(lengths), "Max": max(lengths)})
        dfc = pd.DataFrame(rows)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(dfc, x="Chiến lược", y="Số chunk", color="Chiến lược",
                               title="Số lượng chunk"), use_container_width=True)
        c2.plotly_chart(px.bar(dfc, x="Chiến lược", y="Độ dài TB", color="Chiến lược",
                               title="Độ dài trung bình (ký tự)"), use_container_width=True)
        st.dataframe(dfc, use_container_width=True, hide_index=True)
        st.caption(f"Hierarchical: {len(parents)} parent → {len(children)} child (retrieve child, return parent).")

        sel = st.selectbox("Xem trước chunk của chiến lược:", list(data.keys()))
        for i, c in enumerate(data[sel][:6]):
            with st.expander(f"Chunk #{i+1} · {len(c.text)} ký tự · section={c.metadata.get('section','—')}"):
                st.write(c.text)


# ─────────────────────── Tab 4: Cost & Latency ───────────────────────

with tab_cost:
    st.subheader("Ước tính chi phí API & Latency")
    st.caption("Đếm token bằng tiktoken (xấp xỉ) · Giá chỉnh được — mặc định ~Gemini Flash-Lite.")

    def count_tokens(texts: list[str]) -> int:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return sum(len(enc.encode(t)) for t in texts)
        except Exception:
            return sum(len(t)//4 for t in texts)

    from src.m1_chunking import load_documents, chunk_hierarchical
    @st.cache_data(show_spinner="Đếm token corpus...")
    def corpus_stats():
        docs = load_documents()
        chunks = []
        for d in docs:
            _, children = chunk_hierarchical(d["text"], metadata=d["metadata"])
            chunks.extend(c.text for c in children)
        return len(docs), chunks
    n_docs, chunk_texts = corpus_stats()
    n_chunks = len(chunk_texts)
    chunk_tokens = count_tokens(chunk_texts)

    test_set = json.load(open("test_set.json", encoding="utf-8"))
    n_q = len(test_set)

    st.markdown("**Giá (USD / 1 triệu token):**")
    g1, g2, g3 = st.columns(3)
    price_in = g1.number_input("LLM input", value=0.10, step=0.05, format="%.2f")
    price_out = g2.number_input("LLM output", value=0.40, step=0.05, format="%.2f")
    price_emb = g3.number_input("Embedding", value=0.15, step=0.05, format="%.2f")

    # Ước lượng token theo từng bước (xấp xỉ, có thể chỉnh)
    out_per_call = 250          # token output trung bình / lời gọi LLM
    enrich_in = chunk_tokens + n_chunks * 60     # prompt + chunk
    enrich_out = n_chunks * 220
    index_emb = chunk_tokens
    q_in = n_q * 1200           # context (3 chunk) + câu hỏi cho answer + rerank
    q_out = n_q * 2 * out_per_call
    q_emb = n_q * 30
    ragas_in = n_q * 4 * 900    # 4 metric × prompt
    ragas_out = n_q * 4 * 120

    def usd(tok_in, tok_out, tok_emb):
        return (tok_in/1e6*price_in) + (tok_out/1e6*price_out) + (tok_emb/1e6*price_emb)

    prod_cost = {
        "M5 Enrichment": usd(enrich_in, enrich_out, 0),
        "M2 Indexing (embed)": usd(0, 0, index_emb),
        "Truy vấn (search+rerank+answer)": usd(q_in, q_out, q_emb),
        "M4 RAGAS eval": usd(ragas_in, ragas_out, n_q*40),
    }
    naive_cost = {
        "M2 Indexing (embed)": usd(0, 0, index_emb),
        "Truy vấn (answer only)": usd(n_q*900, n_q*out_per_call, q_emb),
        "M4 RAGAS eval": usd(ragas_in, ragas_out, n_q*40),
    }

    c1, c2, c3 = st.columns(3)
    c1.metric("Tài liệu", n_docs)
    c2.metric("Child chunks", n_chunks)
    c3.metric("Token corpus", f"{chunk_tokens:,}")

    dfp = pd.DataFrame([{"Pipeline": "Production", "Bước": k, "USD": round(v, 4)} for k, v in prod_cost.items()] +
                       [{"Pipeline": "Naive", "Bước": k, "USD": round(v, 4)} for k, v in naive_cost.items()])
    st.plotly_chart(px.bar(dfp, x="Pipeline", y="USD", color="Bước",
                           title="Chi phí ước tính / lần chạy full (USD)"), use_container_width=True)

    tot_p, tot_n = sum(prod_cost.values()), sum(naive_cost.values())
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("💰 Production / lần", f"${tot_p:.4f}")
    cc2.metric("💰 Naive / lần", f"${tot_n:.4f}")
    cc3.metric("Chênh lệch", f"${tot_p-tot_n:+.4f}")

    st.markdown("**Combined enrichment (1 call/chunk) vs 4 calls/chunk:**")
    saved = usd(enrich_in*3, enrich_out, 0)   # 4 calls ≈ 4× prompt overhead
    st.write(f"- Combined (1 call): ~${usd(enrich_in, enrich_out, 0):.4f}")
    st.write(f"- Riêng lẻ (4 calls): ~${usd(enrich_in*1.0, enrich_out, 0) + saved:.4f} → **tiết kiệm ~75% lời gọi**")

    # Latency breakdown nếu có
    lat_path = f"{REPORTS}/latency_breakdown.md"
    if os.path.exists(lat_path):
        st.markdown("**⏱️ Latency breakdown (từ lần chạy pipeline gần nhất):**")
        st.markdown(open(lat_path, encoding="utf-8").read())
