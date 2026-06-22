# Individual Reflection — Lab 18: Production RAG

**Tên:** Huỳnh An Nghiệp · **MSSV:** 2A202600853
**Module phụ trách:** Toàn bộ M1–M5 + pipeline + demo (bài cá nhân)
**Cấu hình thực tế:** LLM `gemini-2.5-flash-lite` · Embedding `gemini-embedding-001` (768-dim) · Reranker = LLM-as-reranker · Qdrant Cloud
**Kết quả:** 37/37 test pass · RAGAS Production: **Faithfulness 0.860** · Answer Relevancy 0.835 · Context Precision 0.804 · Context Recall 0.858 (cả 4 ≥ 0.75, Faithfulness ≥ 0.85 → full bonus)

---

## Phần 1 — Mapping bài giảng → code

| Lecture Concept | Module | Hàm cụ thể | Observation (số liệu thật) |
|----------------|--------|------------|----------------------------|
| Semantic chunking (nhóm câu theo cosine) | M1 | `chunk_semantic()` | Threshold 0.85 + embedding `gemini-embedding-001`; gộp câu cùng chủ đề → ít chunk hơn `chunk_basic` trên cùng văn bản. |
| Hierarchical (parent-child) | M1 | `chunk_hierarchical()` | Parent 2048 / child 256; retrieve child (chính xác) → trả parent (đủ ngữ cảnh). Đây là default cho pipeline production. |
| Structure-aware (markdown header) | M1 | `chunk_structure_aware()` | Cắt theo `^#{1,3}` → giữ nguyên section, có `section` trong metadata, không cắt giữa bảng/list. |
| BM25 tiếng Việt | M2 | `segment_vietnamese()` + `BM25Search` | `underthesea` nối từ ghép bằng `_`; phải `replace("_"," ")` nếu không query "nghỉ phép" (2 token) không khớp "nghỉ_phép" (1 token). |
| Dense retrieval | M2 | `DenseSearch.search()` | Dùng `query_points()` (qdrant-client mới), KHÔNG phải `search()`. Đã thay bge-m3 local bằng embedding API → bỏ ~2.2GB tải. |
| BM25 + Dense fusion (RRF) | M2 | `reciprocal_rank_fusion()` | score = Σ 1/(k+rank+1), k=60. RRF gộp 2 bảng xếp hạng khác thang điểm mà không cần chuẩn hóa score. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Thay CrossEncoder local bằng **LLM-as-reranker** (Gemini chấm 0–10 trong 1 call). Đẩy chunk "nghỉ phép" lên trên "VPN" đúng kỳ vọng. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Faithfulness thấp nhất (0.758) — yếu ở câu **numeric/multi-hop**. Retrieval metrics (precision/recall) tăng so với naive. |
| Diagnostic Tree | M4 | `failure_analysis()` | Map worst_metric → (diagnosis, fix). 3 nhóm lỗi: numeric-generation, ranking-nhiễu, thiếu-chunk. |
| Contextual embeddings / enrichment | M5 | `_enrich_single_call()` | Combined mode: 1 call/chunk sinh summary+questions+context+metadata → tiết kiệm ~75% lời gọi so với 4 hàm riêng. |

---

## Phần 2 — Khó khăn & cách giải quyết (lỗi thật gặp phải)

1. **RAGAS chết toàn bộ với Gemini native**
   - Lỗi (exact): `TypeError: GenerativeServiceClient.generate_content() got an unexpected keyword argument 'temperature'` — **80/80 job fail**, chạy 25 phút ra toàn 0.
   - Debug: đọc traceback → RAGAS truyền `temperature` xuống `langchain-google-genai` nhưng client native không nhận.
   - Fix: chuyển judge sang **OpenAI-compatible endpoint** của Gemini (`langchain_openai.ChatOpenAI` + `OpenAIEmbeddings`, `base_url` = Gemini). Verify 1 câu → 1.0/0.85/1.0/1.0 trong 7s. Sau đó full 20 câu chạy ~90s.

2. **Embedding 404**
   - Lỗi (exact): `NotFoundError: models/text-embedding-004 is not found for API version v1main ... for embedContent`.
   - Debug: gọi `client.models.list()` lọc tên có "embed" → endpoint OpenAI-compat chỉ có `gemini-embedding-001`, `gemini-embedding-2`.
   - Fix: đổi `GEMINI_EMBED_MODEL=gemini-embedding-001`, set `dimensions=768` cho gọn Qdrant.

3. **Qdrant Cloud ReadTimeout**
   - Lỗi (exact): `httpx.ReadTimeout: The read operation timed out` ở `query_points()` (câu thứ 2).
   - Debug: index OK nhưng query timeout → độ trễ mạng tới cluster eu-west-2 vượt timeout mặc định.
   - Fix: `QdrantClient(..., timeout=120)`.

4. **UnicodeEncodeError khi in tiếng Việt**
   - Lỗi (exact): `'charmap' codec can't encode character 'ộ'` (console Windows cp1252).
   - Fix: set `PYTHONIOENCODING=utf-8` + `PYTHONUTF8=1` cho mọi lần chạy.

5. **Test M5 fail — tóm tắt dài hơn bản gốc**
   - Lỗi: `assert 154 <= (65*2)` — Gemini "tóm tắt" câu 65 ký tự thành 154 ký tự.
   - Fix: siết prompt "ngắn hơn đoạn gốc" + guard: nếu `len(summary) > len(text)` → fallback extractive (lấy 1–2 câu đầu).

**Kiến thức thiếu → cách bổ sung:** chưa quen RAGAS dùng LLM nào để chấm; bổ sung bằng cách đọc traceback + tự viết `_build_ragas_judge()` để **inject** llm/embeddings thay vì để RAGAS tự chọn default OpenAI.

---

## Phần 3 — Action Plan cho project cá nhân

### Hiện tại
- RAG pipeline hiện tại: dense-only + paragraph chunking (giống `naive_baseline`), chưa có rerank/enrichment.
- Known issues: câu hỏi numeric/multi-hop bị sai (faithfulness=0), thông tin "phiên bản" (v2023 vs v2024) hay lẫn lộn.

### Plan áp dụng
1. [ ] **Chunking:** dùng **hierarchical** (parent 2048 / child 256) làm mặc định + structure-aware cho tài liệu markdown — giữ trọn điều khoản, recall +0.09 đã chứng minh trong lab.
2. [ ] **Search:** **Hybrid BM25 + Dense + RRF**. BM25 quan trọng với tiếng Việt + số/đơn vị (vd "120 ngày", "200.000.000 VNĐ") mà dense hay bỏ sót.
3. [ ] **Reranking:** có — bắt đầu bằng **LLM-as-reranker** (rẻ, không cần GPU); nâng lên CrossEncoder bge-reranker khi có GPU/độ trễ là vấn đề.
4. [ ] **Evaluation:** **RAGAS 4 metrics** + `failure_analysis()` Diagnostic Tree, chạy lại mỗi lần đổi pipeline để đo Δ thật (không đoán).
5. [ ] **Enrichment:** **combined single-call** (summary + hypothesis questions + contextual prepend + metadata) — tiết kiệm 75% lời gọi; ưu tiên contextual prepend để vá lỗi multi-hop.

### Xử lý điểm yếu faithfulness (numeric/multi-hop)
- Prompt 2 bước: (a) trích nguyên văn điều khoản, (b) trình bày phép tính → mỗi mệnh đề bám 1 câu nguồn.
- Multi-query cho câu multi-hop để lấy đủ tài liệu.

### Timeline
- **Tuần 1:** thay chunking → hierarchical + structure-aware; dựng lại index; đo lại RAGAS baseline mới.
- **Tuần 2:** thêm Hybrid + RRF + LLM-reranker; so sánh Δ precision/recall.
- **Tuần 3:** enrichment combined + prompt 2 bước cho numeric; mục tiêu đẩy faithfulness ≥ 0.85.
- **Tuần 4:** đóng gói demo dashboard (Streamlit) + theo dõi chi phí token/latency.

---

## Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Teamwork | — (bài cá nhân) |
| Problem solving | 5 |
