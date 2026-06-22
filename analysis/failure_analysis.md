# Failure Analysis — Lab 18: Production RAG

**Sinh viên:** Huỳnh An Nghiệp · **MSSV:** 2A202600853 (bài cá nhân)
**Cấu hình:** LLM `gemini-2.5-flash-lite` (answer ở temperature 0) · Embedding `gemini-embedding-001` (768-dim) · Reranker = LLM-as-reranker (Gemini) · Vector DB = Qdrant Cloud
**Nguồn dữ liệu:** `reports/ragas_report.json` (Production) + `reports/naive_baseline_report.json` (Naive) · 20 câu hỏi.

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.810 | **0.860** | **+0.050** |
| Answer Relevancy | 0.903 | 0.835 | −0.068 |
| Context Precision | 0.800 | 0.804 | +0.004 |
| Context Recall | 0.800 | 0.858 | +0.058 |

**Cả 4 metric ≥ 0.75; Faithfulness ≥ 0.85** (đạt full bonus RAGAS).

**Đọc kết quả:** Hybrid Search (BM25+Dense+RRF) + Reranking + Enrichment cải thiện truy hồi (recall +0.058). Faithfulness tăng mạnh nhờ **prompt grounded**: bắt LLM trích nguyên văn con số từ context và **viết rõ phép tính** (vd `85% × 20.000.000 = 17.000.000`) ⇒ mỗi mệnh đề trở nên *suy ra được* từ context. Answer_relevancy giảm nhẹ vì prompt grounded ưu tiên bám sát nguồn hơn là diễn giải trơn tru.

---

## Diagnostic / Error Tree (khung chẩn đoán dùng cho mọi failure)

```text
Câu trả lời KÉM
│
├─ faithfulness thấp?  → câu trả lời chứa mệnh đề KHÔNG suy ra được từ context
│      → LLM hallucinate hoặc tự suy diễn/tính toán không show nguồn
│      → FIX: prompt "chỉ dùng context + show phép tính", temperature 0
│
├─ context_precision thấp?  → chunk đúng CÓ trong context nhưng bị chunk nhiễu xếp trên
│      → retrieval kéo về rác / thứ hạng sai
│      → FIX: reranking mạnh hơn, giảm top_k, metadata filter
│
├─ context_recall thấp?  → context THIẾU chunk chứa đáp án
│      → chunking cắt mất ý / hybrid bỏ sót / câu hỏi đa phần
│      → FIX: cải thiện chunking, tăng trọng số BM25, multi-query
│
└─ answer_relevancy thấp?  → câu trả lời lạc đề / trả lời nhầm phiên bản
       → FIX: cải thiện prompt template, tách câu hỏi đa phần, ưu tiên bản hiện hành
```

---

## Bottom-5 Failures (sắp theo avg_score tăng dần)

### #1 — "Thông tin lương thuộc cấp độ phân loại dữ liệu nào?"  (avg 0.534)
- **Expected:** Lương = dữ liệu **Bí mật (cấp 3)**; cấm chia sẻ; mã hóa khi truyền, hạn chế theo need-to-know.
- **Scores:** faithfulness **0.0** · answer_relevancy 0.635 · context_precision 1.0 · context_recall 0.5
- **Error Tree:** Output sai → precision=1.0 (chunk đúng có trong context) → faithfulness=0 → câu trả lời **ghép 2 tài liệu** (`ky_luong` + `phan_loai_du_lieu`) mà mỗi chunk chỉ nói một nửa.
- **Root cause:** Câu **multi-hop**; recall=0.5 → chỉ lấy được 1/2 chunk cần thiết nên kết luận "cấp 3" không có nguồn trực tiếp.
- **Suggested fix:** multi-query retrieval; prompt yêu cầu trích từng tài liệu nguồn trước khi kết luận.

### #2 — "Laptop 30 triệu cho nhân viên mới: ai phê duyệt và cần gì từ CNTT?"  (avg 0.570)
- **Expected:** 5–50tr ⇒ **Giám đốc (Director)** duyệt; cần **xác nhận cấu hình từ CNTT**; ≥10tr cần **3 báo giá**.
- **Scores:** faithfulness 1.0 · answer_relevancy 0.946 · context_precision **0.0** · context_recall 0.333
- **Error Tree:** faithfulness=1.0 (bám context) nhưng precision=0.0 & recall=0.333 → **lỗi retrieval**: kéo về chunk nhiễu, thiếu chunk "xác nhận cấu hình CNTT" và "3 báo giá".
- **Root cause:** Câu **multi-hop 3 điều kiện** (ngưỡng duyệt + CNTT + báo giá) nằm rải ở 2–3 tài liệu mua sắm.
- **Suggested fix:** tăng `HYBRID_TOP_K`, metadata filter category=procurement, multi-query theo từng điều kiện.

### #3 — "Bao lâu phải đổi mật khẩu một lần?"  (avg 0.625)
- **Expected:** **120 ngày** (chính sách v2.0 hiện hành; v1.0 cũ là 90 ngày, đã thay thế).
- **Scores:** faithfulness 1.0 · answer_relevancy **0.0** · context_precision 0.5 · context_recall 1.0
- **Error Tree:** Context đủ (recall=1.0) nhưng answer_relevancy=0 → câu trả lời **lạc trọng tâm**: nhiều khả năng nêu cả v1.0 lẫn v2.0 hoặc nhầm bản → judge coi không trả lời thẳng câu hỏi.
- **Root cause:** **Version conflict** (`mat_khau_v1` vs `v2`) — cả 2 chunk cùng được lấy, precision 0.5.
- **Suggested fix:** prompt "chỉ trả lời theo bản HIỆN HÀNH"; metadata `version` + ưu tiên bản mới; câu trả lời ngắn đúng trọng tâm.

### #4 — "Tài trợ khóa học 25 triệu, nghỉ sau 8 tháng. Hoàn trả bao nhiêu?"  (avg 0.664)
- **Expected:** Cam kết tối thiểu 1 năm; nghỉ trước hạn ⇒ hoàn trả **100% = 25.000.000 VNĐ**.
- **Scores:** faithfulness **0.0** · answer_relevancy 0.654 · context_precision 1.0 · context_recall 1.0
- **Error Tree:** Context hoàn hảo (precision=recall=1.0) → faithfulness=0 → lỗi **bước sinh**: suy luận "8 tháng < 12 tháng ⇒ 100%" — bước logic không có nguyên văn trong chunk.
- **Root cause:** **Numeric/logic reasoning**; dù prompt grounded vẫn còn 1 vài câu suy luận điều kiện chưa show đủ.
- **Suggested fix:** prompt few-shot cho dạng "điều kiện → kết luận"; bắt liệt kê điều khoản trước khi suy luận.

### #5 — "Senior 9 năm thâm niên: bao nhiêu ngày phép và lương khoảng nào?"  (avg 0.724)
- **Expected:** 15 + 3 (9÷3) = **18 ngày**; lương Senior 20–35tr.
- **Scores:** faithfulness 1.0 · answer_relevancy 0.897 · context_precision **0.0** · context_recall 1.0
- **Error Tree:** faithfulness=1.0 nhưng precision=0.0 (recall=1.0) → chunk đúng *có* nhưng **chunk nhiễu xếp đầu** → lỗi **retrieval ranking**.
- **Root cause:** Câu **multi-hop** kéo cả chunk "phép năm" lẫn "bảng lương"; reranker xếp chunk lương cao hơn chunk thâm niên.
- **Suggested fix:** rerank theo từng khía cạnh (aspect-based); tách truy vấn "ngày phép" và "lương".

---

## Phân nhóm lỗi (insight tổng hợp)

| Pattern | Triệu chứng | Câu hỏi | Tầng lỗi |
|---------|-------------|---------|----------|
| **A. Numeric / multi-hop suy diễn** | faithfulness thấp dù context tốt | #1, #4 | Generation |
| **B. Ranking nhiễu** | context_precision thấp, recall cao | #2, #5, BHYT PVI, malware | Retrieval (rerank) |
| **C. Version conflict / lạc đề** | answer_relevancy thấp | #3 (mật khẩu) | Generation + metadata |
| **D. Thiếu chunk (đa phần)** | context_recall=0.5 | MFA, nghỉ không lương 20 ngày | Retrieval (chunking/hybrid) |

**Kết luận:** Prompt grounded (show phép tính + temperature 0) đẩy **Faithfulness 0.758 → 0.860**, vá phần lớn Pattern A. Phần còn lại là retrieval-ranking (B) và version-conflict (C) — hướng tiếp theo là metadata filter + multi-query.

---

## Case Study (cho presentation)

**Question:** "Bao lâu phải đổi mật khẩu một lần?" (#3 — version conflict)

**Error Tree walkthrough:**
1. Output đúng? → **Một phần** (answer_relevancy 0.0 — không trả lời thẳng "120 ngày")
2. Context đúng? → **Có nhưng lẫn** (recall 1.0, precision 0.5 — lấy cả `mat_khau_v1` 90 ngày lẫn `v2` 120 ngày)
3. Query rewrite OK? → OK, nhưng không phân biệt phiên bản
4. Fix ở bước: **Retrieval metadata + Generation** — gắn `version` vào metadata, prompt "chỉ trả lời bản hiện hành".

**Nếu có thêm 1 giờ, sẽ optimize:**
- Metadata `version` + filter ưu tiên bản mới ⇒ dứt điểm các câu version-conflict (mật khẩu, phép năm, thâm niên).
- Aspect-based rerank + multi-query cho câu multi-hop (#2, #5) ⇒ kéo precision lên.
- Few-shot "điều kiện → kết luận" cho câu numeric (#1, #4) ⇒ đẩy faithfulness từ 0.86 lên cao hơn.
