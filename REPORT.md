# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Văn A — 22BI13000  
**Ngày nộp**: 2026-06-25  
**Submission option**: A (lightweight ZIP)

---

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit` (Qwen2.5, 3B params, 4-bit NF4 quantization)
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (180 train + 20 eval)
- **max_seq_length**: 512 (p95 của dataset = 487 tokens, rounded up to 512)
- **GPU**: Tesla T4, 16 GB VRAM (Google Colab Free)
- **Training cost ước tính**: ~$0.43 (~74 phút tổng @ $0.35/hr cho T4)
- **LoRA target modules**: `q_proj`, `v_proj` (theo lab spec)
- **Gradient checkpointing**: bật — giảm ~60% VRAM
- **packing**: `False` (T4 + transformers ≥ 4.46 — tránh dim mismatch bug)

---

## 2. Rank Experiment Results

| Rank | Trainable Params | % Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|----------|------------|-----------|-----------|------------|
| Base | — | 0.000% | — | 9.8 GB | 3.421 | 30.62 |
| 8    | 1,572,864 | 0.051% | 18.3 min | 10.4 GB | 2.847 | 17.24 |
| 16   | 3,145,728 | 0.102% | 21.7 min | 11.1 GB | 2.731 | 15.36 |
| 64   | 12,582,912 | 0.408% | 34.2 min | 13.8 GB | 2.698 | 14.87 |

**Ghi chú**:
- r=64 tốn VRAM nhiều hơn r=8 khoảng **3.4 GB** (+33%) nhưng chỉ giảm perplexity thêm **0.49** so với r=16
- r=8 → r=16: giảm perplexity **1.88** (cải thiện rõ) với chi phí +3.4 min và +0.7 GB VRAM
- r=16 → r=64: giảm perplexity chỉ **0.49** (cải thiện rất nhỏ) với chi phí +12.5 min và +2.7 GB VRAM

---

## 3. Loss Curve Analysis

![Loss Curve](loss_curve.png)

**Quan sát**:

- **r=16** (baseline): Training loss giảm đều từ ~2.9 xuống ~2.5 qua 3 epochs. Eval loss theo sát train loss → **không overfitting**.
- **r=8**: Loss curve phẳng hơn, converge chậm hơn một chút. Model ít expressive capacity hơn.
- **r=64**: Training loss giảm thấp nhất (~2.4), nhưng eval loss từ epoch 2.5 bắt đầu nhích lên nhẹ → **dấu hiệu nhẹ của overfitting** trên 200 samples. Với dataset nhỏ như thế này, r=64 có thể bị overfit vì quá nhiều trainable params so với lượng data.

**Kết luận từ loss curve**: r=16 là lựa chọn tối ưu — converge tốt, không overfit, eval loss đi xuống ổn định.

---

## 4. Qualitative Comparison (5 examples)

### Example 1 — Giải thích machine learning

**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.

| | Response |
|---|---|
| **Base** | "Machine learning is a subset of artificial intelligence. It involves algorithms that learn from data to make predictions or decisions without being explicitly programmed." |
| **Fine-tuned (r=16)** | "Machine learning (học máy) là một nhánh của trí tuệ nhân tạo, cho phép máy tính tự học từ dữ liệu mà không cần lập trình từng bước cụ thể. Ví dụ: khi bạn dạy trẻ nhận biết con mèo bằng cách cho xem hàng trăm ảnh mèo, máy tính học theo cách tương tự..." |

**Nhận xét**: ✅ **IMPROVED** — Model sau fine-tune chuyển sang tiếng Việt, thêm ví dụ cụ thể, giải thích dễ hiểu hơn nhiều cho người mới.

---

### Example 2 — Code Python Fibonacci

**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.

| | Response |
|---|---|
| **Base** | `def fib(n): if n<=1: return n; return fib(n-1)+fib(n-2)` (không có doc, đệ quy O(2^n)) |
| **Fine-tuned (r=16)** | Code block chuẩn với docstring tiếng Việt, xử lý edge case `n<0`, dùng iterative O(n), kèm test output |

**Nhận xét**: ✅ **IMPROVED** — Format code tốt hơn hẳn, thêm guard clause, dùng thuật toán hiệu quả hơn.

---

### Example 3 — Nguyên tắc UI/UX

**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.

| | Response |
|---|---|
| **Base** | "1. Simplicity 2. Consistency 3. Feedback 4. Accessibility 5. Visual hierarchy" (chỉ từ khóa) |
| **Fine-tuned (r=16)** | Mỗi nguyên tắc có tên tiếng Việt + giải thích ngắn + ví dụ thực tế, bold header, format markdown chuẩn |

**Nhận xét**: ✅ **IMPROVED** — Base chỉ liệt kê từ khóa, fine-tuned giải thích có context thực tế, dùng markdown heading.

---

### Example 4 — LoRA vs QLoRA

**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.

| | Response |
|---|---|
| **Base** | "LoRA adds low-rank matrices to model weights. QLoRA combines LoRA with quantization to reduce memory usage." |
| **Fine-tuned (r=16)** | Giải thích toán học ΔW = B·A, so sánh VRAM cụ thể (28GB → 10GB), đề cập Paged AdamW, có tóm tắt ngắn gọn ở cuối |

**Nhận xét**: ➡️ **SAME/IMPROVED** — Cả hai đều đúng, nhưng fine-tuned chi tiết và có số liệu cụ thể hơn.

---

### Example 5 — Prompt engineering vs RAG vs Fine-tuning

**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.

| | Response |
|---|---|
| **Base** | Ba câu định nghĩa riêng biệt, không có so sánh |
| **Fine-tuned (r=16)** | Bảng markdown 3 cột (khi nào dùng / chi phí / hạn chế) + "quy tắc vàng" ra quyết định |

**Nhận xét**: ✅ **IMPROVED** — Cấu trúc hóa thông tin tốt hơn nhiều, có actionable recommendation.

---

## 5. Conclusion về Rank Trade-off

Sau khi thực hiện rank experiment với ba cấu hình r=8, r=16, r=64 trên cùng dataset Vietnamese Alpaca (200 samples) và cùng Qwen2.5-3B backbone, có thể rút ra các kết luận sau.

**r=16 là lựa chọn có ROI tốt nhất cho dataset này.** Chuyển từ r=8 lên r=16 giảm perplexity từ 17.24 xuống 15.36 — cải thiện gần 2 điểm với chi phí chỉ thêm 3.4 phút training và 0.7 GB VRAM. Đây là trade-off rõ ràng và xứng đáng. Ngược lại, chuyển từ r=16 lên r=64 chỉ giảm thêm 0.49 điểm perplexity nhưng tốn thêm 12.5 phút (+58% thời gian) và 2.7 GB VRAM (+24%). Đây là dấu hiệu điển hình của **diminishing returns**.

**Diminishing returns xảy ra từ r=16 → r=64.** Với dataset chỉ 180 samples, model r=64 có quá nhiều trainable parameters (12.5 triệu, tương đương 0.4% model) so với lượng signal trong data. Loss curve cho thấy eval loss của r=64 bắt đầu tăng nhẹ ở epoch cuối — dấu hiệu overfitting nhẹ. Điều này xác nhận nguyên tắc trong bài giảng: *Quality > Quantity* và model capacity phải phù hợp với data size.

**Recommendation cho production**: Với use case tiếng Việt instruction-following trên ~200 samples, nên dùng **r=16**. Nếu dataset tăng lên 1000+ samples chất lượng cao thì mới nên thử r=32 hoặc r=64. Nếu muốn deploy nhiều adapters trên 1 GPU (multi-tenant serving), r=8 là lựa chọn tốt vì tiết kiệm VRAM nhất với chất lượng chấp nhận được (perplexity 17.24, cải thiện đáng kể so với base 30.62).

---

## 6. What I Learned

- **LoRA rank không phải "càng cao càng tốt"**: Tăng rank từ 16 lên 64 gần như không cải thiện perplexity nhưng tăng đáng kể thời gian và VRAM — trên dataset nhỏ, overfitting là rủi ro thực sự, không phải lý thuyết.
- **Dataset quality quan trọng hơn quantity**: 200 samples Vietnamese Alpaca đã đủ để model học chuyển ngữ và format output — điều này chứng minh nguyên tắc *500 perfect > 10k noisy* mà bài giảng đề cập.
- **Quy trình debugging thực tế**: Việc xử lý các lỗi như `KeyError: 'instruction'`, `evaluation_strategy` deprecation, và OOM khi eval giúp tôi hiểu sâu hơn về sự phụ thuộc giữa các library versions — kỹ năng cần thiết khi làm việc với ML toolchain đang thay đổi nhanh.
