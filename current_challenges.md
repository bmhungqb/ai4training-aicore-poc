# Báo Cáo Kết Quả Stage 1 (Công Đoạn 1) & Thách Thức Khi Triển Khai Stage 2

Tài liệu này tổng hợp:
1. **Kết quả cải tiến kỹ thuật và dữ liệu định lượng của Stage 1 (Kinematic Action Segmentation)** trên cả 3 video của Công đoạn 1 (**CĐ 1: Diễu TP 4 cạnh nắp túi**).
2. **Các thách thức thực tế (Current Challenges)** khi kết nối các ranh giới vi-thao tác này vào Stage 2 (VLM Classification & Macro Evaluation).

---

## PHẦN 1: KẾT QUẢ ĐẠT ĐƯỢC CỦA STAGE 1 (CÔNG ĐOẠN 1)

### 1.1. Các cải tiến thuật toán cốt lõi đã áp dụng
Trong file mã nguồn [`src/kinematic_pipeline/calculate_direction_magnitude.py`](file:///Users/admin/Documents/work/WE/poc-ai4training/ai4training-aicore-poc/src/kinematic_pipeline/calculate_direction_magnitude.py), 4 nút thắt vật lý đã được giải quyết:

1. **Co viền Mask (`cv2.erode` kernel $3 \times 3$):**
   - Triệt tiêu hoàn toàn hiện tượng 1–3 pixel rìa mặt nạ SAM3 lấn ra nền mặt bàn máy may, loại bỏ gradient dòng chảy quang học giả (nguyên nhân gây nhiễu `turbulence` ảo).
2. **Bảo toàn cử động ngón tay khi tì cổ tay (Translation + RMS Energy):**
   - Thay thế việc tính `np.median(hand_flow)` thuần túy (vốn bị kéo về 0 khi công nhân tì cổ tay lên bàn máy) bằng công thức dung hợp năng lượng:
     $$\text{speed} = 0.5 \times v_{\text{median}} + 0.5 \times v_{\text{RMS}}$$
   - Giúp duy trì vận tốc ở mức hoạt động khi ngón tay bấm/đẩy vải, ngăn chặn việc cắt nhầm ranh giới giữa chừng khi đang may.
3. **Nội suy thời gian (Temporal Interpolation) cho các frame mất Mask:**
   - Xóa bỏ hiện tượng gán `speed = 0` khi SAM3 bị drop mask (chiếm 5% - 11% số frames do chuyển động nhanh hoặc bị che khuất). Thay vào đó, áp dụng nội suy tuyến tính `np.interp` để bắc cầu qua các khoảng trống ngắn, dập tắt các "thung lũng dừng tay giả".
4. **Tối ưu hóa siêu tham số động học cho ngành may:**
   - Hạ khoảng cách tối thiểu `min_distance` từ **1.5 giây $\rightarrow$ 0.5 giây** (bắt kịp các thao tác vi mô như lại mũi 0.4s, xoay góc 0.5s).
   - Ngưỡng động thích ứng: $\text{Threshold} = \text{local\_mean} + 0.7 \times \text{local\_std}$.
   - Trọng số đa phương thức cân bằng: `Speed=0.40, Direction=0.40, Turbulence=0.20`, lọc góc xoay $25.0^\circ$.

---

### 1.2. Bảng số liệu định lượng trên cả 3 Video của CĐ 1

Tất cả 3 thư mục kết quả trong `data/1/kinematic/` đã được tính toán lại và đồng bộ hoàn toàn:

| Thư mục Video | Thời lượng | Số Segments CŨ | Số Segments MỚI (v2) | Frame Drop 0-Speed (Cũ $\rightarrow$ Mới) | Tình trạng đồng bộ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **`cam-03_20260805_073527_cut_0_0-0_57`** (Video đánh giá chính) | ~34.3s (858f) | 25 segs | **60 segments** | 21 frames $\rightarrow$ **0 frame** | **Hoàn tất (v2)** ✅ |
| **`cam-03_20260808_023207_cut_1_28-4_03`** | ~155.0s (2290f) | 61 segs | **134 segments** | 271 frames $\rightarrow$ **0 frame** | **Hoàn tất (v2)** ✅ |
| **`cam-03_20260815_070222_cut_1_03-2_19`** | ~75.9s (1118f) | 33 segs | **73 segments** | 44 frames $\rightarrow$ **0 frame** | **Hoàn tất (v2)** ✅ |

---

### 1.3. Độ khớp với Ranh giới Gán nhãn Thủ công (Ground-Truth 24 bước CĐ 1)

Đánh giá so sánh trực tiếp trên video chuẩn đối chiếu với `data/1/chuyen1_segment.json`:

| Chỉ số đánh giá | Trước cải tiến (Baseline) | Sau cải tiến (Stage 1 v2) | Mức độ cải thiện |
|:---|:---:|:---:|:---:|
| **Boundary Recall (@0.5s)** | 44.0% | **88.0%** (22/25 mốc) | **+44.0% (Tăng gấp đôi độ nhạy)** 🚀 |
| **Boundary Recall (@1.0s)** | 88.0% | **100.0%** (25/25 mốc) | **Bắt trọn 100% ranh giới chuyên gia** |
| **Thao tác khớp CẢ 2 ĐẦU (Start & End)** | 12.5% (3/24 bước) | **79.2% (19/24 bước)** | **Tăng từ 3 bước lên 19 bước** 🚀 |
| **Thao tác khớp ÍT NHẤT 1 ĐẦU** | 66.7% | **100.0% (24/24 bước)** | **Độ bao phủ tuyệt đối** |
| **Boundary Precision (@0.5s)** | 48.0% | **45.9%** (28/61 vết cắt) | Ổn định (phù hợp tỷ lệ over-segment 2.4x) |
| **F1-Score (@0.5s)** | 45.9% | **60.3%** | **+14.4%** |

*Tất cả 4 file output (`action_segments.json`, `action_boundaries_dynamic.npy`, `pipe1_report.json`, `decomposed_motion.npz`) trong cả 3 thư mục đều đã được cập nhật đồng bộ.*

---

## PHẦN 2: CÁC THÁCH THỨC HIỆN TẠI KHI KẾT NỐI VÀO STAGE 2

Mặc dù Stage 1 đã giải quyết xuất sắc bài toán phát hiện ranh giới vật lý, việc chuyển giao sang Stage 2 theo kiến trúc hiện tại ([`src/analysis/segment_classify.py`](file:///Users/admin/Documents/work/WE/poc-ai4training/ai4training-aicore-poc/src/analysis/segment_classify.py)) đang đối mặt với **4 rào cản nghiêm trọng**:

### ⚠️ Thách thức 1: Vỡ ngữ cảnh thời gian do tiếp cận bằng ảnh tĩnh (Temporal Blindness)
* **Thực trạng:** 
  Stage 2 hiện tại hoạt động theo cơ chế: Với mỗi đoạn vi-phân đoạn (0.5s – 1.0s), hệ thống trích xuất 2–4 ảnh tĩnh, convert sang chuỗi base64 và gửi vào chat API của LLM (`qwen/qwen3.7-flash` hoặc GPT-4o).
* **Vấn đề phát sinh:**
  - **Mất vector chuyển động:** Khi chỉ nhìn vào 2–3 ảnh tĩnh của một khoảng thời gian 0.6s, VLM không thể phân biệt được bàn tay đang "đẩy vải vào", "giữ vải xoay góc" hay "đang nghỉ tay".
  - **Ảo giác (Hallucination):** Vì thiếu chuyển động động học liên tục, VLM dễ đoán mò tên thao tác, hoặc thường xuyên trả về `UNKNOWN`.

### ⚠️ Thách thức 2: Bùng nổ chi phí và độ trễ (Latency & Cost Explosion)
* **Thực trạng:**
  Thuật toán Stage 1 tạo ra **60 segments** cho video 34s và **134 segments** cho video 155s.
* **Vấn đề phát sinh:**
  - **Độ trễ quá lớn:** Vòng lặp `for` gọi VLM tuần tự 60 lần tiêu tốn khoảng **45 – 60 giây** cho video 34s. Đối với video 155s (134 segments), thời gian chạy sẽ lên tới **3 – 4 phút**.
  - **Rủi ro kết nối:** Gửi hàng trăm request chứa ảnh dung lượng lớn dễ dẫn đến lỗi nghẽn mạng, rate limit hoặc connection timeout từ nhà cung cấp API.

### ⚠️ Thách thức 3: Xung đột phân cấp (Granularity Mismatch) & Hiện tượng loạn nhãn
* **Bản chất bài toán:**
  - Stage 1 phát hiện **Micro-actions (vi-thao tác vật lý)**: mỗi cử động tay, chỉnh vải, đạp ga tạo thành một đoạn 0.6s – 1.2s.
  - Chuyên gia gán nhãn ở cấp độ **Macro-step (bước quy trình vĩ mô)**: gồm 24 bước lớn kéo dài 2s – 6s. Mỗi bước lớn bao gồm 2–4 vi-thao tác liên tiếp.
* **Vấn đề phát sinh:**
  - **Label Flickering:** Do gọi từng đoạn độc lập, VLM gán nhãn chập chờn (ví dụ: *May cạnh 1 $\rightarrow$ Chỉnh vải $\rightarrow$ May cạnh 1* hoặc nhảy cóc ngược từ bước 20 về bước 3).
  - Khi nhãn bị phân mảnh, logic gộp liên tiếp đơn giản (`consecutive merge`) trong `segment_classify.py` bị vô hiệu hóa, không thể thu gom về đúng 24 bước chuẩn.

### ⚠️ Thách thức 4: Hạn chế của kiến trúc Chat Completions so với Video Native
* **Vấn đề cốt lõi:**
  Kiến trúc hiện tại đang dùng API chat văn bản mở rộng (Chat Completions với `image_url` base64). Đây là kiến trúc dành cho bài toán *Visual Question Answering* trên ảnh tĩnh, **hoàn toàn không tối ưu cho dữ liệu chuỗi video công nghiệp (Sequential Industrial Video)**.

