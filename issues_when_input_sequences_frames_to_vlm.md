# Các Vấn Đề Khi Đưa Chuỗi Frame Tĩnh (Sequence of Frames) Vào VLM & Giải Pháp Khắc Phục Bằng Optical Flow / Kinematics

## 1. Bối cảnh & Hiện trạng (Context & Problem Statement)

Trong quy trình phân tích thao tác may công nghiệp (Stage 2: Expert Analysis & Worker Classification), hệ thống hiện đang trích xuất chuỗi các frame kế tiếp (sequence frames, thông thường từ 3 đến 9 frames) từ mỗi phân cảnh (scene/action segment), sau đó áp dụng static ROI mask và gửi chuỗi ảnh này vào Vision-Language Model (VLM, ví dụ Gemini 1.5 Pro / Flash) kèm prompt yêu cầu mô tả chi tiết các thao tác kỹ thuật (cầm vải, xoay góc, kéo căng, gạt cữ, nhấn nút, đạp ga...).

### Nhận định thực tế:
Mặc dù việc trích xuất frame theo phân đoạn động học (kinematic boundaries) đã bắt đúng khoảng thời gian, nhưng việc **đưa nguyên chuỗi frame toàn cảnh (even có áp dụng mask tĩnh)** vào VLM bộc lộ nhiều điểm nghẽn nghiêm trọng về cả độ chính xác nhận diện lẫn hiệu năng xử lý.

Trong khi đó, hệ thống **đã có sẵn dữ liệu vector Optical Flow chất lượng cao (SEA-RAFT `flow.npz`) và mặt nạ tay (SAM3 `masks.npz`)**, cho phép biết chính xác tại mỗi frame và giữa các frame vùng nào đang chuyển động, hướng chuyển động và cường độ chuyển động là bao nhiêu.

---

## 2. Phân tích nguyên nhân gốc rễ (Root Causes)

Việc đưa chuỗi frame tĩnh nguyên bản vào VLM gặp phải 4 hạn chế cốt lõi sau:

### 2.1. Suy giảm độ phân giải vùng thao tác vi mô (Resolution Downsampling Loss)
- **Độ phân giải gốc:** Video camera công nghiệp thường có kích thước lớn ($2304 \times 1296$ px hoặc $1920 \times 1080$ px).
- **Cơ chế nén của VLM:** Khi nạp nhiều ảnh vào một request, VLM sẽ downsample ảnh về kích thước chuẩn (thường cạnh dài là 768px hoặc 1024px) để tính toán visual tokens (ví dụ: patch $14 \times 14$ px / token).
- **Vấn đề thực tế:** 
  - Vùng thao tác trọng yếu (mũi kim, mép gấp vải, chỉ may, đầu ngón tay tiếp xúc phôi) trong khung hình gốc chỉ chiếm khoảng $150 \times 150$ px đến $250 \times 250$ px (~2% đến 4% tổng diện tích ảnh).
  - Khi downsample xuống 768px, vùng này bị thu nhỏ lại chỉ còn khoảng $40 \times 40$ px đến $60 \times 60$ px. Mọi chi tiết vi mô như: chân vịt nhấc lên hay hạ xuống, sợi chỉ có bị tuột không, ngón cái có bấm giữ mép gấp hay không... đều bị mờ nhòe (aliasing & blur), khiến VLM hallucinate (suy đoán mò) thay vì quan sát thực tế.

### 2.2. Nhiễu ngữ cảnh tĩnh và phân tán Token Attention (Visual Noise & Attention Dilution)
- Khoảng 75% – 85% diện tích khung hình là các vùng tĩnh hoặc không chứa thao tác có giá trị đánh giá: thân máy may, mặt bàn gỗ, sàn nhà, thân người công nhân, ánh đèn phản chiếu...
- Ngay cả khi dùng **Static Mask**, phần nền bị làm mờ (dim/blackout) vẫn chiếm lượng lớn visual tokens của mô hình, làm phân tán cơ chế tự chú ý (self-attention) của VLM khỏi điểm tiếp xúc vật lý giữa bàn tay và vải.

### 2.3. Mơ hồ chiều hướng động lực học (Ambiguity in Motion Direction & Speed)
- Chuỗi 5–9 frame tĩnh chỉ thể hiện các "lát cắt trạng thái" rời rạc theo thời gian mà **không biểu diễn trực tiếp vector vận tốc** $(\vec{v} = (u, v))$.
- VLM thường gặp khó khăn khi phân biệt:
  - Công nhân đang **kéo căng** vải về phía mình hay đang **đẩy** vải vào chân vịt?
  - Bàn tay đang **vuốt phẳng** mép vải (lực miết) hay chỉ đơn thuần **chạm nhẹ** dẫn hướng?
  - Thao tác dừng đột ngột (do kẹt vải) hay dừng có chủ đích (để định vị góc may)?

### 2.4. Chi phí Token, Độ trễ (Latency) và Giới hạn Context Window
- Mỗi phân cảnh 24 scenes mà đưa $5 - 9$ frame tĩnh full-resolution vào mô hình sẽ tiêu tốn hàng chục nghìn visual tokens cho một tác vụ đơn lẻ.
- Tăng nguy cơ timeout, tăng chi phí API, và giảm khả năng so sánh đồng thời giữa Expert và Worker trong cùng một context window.

---

## 3. Dữ liệu động học sẵn có trong hệ thống (Available Kinematic Assets)

Trong repository hiện tại, pipeline tiền xử lý đã tính toán và lưu trữ đầy đủ các tài nguyên động học cực kỳ chính xác:

| Tài nguyên | File định dạng | Thuật toán / Nguồn | Thông tin chứa đựng |
| :--- | :--- | :--- | :--- |
| **Dense Optical Flow** | `*_flow.npz` | SEA-RAFT | Mảng 2D vector vận tốc $(u, v)$ tại từng pixel giữa $t$ và $t+1$. Cho biết chính xác pixel nào di chuyển theo hướng nào, tốc độ bao nhiêu. |
| **Hand Segmentation** | `*_masks.npz` | SAM3 Hand Tracker | Mặt nạ phân đoạn chi tiết bàn tay trái/phải, tọa độ tâm (centroid), diện tích tiếp xúc. |
| **Decomposed Motion** | `decomposed_motion.npz` | Phân rã động học | Độ lớn vận tốc tổng thể (magnitude), độ biến thiên hướng (directional turbulence), gia tốc vi mô. |
| **Kinematic Action Segments** | `kinematic_actions.json` | Motion Boundary Detection | Điểm chuyển tiếp thao tác (bắt đầu - tăng tốc - đạt đỉnh - giảm tốc - kết thúc). |

---

## 4. Các giải pháp cải tiến đề xuất (Proposed Solutions)

Dựa trên dữ liệu vector flow và mask có sẵn, có 4 giải pháp nâng cấp hiệu quả đưa vào VLM:

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 RAW FRAME SEQUENCE                     │
                  │             + OPTICAL FLOW (SEA-RAFT)                  │
                  │             + SAM3 HAND MASKS                          │
                  └───────────────────────────┬────────────────────────────┘
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         ▼                                    ▼                                    ▼
┌──────────────────┐               ┌──────────────────┐                 ┌──────────────────┐
│   GIẢI PHÁP 1    │               │   GIẢI PHÁP 2    │                 │   GIẢI PHÁP 3    │
│  DYNAMIC ACTION  │               │    DUAL-VIEW     │                 │ MOTION HEATMAP / │
│     ROI CROP     │               │    COMPOSITE     │                 │ TRAJECTORY (MHI) │
│ (Cắt bám động)   │               │ (Toàn cảnh + Vi mô)│               │ (Vẽ vector lên ảnh)│
└──────────────────┘               └──────────────────┘                 └──────────────────┘
```

---

### Giải pháp 1: Dynamic Action ROI Crop (Cắt khung hình động bám theo Flow & Tay)

#### Cơ chế hoạt động:
1. Từ `flow.npz`, tính độ lớn vận tốc tại mỗi pixel: $M(x, y) = \sqrt{u(x, y)^2 + v(x, y)^2}$.
2. Lọc các pixel có chuyển động vượt ngưỡng $M(x, y) > \tau_{motion}$.
3. Kết hợp với bounding box của 2 bàn tay (từ SAM3 `masks.npz`) và vị trí mũi kim cố định (needle point).
4. Tính toán **Bounding Box bao quanh vùng tương tác chính (Active Interaction Zone)**, mở rộng margin 15–20% để giữ ngữ cảnh liền kề.
5. Crop ảnh ở độ phân giải gốc $2304 \times 1296$ về vùng kích thước nhỏ (ví dụ $\sim 450 \times 450$ px), sau đó mới đưa vào VLM.

#### Ưu điểm:
- Giữ nguyên $100\%$ độ sắc nét của ngón tay, mép vải, chân vịt khi đưa vào VLM mà không bị VLM downsample làm mất chi tiết vi mô.
- Loại bỏ $80\%$ diện tích thừa thãi (mặt bàn, thân máy, nền phòng).
- Tiết kiệm token và giảm thiểu hallucination cực lớn.

---

### Giải pháp 2: Dual-View Composite (Ảnh ghép đôi Toàn cảnh - Vi mô)

#### Cơ chế hoạt động:
Mỗi mốc thời gian quan trọng chỉ tạo **1 ảnh duy nhất** gồm 2 nửa (hoặc 1 ảnh nền kèm 1 góc zoom picture-in-picture):
- **Khung hình trái (Macro View):** Toàn cảnh đã downsample có static mask (để VLM biết tư thế ngồi, hai tay ở đâu so với máy).
- **Khung hình phải (Micro Action Crop):** Vùng crop độ nét cao $1:1$ quanh bàn tay và mũi kim (như Giải pháp 1).

#### Ưu điểm:
- Cung cấp trọn vẹn cả ngữ cảnh vĩ mô (tư thế, vị trí phôi tổng thể) lẫn ngữ cảnh vi mô (cử chỉ ngón tay, nếp may).
- Chỉ tốn token cho 1 ảnh duy nhất thay vì phải gửi riêng lẻ nhiều ảnh.

---

### Giải pháp 3: Motion Heatmap / Directional Vector Overlay (Trực quan hóa Flow lên Frame)

#### Cơ chế hoạt động:
Thay vì bắt VLM tự đoán chiều chuyển động từ các ảnh tĩnh rời rạc, ta **nhúng trực tiếp thông tin vector flow** lên frame:
- **Motion History Image (MHI) / Flow Arrows:** Vẽ các mũi tên vector $(\vec{u}, \vec{v})$ có mã hóa màu sắc (ví dụ: mũi tên đỏ là tay trái kéo, xanh lá là tay phải đẩy) lên keyframe.
- **Vệt quỹ đạo (Temporal Trail):** Vẽ đường quỹ đạo của đầu ngón tay hoặc mép vải trong 0.5s gần nhất.

#### Ưu điểm:
- VLM "nhìn" thấy ngay lập tức hướng tác dụng lực và biên độ di chuyển mà không cần tốn nơ-ron suy diễn thời gian.
- Cho phép 1 keyframe duy nhất truyền tải được toàn bộ hành động của cả một phân đoạn 1–2 giây.

---

### Giải pháp 4: Kinematic Extremum Sampling (Lấy mẫu tại cực trị thay vì lấy đều)

#### Cơ chế hoạt động:
Thay vì chia đều thời gian (uniform sampling: 0%, 25%, 50%, 75%, 100%), hệ thống dựa vào đường cong vận tốc từ `decomposed_motion.npz` để chọn:
1. **$v_{\min}$ (Vận tốc cực tiểu / Trạng thái ổn định):** Thời điểm phôi vừa được căn chỉnh ngay ngắn trước khi may $\rightarrow$ Dùng frame này để VLM kiểm tra độ thẳng mép, độ khớp lót.
2. **$v_{\max}$ (Vận tốc cực đại / Đỉnh thao tác):** Thời điểm tay đang lướt kéo hoặc đẩy mạnh nhất $\rightarrow$ Dùng frame này (kèm vector flow overlay) để VLM kiểm tra kỹ thuật điều khiển lực.
3. **Boundary Transition (Điểm chuyển đổi):** Thời điểm nhấc tay hoặc đổi chiều thao tác.

---

## 5. Bảng so sánh các phương án & Đề xuất lộ trình

| Tiêu chí | Chuỗi Frame Tĩnh (Hiện tại) | Giải pháp 1: Dynamic Action Crop | Giải pháp 2: Dual-View Composite | Giải pháp 3: Flow Vector Overlay |
| :--- | :--- | :--- | :--- | :--- |
| **Độ rõ nét chi tiết may (mũi kim, mép vải)** | 🔴 Thấp (bị nén mờ) | 🟢 Rất cao (Full 1:1) | 🟢 Rất cao (Full 1:1) | 🟡 Trung bình |
| **Nhận thức hướng lực / chuyển động** | 🔴 Kém (suy đoán tĩnh) | 🟡 Gián tiếp qua chuỗi crop | 🟡 Gián tiếp | 🟢 Tuyệt đối trực quan |
| **Tiêu thụ Token VLM** | 🔴 Cao (5–9 full frames) | 🟢 Thấp (crop nhỏ gọn) | 🟢 Rất thấp (1 frame ghép) | 🟢 Rất thấp (1-2 keyframes) |
| **Độ phức tạp tiền xử lý** | 🟢 Đơn giản | 🟡 Trung bình (dùng flow + SAM3) | 🟡 Trung bình | 🟡 Trung bình (vẽ vector) |

### Lộ trình triển khai khuyến nghị:
1. **Giai đoạn 1 (Quick Win):** Triển khai **Giải pháp 1 (Dynamic Action Crop)** kết hợp **Giải pháp 4 (Kinematic Extremum Sampling)**.
   - Sử dụng `flow.npz` và `masks.npz` đã có sẵn trong folder `kinematic/`.
   - Mỗi cảnh chỉ cần 2–3 frame crop sắc nét tập trung vào bàn tay và mũi kim thay vì 9 frame full view mờ nhòe.
2. **Giai đoạn 2 (Advanced Visual Prompting):** Thử nghiệm **Giải pháp 3 (Overlay Motion Vectors)** trên các thao tác phức tạp (xoay góc diễu túi, kéo lót túi) để VLM nhận diện chính xác hướng thao tác mà không bị nhầm lẫn giữa đẩy và kéo.
