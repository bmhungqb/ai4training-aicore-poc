# Báo Cáo Đánh Giá Độ Align Của Action Segmentation (Stage 1: SAM + Sea-RAFT)

> **Ngày thực hiện**: 10/09/2026  
> **Tập dữ liệu**: Chuyền 1 (9 công đoạn độc lập: CĐ 1, 2, 3, 4, 5, 6, 8, 9, 10)  
> **Dữ liệu Ground Truth**: `data/{cd}/chuyen1_segment.json`  
> **Dữ liệu dự đoán Stage 1**: `data_result/{cd}/kinematic/{video}/action_segments.json`  

---

## 1. Tổng quan & Phương pháp đánh giá

### 1.1 Mục tiêu
Đánh giá mức độ khớp (alignment) giữa các ranh giới phân tách thao tác do mô hình Kinematic (Stage 1 kết hợp SAM + Sea-RAFT) tự động trích xuất với các mốc thời gian chuẩn do chuyên gia ghi nhận trong file Ground Truth (`chuyen1_segment.json`).

### 1.2 Dữ liệu đánh giá (9 Công Đoạn Độc Lập)
*(Ghi chú: CĐ 11 là bản sao/alias của CĐ 10 vì cùng chung một video "Tra tay chính" `cam-03_20260807_030000_cut_4_54-12_29.mp4`, do đó đã được loại trừ để tránh trùng lặp số liệu).*

| CĐ | Tên công đoạn | Video tương ứng | Số bước GT | Số mốc ranh giới GT | Số vết cắt máy (Stage 1) |
|:---:|:---|:---|:---:|:---:|:---:|
| **1** | Diễu TP 4 cạnh túi lai x2 | `cam-03_20260805_073527_cut_0_0-0_57.mp4` | 24 | 25 | 64 |
| **2** | Khóa lưỡi gà + doup đoạn cơi | `cam-03_20260805_074035_cut_1_15-3_16.mp4` | 37 | 38 | 145 |
| **3** | Ráp chèn tay lót | `cam-03_20260805_074444_cut_0_7-0_44.mp4` | 14 | 15 | 37 |
| **4** | May đáp túi lai | `cam-03_20260805_074840_cut_0_2-0_31.mp4` | 11 | 12 | 35 |
| **5** | Rập lược TP cầu dk | `cam-03_20260807_023331_cut_0_33-2_47.mp4` | 25 | 26 | 142 |
| **6** | Ráp ngang đô sau | `cam-03_20260807_024253_cut_1_24-2_55.mp4` | 24 | 25 | 96 |
| **8** | Tra tay lót | `cam-03_20260805_081345_cut_0_0-5_24.mp4` | 63 | 66 | 343 |
| **9** | Tra cổ chính | `cam-03_20260805_082208_cut_3_30-5_29.mp4` | 40 | 41 | 142 |
| **10** | Tra tay chính | `cam-03_20260807_030000_cut_4_54-12_29.mp4` | 111 | 115 | 501 |
| **Tổng** | **9 công đoạn** | **9 video** | **349** | **363** | **1,505** *(3,001 mốc start/end)* |

---

## 2. Phương pháp tính chỉ số (Metrics & Formulation)

### 2.1 Mức độ khớp ranh giới (Boundary Match)
Với mỗi mốc thời gian $t_{gt}$ trong Ground Truth và tập vết cắt $P$ do Stage 1 sinh ra:
- Mốc $t_{gt}$ được tính là **Trúng (Hit)** nếu:
  $$\exists p \in P : |t_{gt} - p| \le \tau \quad (\tau \text{ là dung sai, chuẩn mặc định } \pm 0.5\text{s})$$

### 2.2 Phân biệt cách tính Macro và Micro
Mỗi video $i$ có $G_i$ mốc GT ($H_i^{GT}$ mốc trúng), và $P_i$ vết cắt máy ($H_i^{Pred}$ vết cắt hợp lệ):

- **Macro (Trung bình cộng giữa các công đoạn)**:
  Tính riêng cho từng công đoạn $i$ rồi chia đều trọng số ($1/N$):
  $$\text{Macro Recall} = \frac{1}{N} \sum_{i=1}^{N} \frac{H_i^{GT}}{G_i} \times 100\%$$
  $$\text{Macro Precision} = \frac{1}{N} \sum_{i=1}^{N} \frac{H_i^{Pred}}{P_i} \times 100\%$$
  $$\text{Macro F1} = \frac{1}{N} \sum_{i=1}^{N} \frac{2 \cdot \text{Prec}_i \cdot \text{Rec}_i}{\text{Prec}_i + \text{Rec}_i}$$
  *Ý nghĩa*: Đảm bảo đánh giá công bằng độ ổn định qua từng công đoạn may, không bị chi phối bởi video quá dài.

- **Micro (Gộp tổng số mốc toàn bộ lại rồi chia)**:
  Gom toàn bộ mốc và vết cắt của cả 9 video thành một tập lớn:
  $$\text{Micro Recall} = \frac{\sum_{i=1}^{N} H_i^{GT}}{\sum_{i=1}^{N} G_i} \times 100\% = \frac{326}{363} \approx \mathbf{89.81\%}$$
  $$\text{Micro Precision} = \frac{\sum_{i=1}^{N} H_i^{Pred}}{\sum_{i=1}^{N} P_i} \times 100\% = \frac{828}{3001} \approx \mathbf{27.59\%}$$
  *Ý nghĩa*: Đánh giá hiệu suất trên tổng khối lượng thao tác thực tế mà toàn hệ thống đã xử lý.

---

## 3. Kết quả đánh giá chi tiết (Cửa sổ $\pm 0.5$ giây)

| CĐ | Tên công đoạn | Boundary Recall | Boundary Precision | F1-Score | Khớp CẢ 2 đầu (Start & End) | Khớp ÍT NHẤT 1 đầu |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | Diễu TP 4 cạnh túi lai x2 | **100.0%** | 47.2% | **64.2%** | **24/24 (100.0%)** | 24/24 (100.0%) |
| **2** | Khóa lưỡi gà + doup đoạn cơi | **92.1%** | 35.6% | **51.4%** | **31/37 (83.8%)** | 37/37 (100.0%) |
| **3** | Ráp chèn tay lót | **86.7%** | 43.8% | **58.2%** | **10/14 (71.4%)** | 14/14 (100.0%) |
| **4** | May đáp túi lai | 75.0% | 31.9% | 44.7% | 5/11 (45.5%) | 11/11 (100.0%) |
| **5** | Rập lược TP cầu dk | **92.3%** | 20.5% | 33.5% | **22/25 (88.0%)** | 25/25 (100.0%) |
| **6** | Ráp ngang đô sau | **88.0%** | 25.7% | 39.7% | **18/24 (75.0%)** | 24/24 (100.0%) |
| **8** | Tra tay lót | **81.8%** | 20.1% | 32.3% | **42/63 (66.7%)** | 63/63 (100.0%) |
| **9** | Tra cổ chính | **95.1%** | 35.0% | **51.2%** | **37/40 (92.5%)** | 40/40 (100.0%) |
| **10** | Tra tay chính | **91.3%** | 26.7% | 41.3% | **92/111 (82.9%)** | 110/111 (99.1%) |

---

## 4. Tổng hợp thống kê toàn diện (Baseline Stage 1)

Ở mức dung sai chuẩn $\pm 0.5\text{s}$:
- **Macro Recall**: **$89.15\%$**
- **Micro Recall**: **$89.81\%$** *(Bắt trúng 326 / 363 mốc GT)*
- **Macro Precision**: **$31.84\%$** *(Micro Precision: 27.59%)*
- **Macro F1-Score**: **$46.29\%$**
- **Thao tác khớp CẢ 2 ĐẦU (Start & End)**: **$80.52\%$** *(281 / 349 thao tác)*
- **Thao tác khớp ÍT NHẤT 1 ĐẦU**: **$99.71\%$** *(348 / 349 thao tác — toàn bộ 9 video chỉ có duy nhất 1 thao tác bị lệch cả 2 đầu)*
- **Độ sai lệch trung bình (MAE - Mean Absolute Error)**:
  - Trong phạm vi $\pm 0.5\text{s}$: $\text{MAE} = \mathbf{0.191\text{s}}$ (Trung vị: $0.166\text{s}$, tương đương lệch chỉ ~3-5 khung hình).
  - Trong phạm vi $\pm 1.0\text{s}$: $\text{MAE} = \mathbf{0.233\text{s}}$ (Trung vị: $0.198\text{s}$).

---

## 5. Tiến trình độ khớp theo dải dung sai (Tolerance Sweep)

| Cửa sổ dung sai | Recall (Macro / Micro) | Precision (Macro / Micro) | F1-Score | Khớp CẢ 2 đầu | Biểu đồ trực quan |
|:---:|:---:|:---:|:---:|:---:|:---|
| **$\pm 0.25$s** | 55.6% / 57.0% (207/363) | 14.9% / 12.8% (384/3001) | 23.5% | 31.5% (110/349) | `[█████████░░░░░░░]` 55.6% |
| **$\pm 0.50$s** | **89.1% / 89.8% (326/363)** | **31.8% / 27.6% (828/3001)** | **46.9%** | **80.5% (281/349)** | `[██████████████░░]` 89.1% (*Chuẩn*) |
| **$\pm 0.75$s** | **95.4% / 95.6% (347/363)** | 43.9% / 38.0% (1139/3001) | **60.1%** | **91.4% (319/349)** | `[███████████████░]` 95.4% |
| **$\pm 1.00$s** | **98.4% / 98.6% (358/363)** | 54.4% / 47.1% (1412/3001) | **70.0%** | **97.4% (340/349)** | `[████████████████]` 98.4% |
| **$\pm 1.50$s** | **100.0% / 100.0% (363/363)** | 68.8% / 60.0% (1800/3001) | **81.5%** | **100.0% (349/349)** | `[████████████████]` 100.0% |
| **$\pm 2.00$s** | **100.0% / 100.0% (363/363)** | 75.6% / 66.6% (1998/3001) | **86.1%** | **100.0% (349/349)** | `[████████████████]` 100.0% |

---

## 6. Nhận xét & Kết luận

1. **Khả năng dò bắt ranh giới chuyển động (Recall) cực kỳ xuất sắc**:
   - Tỷ lệ Recall đạt gần **$90\%$ ở $\pm 0.5\text{s}$**, và đạt tới **$95.4\%$ ở $\pm 0.75\text{s}$**, $100\%$ ở $\pm 1.5\text{s}$.
   - Tỷ lệ bắt trúng ít nhất một đầu đạt **$99.71\%$**, chứng minh các tín hiệu vận tốc và góc chuyển động từ SAM + Sea-RAFT phản ánh chân thực các mốc chuyển tiếp thao tác của công nhân.

2. **Bản chất Over-segmentation của Stage 1**:
   - Precision đạt $\sim 31.8\%$ vì Stage 1 nhận diện chi tiết các chuyển động vi mô (dừng tay chỉnh vải, đổi hướng đưa vải, hạ chân vịt).
   - Đây là thiết kế có chủ đích cho pipeline 2 giai đoạn: **Stage 1 cắt nhạy để tối đa hóa Recall (không bỏ sót mốc)**, làm đầu vào hoàn hảo cho **Stage 2 (VLM) phân loại nhãn và ghép các micro-segments liên tiếp thành thao tác hoàn chỉnh**.

---

## 7. Lệnh tái hiện kết quả

```bash
uv run python -m tools.eval_boundary_recall --no-tune --exclude-cd 11 --out eval_report.json
```
