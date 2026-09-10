# Báo Cáo Đánh Giá Độ Align Của Action Segmentation (Stage 1: SAM + Sea-RAFT)

> **Ngày thực hiện**: 10/09/2026  
> **Tập dữ liệu**: Chuyền 1 (9 công đoạn độc lập: CĐ 1, 2, 3, 4, 5, 6, 8, 9, 10 | Đã loại trừ CĐ 11 trùng)  
> **Dữ liệu Ground Truth**: `data/{cd}/chuyen1_segment.json`  
> **Dữ liệu dự đoán Stage 1**: `data_result/{cd}/kinematic/{video}/action_segments.json`  
> **File Excel chi tiết kèm theo**: [`evaluation_result_9cd.xlsx`](evaluation_result_9cd.xlsx)

---

## 1. Bản chất dữ liệu & Phương pháp đánh giá chuẩn xác

### 1.1 Bản chất phân đoạn liền mạch (Contiguous Segments)
Trong file `action_segments.json`:
- Stage 1 chia video thành chuỗi phân đoạn hành động vi mô liên tục (không có khoảng hở hay chồng lấn giữa các khung hình).
- Giữa 2 phân đoạn liên tiếp $S_{i-1}$ và $S_i$, thời điểm chuyển tiếp (transition boundary) giữa khung hình cuối của đoạn trước và khung hình đầu của đoạn sau tạo nên **1 ranh giới duy nhất**:
  $$t_{\text{trans}} = \frac{S_{i-1}.\text{end\_time} + S_i.\text{start\_time}}{2}$$
- Như vậy, một video có $M$ phân đoạn sẽ tương ứng với đúng **$M - 1$ điểm cắt bên trong** và **$M + 1$ ranh giới chuyển tiếp** (tính cả 2 đầu video).
- *Lưu ý*: Việc tính riêng rẽ cả `start_time` và `end_time` của từng frame rời rạc (lệch 1 frame $\approx 0.04-0.06\text{s}$) trước đây làm nhân đôi số mốc máy một cách hình thức. Phương pháp đánh giá chuẩn xác ở đây dựa trên **ranh giới chuyển tiếp thực sự** giữa các phân đoạn.

### 1.2 Quy mô dữ liệu 9 công đoạn độc lập

| CĐ | Tên công đoạn | Video tương ứng | Số bước GT (Thao tác) | Số mốc GT (Ranh giới) | Số đoạn máy chia (Segments) | Ranh giới chuyển tiếp của máy |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| **1** | Diễu TP 4 cạnh túi lai x2 | `cam-03_20260805_073527_cut_0_0-0_57.mp4` | 24 | 25 | 64 | **65** |
| **2** | Khóa lưỡi gà + doup đoạn cơi | `cam-03_20260805_074035_cut_1_15-3_16.mp4` | 37 | 38 | 145 | **146** |
| **3** | Ráp chèn tay lót | `cam-03_20260805_074444_cut_0_7-0_44.mp4` | 14 | 15 | 37 | **38** |
| **4** | May đáp túi lai | `cam-03_20260805_074840_cut_0_2-0_31.mp4` | 11 | 12 | 35 | **36** |
| **5** | Rập lược TP cầu dk | `cam-03_20260807_023331_cut_0_33-2_47.mp4` | 25 | 26 | 142 | **143** |
| **6** | Ráp ngang đô sau | `cam-03_20260807_024253_cut_1_24-2_55.mp4` | 24 | 25 | 96 | **97** |
| **8** | Tra tay lót | `cam-03_20260805_081345_cut_0_0-5_24.mp4` | 63 | 66 | 343 | **344** |
| **9** | Tra cổ chính | `cam-03_20260805_082208_cut_3_30-5_29.mp4` | 40 | 41 | 142 | **143** |
| **10** | Tra tay chính | `cam-03_20260807_030000_cut_4_54-12_29.mp4` | 111 | 115 | 501 | **502** |
| **Tổng** | **9 công đoạn** | **9 video** | **349** | **363** | **1,505** | **1,514** |

---

## 2. Tiêu chuẩn tính toán chỉ số (Metrics & Formulation)

1. **Khớp ranh giới (Boundary Match)**:
   Mốc Ground Truth $t_{gt}$ được tính là **ĐẠT (Hit)** nếu tồn tại ranh giới chuyển tiếp $p$ của máy sao cho $|t_{gt} - p| \le \tau$ (chuẩn mặc định $\tau = \pm 0.5\text{s}$).

2. **Độ bao phủ mốc thao tác (Macro vs Micro Recall)**:
   Mỗi video $i$ có $G_i$ mốc GT ($H_i^{GT}$ mốc trúng bởi ít nhất một ranh giới máy):

- **Macro Recall (Trung bình giữa các công đoạn)**:
  $$\text{Macro Recall} = \frac{1}{N} \sum_{i=1}^{N} \frac{H_i^{GT}}{G_i} \times 100\% = \mathbf{86.20\%}$$
  *Ý nghĩa*: Đánh giá mức độ đồng đều và tin cậy của mô hình trên mọi loại công đoạn (mỗi công đoạn trọng số bình đẳng $1/9$).

- **Micro Recall (Tổng hợp toàn thể)**:
  $$\text{Micro Recall} = \frac{\sum H_i^{GT}}{\sum G_i} = \frac{317}{363} \approx \mathbf{87.33\%}$$
  *Ý nghĩa*: Đánh giá hiệu suất trên toàn bộ khối lượng dữ liệu thực tế (bắt trúng 317 / 363 mốc GT).

- **Khớp thao tác nghiệp vụ (Step-level Alignment)**:
  - **Khớp CẢ 2 ĐẦU (Both Start & End)**: Bước thao tác $[T_{\text{start}}, T_{\text{end}}]$ có cả điểm bắt đầu và điểm kết thúc đều khớp vết cắt máy trong khoảng $\pm \tau$. Đây là điều kiện tiên quyết để cô lập trọn vẹn 1 thao tác.
  - **Khớp ÍT NHẤT 1 ĐẦU**: Bắt trúng ít nhất Start hoặc End.

---

## 3. Kết quả đánh giá chi tiết 9 công đoạn (Cửa sổ $\pm 0.5$ giây)

| CĐ | Tên công đoạn | Số bước GT | Số mốc GT | Số ranh giới máy | Tỷ lệ cắt máy/GT | Boundary Recall | Độ lệch MAE (s) | Khớp CẢ 2 đầu (Start & End) | Khớp ÍT NHẤT 1 đầu |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | Diễu TP 4 cạnh túi lai x2 | 24 | 25 | 65 | 2.7x | **100.0%** | 0.135s | **24/24 (100.0%)** | 24/24 (100.0%) |
| **2** | Khóa lưỡi gà + doup đoạn cơi | 37 | 38 | 146 | 3.9x | **92.1%** | 0.203s | **31/37 (83.8%)** | 37/37 (100.0%) |
| **3** | Ráp chèn tay lót | 14 | 15 | 38 | 2.7x | **86.7%** | 0.231s | **10/14 (71.4%)** | 14/14 (100.0%) |
| **4** | May đáp túi lai | 11 | 12 | 36 | 3.3x | 75.0% | 0.198s | 5/11 (45.5%) | 11/11 (100.0%) |
| **5** | Rập lược TP cầu dk | 25 | 26 | 143 | 5.7x | **80.8%** | 0.210s | **17/25 (68.0%)** | 24/25 (96.0%) |
| **6** | Ráp ngang đô sau | 24 | 25 | 97 | 4.0x | **80.0%** | 0.238s | **15/24 (62.5%)** | 24/24 (100.0%) |
| **8** | Tra tay lót | 63 | 66 | 344 | 5.5x | **77.3%** | 0.233s | **38/63 (60.3%)** | 61/63 (96.8%) |
| **9** | Tra cổ chính | 40 | 41 | 143 | 3.6x | **92.7%** | 0.192s | **35/40 (87.5%)** | 40/40 (100.0%) |
| **10** | Tra tay chính | 111 | 115 | 502 | 4.5x | **91.3%** | 0.238s | **92/111 (82.9%)** | 110/111 (99.1%) |

---

## 4. Tổng hợp thống kê toàn diện (Baseline Stage 1)

Ở mức dung sai chuẩn $\pm 0.5\text{s}$:
- **Macro Recall**: **$86.20\%$**
- **Micro Recall**: **$87.33\%$** *(Bắt trúng 317 / 363 mốc ranh giới Ground Truth)*
- **Thao tác khớp CẢ 2 ĐẦU (Start & End)**: **$76.50\%$** *(267 / 349 thao tác)*
- **Thao tác khớp ÍT NHẤT 1 ĐẦU**: **$98.28\%$** *(343 / 349 thao tác)*
- **Độ sai lệch trung bình (MAE)**:
  - Trong phạm vi $\pm 0.5\text{s}$: $\text{MAE} = \mathbf{0.213\text{s}}$ (Trung vị: $0.198\text{s}$, tương đương lệch ~3–4 khung hình).
  - Trong phạm vi $\pm 1.0\text{s}$: $\text{MAE} = \mathbf{0.233\text{s}}$ (Trung vị: $0.198\text{s}$).

---

## 5. Tiến trình theo dải dung sai thời gian (Tolerance Window Sweep)

| Cửa sổ dung sai | Macro Recall (%) | Micro Recall (%) | Số mốc GT trúng | Khớp CẢ 2 đầu | Khớp ÍT NHẤT 1 đầu | Sai lệch MAE (s) | Biểu đồ trực quan |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **$\pm 0.25$s** | 50.5% | 52.1% | 189 / 363 | 28.1% (98/349) | 80.2% (280/349) | 0.126s | `[████████░░░░░░░░]` 50.5% |
| **$\pm 0.50$s** | **86.2%** | **87.3%** | **317 / 363** | **76.5% (267/349)** | **98.3% (343/349)** | **0.213s** | `[██████████████░░]` 86.2% (*Chuẩn*) |
| **$\pm 0.75$s** | **95.4%** | **95.6%** | **347 / 363** | **91.4% (319/349)** | **100.0% (349/349)** | **0.255s** | `[███████████████░]` 95.4% |
| **$\pm 1.00$s** | **98.4%** | **98.6%** | **358 / 363** | **97.4% (340/349)** | **100.0% (349/349)** | **0.278s** | `[████████████████]` 98.4% |
| **$\pm 1.50$s** | **100.0%** | **100.0%** | **363 / 363** | **100.0% (349/349)** | **100.0% (349/349)** | **0.297s** | `[████████████████]` 100.0% |
| **$\pm 2.00$s** | **100.0%** | **100.0%** | **363 / 363** | **100.0% (349/349)** | **100.0% (349/349)** | **0.301s** | `[████████████████]` 100.0% |

---

## 6. Nhận xét cốt lõi về chất lượng Stage 1 (Kinematic)

1. **Khả năng bắt đúng ranh giới thao tác rất cao (Recall đạt $87.3\% \to 95.6\%$)**:
   - Ở mức $\pm 0.5\text{s}$, máy bắt trúng $317 / 363$ mốc GT ($87.33\%$). Khi nới lên $\pm 0.75\text{s}$, Recall đạt tới **$95.6\%$** và khớp cả 2 đầu đạt **$91.4\%$**.
   - Tỷ lệ thao tác có ít nhất 1 đầu được máy bắt trúng đạt **$98.28\%$** (chỉ có 6 thao tác nhỏ bị lệch cả 2 đầu).
2. **Bản chất Over-segmentation**:
   - Máy chia 1,505 đoạn (tương ứng 1,514 ranh giới), gấp $\sim 4.1$ lần so với 363 mốc GT.
   - Đây là hành vi đúng thiết kế của Stage 1: phát hiện các chuyển động đổi hướng/vi mô để không bỏ sót thao tác (Recall cao), sau đó Stage 2 (VLM) sẽ gom các đoạn cùng nhãn lại thành đúng 349 bước chuẩn.

---

## 7. Tài nguyên đi kèm & Lệnh chạy

- **File Excel chi tiết**: [`evaluation_result_9cd.xlsx`](evaluation_result_9cd.xlsx) (chứa Sheet tổng quan và Sheet chi tiết đối soát từng bước trong 349 bước).
- **Lệnh tái hiện kết quả**:
  ```bash
  uv run python -m tools.eval_boundary_recall --no-tune --exclude-cd 11 --out eval_report.json
  ```
