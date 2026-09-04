# Hướng dẫn chạy Stage 1: Kinematic Pre-Segmentation (Phân đoạn hành động vật lý)

Tài liệu hướng dẫn chi tiết quy trình chuẩn bị môi trường, tải dữ liệu và thực thi **Stage 1 (Phase 1)** trong pipeline AI for Training.

---

## 1. Giới thiệu Stage 1 (Phase 1)

Stage 1 là bước **phân đoạn hành động vật lý (Kinematic Pre-segmentation)** của công nhân:
- **Nguyên lý**: Kết hợp **SAM 3** (nhận diện và tracking tay/cánh tay) + **SEA-RAFT** (tính toán dòng quang học optical flow) + thuật toán **Fusion Magnitude/Direction** (phát hiện điểm chuyển hướng và thung lũng vận tốc của bàn tay).
- **Đặc điểm**:
  - Không sử dụng VLM, **không cần OpenRouter API key**, không phụ thuộc video chuyên gia.
  - Phân tích thuần động học để tìm ranh giới vật lý: *Hành động bắt đầu và kết thúc khi nào*.
- **Đầu ra**: File `action_segments.json` (thời điểm `start_time_s`, `end_time_s`, `duration_s`) làm đầu vào cho Stage 2 (phân loại công đoạn với VLM).

---

## 2. Chuẩn bị môi trường (Setup)

### 2.1. Yêu cầu hệ thống
- **Hệ điều hành**: Linux (Ubuntu 20.04/22.04 khuyến nghị) hoặc macOS / Windows WSL2.
- **Phần cứng**: Khuyến nghị có **NVIDIA GPU** (VRAM ≥ 8GB - 16GB) vì mô hình SAM3 và SEA-RAFT chạy deep learning nặng.
- **Công cụ hệ thống**: `ffmpeg` (để cắt và xử lý video).

Cài đặt `ffmpeg`:
```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg

# macOS
brew install ffmpeg
```

### 2.2. Khởi tạo Python Environment
Khuyến nghị sử dụng Python **3.10** hoặc **3.11**:

```bash
# Tạo virtual environment
python3 -m venv .venv

# Kích hoạt môi trường
source .venv/bin/activate   # Trên Linux/macOS
# hoặc: .venv\Scripts\activate trên Windows
```

### 2.3. Cài đặt Dependencies

1. **Cài đặt PyTorch phù hợp với phiên bản CUDA của máy**:
   *(Truy cập [pytorch.org](https://pytorch.org) để chọn bản phù hợp)*:
   ```bash
   # Ví dụ CUDA 12.1:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

   # Hoặc CUDA 11.8:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Cài đặt các thư viện cơ bản**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Cài đặt các thư viện cho Stage 1 (Kinematic Pipeline)**:
   ```bash
   pip install -r requirements-kinematic.txt
   ```

### 2.4. Đăng nhập Hugging Face (Quan trọng cho SAM 3)
Mô hình SAM 3 (`facebook/sam3`) được tải tự động qua Hugging Face:
- Hãy chắc chắn tài khoản Hugging Face của bạn đã được cấp quyền truy cập mô hình `facebook/sam3` trên [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3).
- Đăng nhập Hugging Face CLI trên máy chạy:
  ```bash
  huggingface-cli login
  ```
  *(Dán Access Token của bạn có quyền read)*.

---

## 3. Tải dữ liệu (Download Data)

Dữ liệu toàn bộ các video công đoạn đã được đóng gói sẵn thành file zip trên Google Drive:
- **Link Drive**: [data.zip](https://drive.google.com/file/d/1qwXBnvnvwC3THJAZetMWl0Rcr50MiIii/view?usp=sharing)
- **File ID**: `1qwXBnvnvwC3THJAZetMWl0Rcr50MiIii`

### 3.1. Tải bằng dòng lệnh qua `gdown` (Khuyến nghị)
Sử dụng `gdown` (đã có trong `requirements.txt`) để tải trực tiếp file zip về thư mục gốc:

```bash
# Tải file data.zip
gdown --id 1qwXBnvnvwC3THJAZetMWl0Rcr50MiIii -O data.zip
```

*(Hoặc có thể mở trực tiếp link trình duyệt ở trên để tải về thủ công và đặt vào thư mục dự án).*

### 3.2. Giải nén dữ liệu
Giải nén `data.zip` ra thư mục `data/`:

```bash
# Giải nén
unzip data.zip
```

Sau khi giải nén, cấu trúc thư mục `data/` sẽ bao gồm các thư mục công đoạn:
```
data/
├── 1/
├── 2/
├── ...
└── 20/
```

---

## 4. Hướng dẫn chạy Stage 1 (Run Stage 1)

> **Ghi chú về ROI Mask**: Trong thư mục `data/` tải về đã có sẵn các file mặt nạ `<tên_video>.mask.png` (khoanh vùng công nhân mục tiêu để loại trừ người ngồi cạnh). Pipeline sẽ **tự động phát hiện và áp dụng** các mask này.

### 4.1. Chạy Batch toàn bộ các công đoạn (Run All)
Chạy phân đoạn hành động cho **tất cả** các video `.mp4` có trong thư mục `data/` (tất cả các công đoạn), kèm theo cờ `--visualize` để sinh video debug và đồ thị phân tích:

```bash
# Lệnh cơ bản có hiển thị trực quan hóa (visualize):
python pipeline.py segment --all-data --visualize

# Hoặc cú pháp tương đương:
python pipeline.py segment --cong-doan all --visualize
```

### 4.2. Chạy cho một công đoạn cụ thể
Ví dụ chỉ chạy cho tất cả các video thuộc **Công đoạn 1** (`data/1/*.mp4`):
```bash
python pipeline.py segment --cong-doan 1
# hoặc:
python pipeline.py segment --cd 1
```

### 4.3. Chạy cho một video đơn lẻ tùy chọn
Chỉ định chính xác đường dẫn video và thư mục lưu kết quả:
```bash
python pipeline.py segment \
    --video data/1/cam-03_20260805_073527_cut_0_0-0_57.mp4 \
    --out-dir data/1/kinematic/cam-03_20260805_073527_cut_0_0-0_57
```

---

## 5. Cấu trúc kết quả đầu ra (Output Layout)

Sau khi Stage 1 hoàn tất, kết quả của mỗi video sẽ được lưu tại:
`data/{công_đoạn}/kinematic/{tên_video_stem}/`

Ví dụ với video `cam-03_20260805_073527_cut_0_0-0_57.mp4` trong `data/1/`:
```
data/1/kinematic/cam-03_20260805_073527_cut_0_0-0_57/
├── action_segments.json                # KẾT QUẢ CHÍNH: danh sách segments (thời gian start/end)
├── pipe1_report.json                   # Báo cáo tổng hợp chi tiết phân đoạn động học
├── action_boundaries_dynamic.npy       # Ranh giới frame cắt động học dạng mảng NumPy
├── decomposed_motion.npz               # Vector phân tích chuyển động (vận tốc, hướng)
├── cam-03_..._masks.npz                # Mặt nạ tay SAM 3 được nén
├── cam-03_..._flow.npz                 # Tensor optical flow SEA-RAFT
├── motion_decomposition_smooth_plot.png # (Nếu có --visualize) Biểu đồ vận tốc và hướng
└── cam-03_..._pipe1_viz.mp4            # (Nếu có --visualize) Video kết quả hiển thị ranh giới
```

---

## 6. Xử lý sự cố thường gặp (Troubleshooting)

1. **CUDA Out of Memory (OOM)**:
   - **Nguyên nhân**: Video độ phân giải cao hoặc VRAM GPU < 16GB.
   - **Khắc phục**: Thêm các cờ giảm tải `--resize-scale 0.25 --frame-step 2 --frame-by-frame`.

2. **Lỗi `Cannot access gated repo for model facebook/sam3`**:
   - **Khắc phục**: Vào link [facebook/sam3](https://huggingface.co/facebook/sam3) chấp nhận điều khoản, sau đó chạy `huggingface-cli login` trên terminal.

3. **Video chạy lại bị bỏ qua (Reusing existing report)**:
   - **Giải thích**: Pipeline tự động phát hiện nếu video đã có file `pipe1_report.json` và tái sử dụng để tiết kiệm thời gian.
   - **Khắc phục**: Thêm cờ `--force-segment` để ép phân tích lại từ đầu.

4. **Nhiễu chuyển động từ công nhân ngồi cạnh**:
   - **Giải thích**: Các video trong `data.zip` đã được vẽ sẵn mask `.mask.png` đi kèm nên pipeline tự động lọc người khác. Nếu có video mới phát sinh, có thể dùng công cụ `python -m tools.mask_editor.server --dir data --port 8765` để vẽ thêm.

