# Stage 1: Kinematic Pre-Segmentation Guide

Step-by-step instructions to set up the environment, download data, and run **Stage 1 (Phase 1)** of the AI for Training pipeline.

---

## 1. Overview

**Stage 1 (Phase 1)** performs physical action pre-segmentation on worker videos:
- **How it works**: Uses **SAM 3** (hand/arm tracking) + **SEA-RAFT** (optical flow) + **Magnitude/Direction Fusion** (detecting hand speed valleys and direction changes).
- **Key traits**: No VLM required, **no API key needed**, purely kinematic boundary detection.
- **Output**: `action_segments.json` (action boundaries with timestamps `start_time_s`, `end_time_s`, `duration_s`) used as input for Stage 2 (operation classification).

---

## 2. Setup

### Requirements
- **OS**: Linux (recommended) or macOS / Windows WSL2.
- **GPU**: NVIDIA GPU with CUDA (recommended for SAM 3 and SEA-RAFT).
- **System Tool**: `ffmpeg`.

### Quick Install

1. **Install `ffmpeg`**:
   ```bash
   # Ubuntu / Debian
   sudo apt update && sudo apt install -y ffmpeg

   # macOS
   brew install ffmpeg
   ```

2. **Set up Python environment (Python 3.10+)**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   # PyTorch with CUDA (adjust CUDA version if needed, e.g. cu121 or cu118)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

   # Project dependencies
   pip install -r requirements.txt
   pip install -r requirements-kinematic.txt
   ```

4. **Hugging Face Login (Required for SAM 3)**:
   Request access on [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), then log in via CLI:
   ```bash
   huggingface-cli login
   ```

---

## 3. Download Data

All operation videos and bundled ROI masks are pre-packaged into a single zip file on Google Drive:
- **Drive Link**: [data.zip](https://drive.google.com/file/d/1qwXBnvnvwC3THJAZetMWl0Rcr50MiIii/view?usp=sharing)
- **File ID**: `1qwXBnvnvwC3THJAZetMWl0Rcr50MiIii`

### Download & Extract:
```bash
# Download data.zip using gdown (included in requirements.txt)
gdown --id 1qwXBnvnvwC3THJAZetMWl0Rcr50MiIii -O data.zip

# Extract to project root
unzip data.zip
```

After extraction, the `data/` directory will contain folders `1/`, `2/`, ..., `20/`.

---

## 4. Run Stage 1 (Segmentation)

> **Note on ROI Masks**: The downloaded `data/` directory already includes `<video_stem>.mask.png` masks to isolate the target worker from adjacent people. The pipeline **automatically detects and applies** these masks.

### 4.1. Run All Videos Across All Operations (Batch Run)
Process all `.mp4` videos in `data/` with debug visualizations enabled:

```bash
python pipeline.py segment --all-data --visualize
```

*(Equivalent command: `python pipeline.py segment --cong-doan all --visualize`)*

### 4.2. Run a Single Operation
Process all videos for a specific operation (e.g. Operation 1):

```bash
python pipeline.py segment --cong-doan 1 --visualize
# or:
python pipeline.py segment --cd 1 --visualize
```

### 4.3. Run a Single Video File
Run segmentation on a specific video file:

```bash
python pipeline.py segment \
    --video data/1/cam-03_20260805_073527_cut_0_0-0_57.mp4 \
    --out-dir data/1/kinematic/cam-03_20260805_073527_cut_0_0-0_57 \
    --visualize
```

---

## 5. Output Layout

Outputs for each video are stored under:
`data/{operation_id}/kinematic/{video_stem}/`

Example output files:
```
data/1/kinematic/cam-03_20260805_073527_cut_0_0-0_57/
├── action_segments.json                 # Primary output: segment start/end timestamps
├── pipe1_report.json                    # Detailed kinematic report
├── action_boundaries_dynamic.npy        # Detected frame boundary indices
├── decomposed_motion.npz                # Motion decomposition vectors
├── cam-03_..._masks.npz                 # Compressed SAM 3 hand masks
├── cam-03_..._flow.npz                  # SEA-RAFT optical flow tensors
├── motion_decomposition_smooth_plot.png  # (From --visualize) Velocity & direction plot
└── cam-03_..._pipe1_viz.mp4             # (From --visualize) Annotated boundary video
```

---

## 6. Troubleshooting

1. **CUDA Out of Memory (OOM)**:
   - For high-resolution or long videos on smaller GPUs, add memory-reduction flags:
     ```bash
     python pipeline.py segment --all-data --resize-scale 0.25 --frame-step 2 --frame-by-frame
     ```

2. **`Cannot access gated repo for model facebook/sam3`**:
   - Accept terms on [facebook/sam3](https://huggingface.co/facebook/sam3) and authenticate using `huggingface-cli login`.

3. **Existing Report Reused**:
   - The pipeline skips videos that already have `pipe1_report.json`. To force re-running:
     ```bash
     python pipeline.py segment --all-data --force-segment
     ```
