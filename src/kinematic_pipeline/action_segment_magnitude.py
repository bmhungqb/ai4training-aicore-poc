#!/usr/bin/env python3
"""
action_segment_magnitude.py
Phân đoạn hành động dựa trên magnitude (tốc độ) của 2 tay.
Phát hiện các điểm đổi hướng/dừng chuyển động dựa trên đỉnh/đáy của đường magnitude.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter
import argparse

def load_data(flow_path, mask_path):
    """Load flow và masks"""
    flow_data = np.load(flow_path)
    mask_data = np.load(mask_path, allow_pickle=True)
    
    flows = flow_data['flow']  # [N, H, W, 2]
    left_masks = mask_data['left_masks']
    right_masks = mask_data['right_masks']
    fps = float(flow_data['fps'])
    
    return flows, left_masks, right_masks, fps

def _safe_mask(mask):
    """
    Guard against 0-dim object arrays and other degenerate SAM3 mask outputs.
    Returns a proper 2-D boolean numpy array or None.
    """
    if mask is None:
        return None
    if isinstance(mask, np.ndarray) and mask.ndim == 0:
        inner = mask.item()
        if inner is None:
            return None
        mask = np.asarray(inner)
    if not isinstance(mask, np.ndarray) or mask.ndim < 2:
        return None
    if mask.size == 0 or not mask.any():
        return None
    if mask.dtype != bool:
        mask = mask.astype(bool)
    return mask

def compute_hand_magnitudes(flows, left_masks, right_masks):
    """Tính magnitude (tốc độ) trung bình cho mỗi tay trong mỗi frame"""
    N = flows.shape[0]
    left_mags = []
    right_mags = []
    
    for i in range(N):
        flow = flows[i]
        left_mask = left_masks[i] if i < len(left_masks) else None
        right_mask = right_masks[i] if i < len(right_masks) else None
        
        # Left hand
        if left_mask is not None and left_mask.any():
            if left_mask.dtype != bool:
                left_mask = left_mask.astype(bool)
            # Resize mask to flow size if needed
            if left_mask.shape != (flows.shape[1], flows.shape[2]):
                left_mask = cv2.resize(left_mask.astype(np.uint8), 
                                      (flows.shape[2], flows.shape[1]), 
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
            hand_flow = flow[left_mask]
            if len(hand_flow) > 0:
                mag = np.mean(np.sqrt(hand_flow[:,0]**2 + hand_flow[:,1]**2))
                left_mags.append(mag)
            else:
                left_mags.append(0.0)
        else:
            left_mags.append(0.0)
        
        # Right hand
        if right_mask is not None and right_mask.any():
            if right_mask.dtype != bool:
                right_mask = right_mask.astype(bool)
            if right_mask.shape != (flows.shape[1], flows.shape[2]):
                right_mask = cv2.resize(right_mask.astype(np.uint8), 
                                       (flows.shape[2], flows.shape[1]), 
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
            hand_flow = flow[right_mask]
            if len(hand_flow) > 0:
                mag = np.mean(np.sqrt(hand_flow[:,0]**2 + hand_flow[:,1]**2))
                right_mags.append(mag)
            else:
                right_mags.append(0.0)
        else:
            right_mags.append(0.0)
    
    return np.array(left_mags), np.array(right_mags)

def smooth_magnitudes(mags, window=5):
    """Làm mượt magnitude bằng Savgol filter"""
    if len(mags) > window:
        try:
            return savgol_filter(mags, window, 3)
        except:
            return mags
    return mags

def find_peaks_and_valleys(mags, fps, min_distance=0.5, prominence=0.2):
    """Tìm đỉnh và đáy của đường magnitude"""
    # Tìm đỉnh (peaks)
    peaks, _ = find_peaks(mags, distance=int(fps*min_distance), prominence=prominence)
    # Tìm đáy (valleys) bằng cách tìm peaks trên -mags
    valleys, _ = find_peaks(-mags, distance=int(fps*min_distance), prominence=prominence)
    return peaks, valleys

def find_joint_boundaries(left_peaks, left_valleys, right_peaks, right_valleys, *args, margin=2, **kwargs):
    """
    Tìm các timestamp mà cả 2 tay đều có đỉnh hoặc đáy cùng lúc (sai số margin)
    """
    if len(args) > 0 and isinstance(args[0], int) and margin == 2:
        # If margin was passed as 5th positional arg
        margin = args[0]
    # Gộp đỉnh và đáy của cả 2 tay
    left_turning = np.concatenate([left_peaks, left_valleys])
    right_turning = np.concatenate([right_peaks, right_valleys])
    
    # Sắp xếp
    left_turning = np.sort(left_turning)
    right_turning = np.sort(right_turning)
    
    # Tìm các điểm chung (gần nhau)
    boundaries = []
    i, j = 0, 0
    while i < len(left_turning) and j < len(right_turning):
        if abs(left_turning[i] - right_turning[j]) <= margin:
            # Lấy trung bình
            boundary = int(round((left_turning[i] + right_turning[j]) / 2))
            boundaries.append(boundary)
            i += 1
            j += 1
        elif left_turning[i] < right_turning[j]:
            i += 1
        else:
            j += 1
    
    # Loại bỏ trùng lặp gần nhau
    unique_boundaries = []
    for b in boundaries:
        if not unique_boundaries or b - unique_boundaries[-1] > margin:
            unique_boundaries.append(b)
    
    return np.array(unique_boundaries)

def visualize_segments_on_video(video_path, mask_path, left_mags, right_mags, boundaries, output_dir, fps):
    """Tạo video với overlay magnitude và boundaries"""
    print("\nTạo video visualization...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở video {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Load masks để vẽ overlay
    mask_data = np.load(mask_path, allow_pickle=True)
    left_masks = mask_data['left_masks']
    right_masks = mask_data['right_masks']
    
    # Video writer
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "action_segments.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, video_fps, (width, height))
    
    if not writer.isOpened():
        print("Không thể tạo video writer")
        return
    
    frame_idx = 0
    processed = 0
    N = len(left_mags)  # số lượng flow frames
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Chỉ xử lý nếu frame_idx < N (có dữ liệu magnitude)
        if frame_idx < N:
            # Vẽ magnitude lên góc trái
            cv2.putText(frame, f"L mag: {left_mags[frame_idx]:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"R mag: {right_mags[frame_idx]:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Vẽ mask overlay nếu có
            if frame_idx < len(left_masks):
                left_mask = left_masks[frame_idx]
                right_mask = right_masks[frame_idx]
                
                # Left mask
                if left_mask is not None and left_mask.any():
                    if left_mask.dtype != bool:
                        left_mask = left_mask.astype(bool)
                    if left_mask.shape != (height, width):
                        left_mask = cv2.resize(left_mask.astype(np.uint8), (width, height), 
                                              interpolation=cv2.INTER_NEAREST).astype(bool)
                    overlay = np.zeros_like(frame)
                    overlay[left_mask] = [0, 0, 255]  # Red
                    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                
                # Right mask
                if right_mask is not None and right_mask.any():
                    if right_mask.dtype != bool:
                        right_mask = right_mask.astype(bool)
                    if right_mask.shape != (height, width):
                        right_mask = cv2.resize(right_mask.astype(np.uint8), (width, height), 
                                              interpolation=cv2.INTER_NEAREST).astype(bool)
                    overlay = np.zeros_like(frame)
                    overlay[right_mask] = [0, 255, 0]  # Green
                    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Vẽ boundary lines nếu frame_idx là boundary
            if frame_idx in boundaries:
                cv2.line(frame, (0, 0), (width, height), (255, 255, 0), 3)
                cv2.putText(frame, "BOUNDARY", (width//2-50, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Frame number
        cv2.putText(frame, f"Frame: {frame_idx}", (10, height-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        writer.write(frame)
        processed += 1
        if processed % 30 == 0:
            print(f"Processed {processed}/{total_frames}")
        
        frame_idx += 1
    
    cap.release()
    writer.release()
    print(f"Video saved: {out_path}")

def plot_magnitudes(left_mags, right_mags, left_peaks, left_valleys, right_peaks, right_valleys, boundaries, output_dir, *args, fps=25.0, **kwargs):
    """Vẽ biểu đồ magnitude và đánh dấu các đỉnh/đáy/boundaries"""
    fig, ax = plt.subplots(figsize=(15, 6))
    
    x = np.arange(len(left_mags))
    ax.plot(x, left_mags, 'r-', label='Left hand magnitude', linewidth=1.5)
    ax.plot(x, right_mags, 'g-', label='Right hand magnitude', linewidth=1.5)
    
    # Đánh dấu đỉnh/đáy
    ax.scatter(left_peaks, left_mags[left_peaks], color='darkred', s=50, marker='^', label='Left peak')
    ax.scatter(left_valleys, left_mags[left_valleys], color='red', s=50, marker='v', label='Left valley')
    ax.scatter(right_peaks, right_mags[right_peaks], color='darkgreen', s=50, marker='^', label='Right peak')
    ax.scatter(right_valleys, right_mags[right_valleys], color='green', s=50, marker='v', label='Right valley')
    
    # Đánh dấu boundaries
    for b in boundaries:
        ax.axvline(x=b, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(b, ax.get_ylim()[1]*0.9, f'B{b}', rotation=90, fontsize=8, color='blue')
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Magnitude (speed)')
    ax.set_title('Hand Magnitudes with Peaks, Valleys and Boundaries')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = Path(output_dir)
    out_path = output_dir if output_dir.suffix == '.png' else output_dir / 'magnitude_analysis.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description='Phân đoạn hành động dựa trên magnitude')
    parser.add_argument('--flows', required=True, help='Flow .npz file')
    parser.add_argument('--masks', required=True, help='Masks .npz file')
    parser.add_argument('--video', required=True, help='Video input')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--smooth_window', type=int, default=5, help='Cửa sổ làm mượt')
    parser.add_argument('--min_distance', type=float, default=0.5, help='Khoảng cách tối thiểu giữa các đỉnh/đáy (giây)')
    parser.add_argument('--prominence', type=float, default=0.2, help='Độ nổi bật của đỉnh/đáy')
    parser.add_argument('--margin', type=int, default=2, help='Sai số khi ghép đỉnh/đáy của 2 tay')
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    flows, left_masks, right_masks, fps = load_data(args.flows, args.masks)
    print(f"Loaded {len(flows)} frames")
    
    # Compute magnitudes
    print("Computing hand magnitudes...")
    left_mags, right_mags = compute_hand_magnitudes(flows, left_masks, right_masks)
    
    # Smooth magnitudes
    print("Smoothing magnitudes...")
    left_mags_s = smooth_magnitudes(left_mags, args.smooth_window)
    right_mags_s = smooth_magnitudes(right_mags, args.smooth_window)
    
    # Find peaks and valleys
    print("Finding peaks and valleys...")
    left_peaks, left_valleys = find_peaks_and_valleys(left_mags_s, fps, args.min_distance, args.prominence)
    right_peaks, right_valleys = find_peaks_and_valleys(right_mags_s, fps, args.min_distance, args.prominence)
    
    # Find joint boundaries
    print("Finding joint boundaries...")
    boundaries = find_joint_boundaries(left_peaks, left_valleys, right_peaks, right_valleys, args.margin)
    
    print(f"Left hand: {len(left_peaks)} peaks, {len(left_valleys)} valleys")
    print(f"Right hand: {len(right_peaks)} peaks, {len(right_valleys)} valleys")
    print(f"Joint boundaries: {len(boundaries)}")
    
    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot
    print("Plotting...")
    plot_magnitudes(left_mags_s, right_mags_s, left_peaks, left_valleys, right_peaks, right_valleys, boundaries, output_dir)
    
    # Save results
    results = {
        'left_mags': left_mags_s,
        'right_mags': right_mags_s,
        'left_peaks': left_peaks,
        'left_valleys': left_valleys,
        'right_peaks': right_peaks,
        'right_valleys': right_valleys,
        'boundaries': boundaries,
        'fps': fps
    }
    np.save(output_dir / 'action_boundaries.npy', results, allow_pickle=True)
    print(f"Results saved to {output_dir}/action_boundaries.npy")
    
    # Visualize on video
    print("Creating video visualization...")
    visualize_segments_on_video(args.video, args.masks, left_mags_s, right_mags_s, boundaries, output_dir, fps)
    
    print("\n" + "="*70)
    print("Action segmentation complete!")
    print(f"Boundaries found at frames: {boundaries.tolist()}")
    print(f"Video saved to {output_dir}/action_segments.mp4")
    print("="*70)

if __name__ == "__main__":
    main()