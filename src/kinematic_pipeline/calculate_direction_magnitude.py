#!/usr/bin/env python3
"""
calculate_direction_magnitude.py

Script to analyze hand motion from optical flow by decomposing it into:
1. Global Motion (Median Magnitude & Angle) - Represents the hand's overall translation.
2. Local Turbulence (Mean Absolute Deviation) - Represents finger articulation/noise.

Inputs:
- flow.npz (from SEA-RAFT)
- masks.npz (from SAM3)
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy.signal import savgol_filter, find_peaks
    import scipy.ndimage as ndimage
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def _gaussian_filter1d(data: np.ndarray, sigma: float) -> np.ndarray:
    """1D Gaussian filter with pure numpy fallback if scipy is not installed."""
    if HAS_SCIPY:
        return gaussian_filter1d(data, sigma=sigma)
    radius = int(max(3.0, 3.0 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / max(sigma, 1e-5))**2)
    kernel /= kernel.sum()
    return np.convolve(data, kernel, mode="same")

def _uniform_filter1d(data: np.ndarray, size: int) -> np.ndarray:
    """1D Uniform (moving average) filter with pure numpy fallback."""
    if HAS_SCIPY:
        return ndimage.uniform_filter1d(data, size=size)
    if size <= 1:
        return data.copy()
    kernel = np.ones(size, dtype=np.float32) / float(size)
    return np.convolve(data, kernel, mode="same")

def _find_peaks_1d(signal: np.ndarray, height: np.ndarray, distance: int) -> tuple[np.ndarray, dict]:
    """Peak finding with height threshold and minimum distance constraint."""
    if HAS_SCIPY:
        return find_peaks(signal, height=height, distance=distance)
    
    n = len(signal)
    cand_idx = []
    for i in range(1, n - 1):
        if signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
            h = height[i] if isinstance(height, np.ndarray) else height
            if signal[i] >= h:
                cand_idx.append(i)
    
    cand_idx.sort(key=lambda idx: signal[idx], reverse=True)
    selected = []
    for idx in cand_idx:
        if all(abs(idx - s) >= distance for s in selected):
            selected.append(idx)
    selected.sort()
    return np.array(selected, dtype=int), {}


def _safe_mask(mask) -> np.ndarray | None:
    """Guard against degenerate SAM3 masks."""
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

def load_data(flow_path: str, mask_path: str):
    flow_data = np.load(flow_path)
    mask_data = np.load(mask_path, allow_pickle=True)
    flows = flow_data["flow"]
    left_masks = mask_data["left_masks"]
    right_masks = mask_data["right_masks"]
    fps = float(flow_data.get("fps", 25.0))
    return flows, left_masks, right_masks, fps

def compute_decomposition(flows, masks, min_speed=0.3, erode_ksize=3):
    """
    Computes Global Speed, Angle, and Local Turbulence for a given hand mask over time.

    Key improvements:
    1. Mask Boundary Erosion (cv2.erode): Strips boundary bleeding pixels where background
       movement could cause false turbulence spikes.
    2. Combined Translation + RMS Energy: Combines median vector speed with root-mean-square
       speed so finger articulation is captured even when the palm/wrist is resting on the table.
    3. Temporal Interpolation: Interpolates missing/dropped mask frames instead of zero-filling,
       preventing fake 'stops' (valleys) that trigger false boundaries.
    """
    N, H, W, _ = flows.shape
    global_speeds = np.full(N, np.nan, dtype=np.float32)
    global_angles = np.full(N, np.nan, dtype=np.float32)
    turbulences = np.full(N, np.nan, dtype=np.float32)
    has_valid_mask = np.zeros(N, dtype=bool)

    kernel = np.ones((erode_ksize, erode_ksize), np.uint8) if erode_ksize > 1 else None

    for i in range(N):
        mask = _safe_mask(masks[i]) if i < len(masks) else None
        if mask is not None:
            if mask.shape != (H, W):
                mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            
            # 1. Mask Boundary Erosion: remove 1-2 edge pixels contaminated by background
            if kernel is not None:
                eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                if eroded.any():
                    mask = eroded

            hand_flow = flows[i][mask]
            if len(hand_flow) > 0:
                u_vals = hand_flow[:, 0]
                v_vals = hand_flow[:, 1]
                pixel_speeds = np.sqrt(u_vals**2 + v_vals**2)

                # 2. Combined Translation (Median) + RMS Energy
                u_global = np.median(u_vals)
                v_global = np.median(v_vals)
                med_speed = float(np.sqrt(u_global**2 + v_global**2))
                rms_speed = float(np.sqrt(np.mean(pixel_speeds**2)))

                # Preserve finger activity: when wrist rests, rms_speed keeps speed > 0
                speed = 0.5 * med_speed + 0.5 * rms_speed

                # Local Turbulence: deviation from global translation
                diff_vectors = hand_flow - np.array([u_global, v_global])
                turbulence = float(np.mean(np.linalg.norm(diff_vectors, axis=1)))

                global_speeds[i] = speed
                turbulences[i] = turbulence
                has_valid_mask[i] = True

                # Direction computation
                if med_speed >= min_speed:
                    global_angles[i] = np.degrees(np.arctan2(v_global, u_global))
                elif rms_speed >= min_speed:
                    # If whole hand translation is small, use dominant direction of active pixels (fingers)
                    top_k = max(1, int(0.3 * len(pixel_speeds)))
                    fast_idx = np.argpartition(pixel_speeds, -top_k)[-top_k:]
                    u_active = np.mean(u_vals[fast_idx])
                    v_active = np.mean(v_vals[fast_idx])
                    global_angles[i] = np.degrees(np.arctan2(v_active, u_active))

    # 3. Temporal Interpolation across missing/dropped masks
    valid_idx = np.where(has_valid_mask)[0]
    if len(valid_idx) > 0:
        all_idx = np.arange(N)
        global_speeds = np.interp(all_idx, valid_idx, global_speeds[valid_idx]).astype(np.float32)
        turbulences = np.interp(all_idx, valid_idx, turbulences[valid_idx]).astype(np.float32)
    else:
        global_speeds = np.zeros(N, dtype=np.float32)
        turbulences = np.zeros(N, dtype=np.float32)

    return global_speeds, global_angles, turbulences

def smooth_linear(data, fps, window_sec=0.3):
    """Smooths linear data using Savitzky-Golay filter or moving average fallback."""
    if len(data) == 0:
        return data
    window = int(fps * window_sec)
    if window % 2 == 0:
        window += 1
    window = max(5, window)
    if len(data) > window:
        if HAS_SCIPY:
            try:
                return savgol_filter(data, window, 3)
            except Exception:
                pass
        # Robust fallback using moving average with reflection padding
        pad_width = window // 2
        padded = np.pad(data, pad_width, mode="reflect")
        kernel = np.ones(window, dtype=np.float32) / float(window)
        return np.convolve(padded, kernel, mode="valid")
    return data

def smooth_circular(angles_deg, fps, window_sec=0.3, max_gap_sec=0.6):
    """Smooths circular data (degrees) using Gaussian filter on Sin/Cos components,
    bridging short dropped-frame gaps without inventing fake angles across long voids."""
    if len(angles_deg) == 0:
        return angles_deg
    
    sigma = max(1.0, (fps * window_sec) / 3.0) 
    valid = ~np.isnan(angles_deg)
    smoothed = np.full_like(angles_deg, np.nan)
    
    if not np.any(valid):
        return smoothed
        
    angles_rad = np.radians(angles_deg)
    idx = np.arange(len(angles_deg))
    valid_idx = idx[valid]
    
    cos_interp = np.interp(idx, valid_idx, np.cos(angles_rad[valid]))
    sin_interp = np.interp(idx, valid_idx, np.sin(angles_rad[valid]))
    
    cos_smooth = _gaussian_filter1d(cos_interp, sigma=sigma)
    sin_smooth = _gaussian_filter1d(sin_interp, sigma=sigma)
    
    smoothed_rad = np.arctan2(sin_smooth, cos_smooth)
    
    # Bridge short gaps (<= max_gap_sec), keep NaN only for long absences
    max_gap_frames = int(fps * max_gap_sec)
    min_dist = np.min(np.abs(idx[:, None] - valid_idx[None, :]), axis=1)
    keep_mask = min_dist <= max_gap_frames
    smoothed[keep_mask] = np.degrees(smoothed_rad)[keep_mask]
    
    return smoothed

def normalize_linear(data):
    """Normalize data to [0, 1] using 95th percentile."""
    valid_data = data[~np.isnan(data) & (data > 0)]
    if len(valid_data) == 0:
        return np.zeros_like(data)
    p95 = np.percentile(valid_data, 95)
    if p95 <= 1e-5:
        p95 = 1.0
    normed = np.clip(data / p95, 0.0, 1.0)
    return normed

def normalize_angle_shift(angles_deg, noise_threshold=30.0):
    """Calculate frame-to-frame angle shift and normalize to [0, 1] with noise threshold."""
    N = len(angles_deg)
    shifts_norm = np.zeros(N, dtype=np.float32)
    for i in range(1, N):
        a1, a2 = angles_deg[i-1], angles_deg[i]
        if np.isnan(a1) or np.isnan(a2):
            shifts_norm[i] = 0.0
            continue
        
        # Shortest distance between angles
        diff = abs((a2 - a1 + 180) % 360 - 180)
        
        if diff <= noise_threshold:
            shifts_norm[i] = 0.0
        else:
            shifts_norm[i] = (diff - noise_threshold) / (180.0 - noise_threshold)
            
    return shifts_norm

def calculate_boundary_likelihood(speed_norm, turb_norm, shift_norm,
                                  w_speed=0.40, w_shift=0.40, w_turb=0.20):
    """Calculates the Boundary Likelihood Signal."""
    # Derivative of turbulence
    turb_diff = np.abs(np.gradient(turb_norm))
    turb_diff_norm = normalize_linear(turb_diff)
    
    # Speed valley means high likelihood of an action transition
    speed_valley = 1.0 - speed_norm
    
    likelihood = w_speed * speed_valley + w_shift * shift_norm + w_turb * turb_diff_norm
    return likelihood, turb_diff_norm

def plot_timeseries(l_speeds, l_angles, l_turb, r_speeds, r_angles, r_turb, 
                    sl_speeds, sl_angles, sl_turb, sr_speeds, sr_angles, sr_turb,
                    nl_speeds, nl_turb, nl_shift, nr_speeds, nr_turb, nr_shift,
                    l_likelihood, r_likelihood, overall_likelihood, overall_peaks,
                    overall_threshold,
                    fps, output_path):
    if not HAS_MATPLOTLIB:
        print(f"  [Notice] matplotlib not installed — skipping plot to {output_path}")
        return
    N = len(l_speeds)
    x = np.arange(N)
    time_sec = x / max(fps, 1.0)

    fig, axes = plt.subplots(6, 1, figsize=(15, 24), sharex=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes

    # 1. Global Speed
    ax1.plot(time_sec, l_speeds, color="crimson", alpha=0.3, label="Left Raw")
    ax1.plot(time_sec, sl_speeds, color="crimson", linewidth=2.0, label="Left Smoothed")
    ax1.plot(time_sec, r_speeds, color="forestgreen", alpha=0.3, label="Right Raw")
    ax1.plot(time_sec, sr_speeds, color="forestgreen", linewidth=2.0, label="Right Smoothed")
    ax1.set_ylabel("Speed (px/f)", fontweight="bold")
    ax1.set_title("Global Translation Speed (Median)", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Local Turbulence
    ax2.plot(time_sec, l_turb, color="lightcoral", alpha=0.3, label="Left Raw")
    ax2.plot(time_sec, sl_turb, color="crimson", linewidth=2.0, label="Left Smoothed")
    ax2.plot(time_sec, r_turb, color="lightgreen", alpha=0.3, label="Right Raw")
    ax2.plot(time_sec, sr_turb, color="forestgreen", linewidth=2.0, label="Right Smoothed")
    ax2.set_ylabel("Turbulence (px/f)", fontweight="bold")
    ax2.set_title("Local Turbulence (MAD from Global Motion)", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Global Angle
    for y_deg in [-180, -90, 0, 90, 180]:
        ax3.axhline(y=y_deg, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
    ax3.plot(time_sec, l_angles, color="darkred", alpha=0.3, label="Left Raw")
    ax3.plot(time_sec, sl_angles, color="darkred", linewidth=2.5, label="Left Smoothed")
    ax3.plot(time_sec, r_angles, color="darkgreen", alpha=0.3, label="Right Raw")
    ax3.plot(time_sec, sr_angles, color="darkgreen", linewidth=2.5, label="Right Smoothed")
    ax3.set_ylabel("Angle (degrees)", fontweight="bold")
    ax3.set_yticks([-180, -90, 0, 90, 180])
    ax3.set_ylim(-195, 195)
    ax3.set_title("Global Motion Angle", fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Normalized Multi-Modal Signals [LEFT]
    ax4.plot(time_sec, nl_speeds, color="crimson", linewidth=2.0, label="Speed Norm")
    ax4.plot(time_sec, nl_turb, color="purple", linewidth=2.0, label="Turbulence Norm")
    ax4.plot(time_sec, nl_shift, color="darkorange", linewidth=2.0, label="Angle Shift Norm")
    ax4.set_title("Normalized Signals [0, 1] (LEFT HAND)", fontweight="bold")
    ax4.set_ylabel("Norm Score", fontweight="bold")
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    # 5. Normalized Multi-Modal Signals [RIGHT]
    ax5.plot(time_sec, nr_speeds, color="forestgreen", linewidth=2.0, label="Speed Norm")
    ax5.plot(time_sec, nr_turb, color="purple", linewidth=2.0, label="Turbulence Norm")
    ax5.plot(time_sec, nr_shift, color="darkorange", linewidth=2.0, label="Angle Shift Norm")
    ax5.set_title("Normalized Signals [0, 1] (RIGHT HAND)", fontweight="bold")
    ax5.set_ylabel("Norm Score", fontweight="bold")
    ax5.set_ylim(-0.05, 1.05)
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)

    # 6. Boundary Likelihood Fusion (Left, Right, Overall)
    ax6.plot(time_sec, l_likelihood, color="crimson", alpha=0.4, linestyle="--", label="Left Likelihood")
    ax6.plot(time_sec, r_likelihood, color="forestgreen", alpha=0.4, linestyle="--", label="Right Likelihood")
    ax6.plot(time_sec, overall_likelihood, color="black", linewidth=2.5, label="Overall Likelihood (Max)")
    
    # Plot Peaks on Overall
    ax6.plot(time_sec[overall_peaks], overall_likelihood[overall_peaks], "rx", markersize=12, markeredgewidth=2, label="Detected Boundaries")
    
    ax6.plot(time_sec, overall_threshold, color="red", linestyle=":", alpha=0.8, label="Dynamic Threshold")
    ax6.set_title("Boundary Likelihood Fusion", fontweight="bold")
    ax6.set_ylabel("Likelihood", fontweight="bold")
    ax6.set_ylim(-0.05, 1.05)
    ax6.set_xlabel("Time (seconds)", fontweight="bold")
    ax6.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Calculate decomposed motion metrics (Global Speed, Angle, Turbulence)")
    parser.add_argument("--flows", required=True, help="Path to flow .npz")
    parser.add_argument("--masks", required=True, help="Path to masks .npz")
    parser.add_argument("--output", required=True, help="Output directory to save plots and data")
    parser.add_argument("--min-speed", type=float, default=0.3, help="Minimum global speed to calculate angle (default: 0.3)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    flows, left_masks, right_masks, fps = load_data(args.flows, args.masks)
    print(f"Loaded {len(flows)} frames.")

def run_multimodal_dynamic_segmentation(
    flows, left_masks, right_masks, fps, output_dir,
    min_speed=0.5, min_distance_sec=0.5, k_std=0.7,
    noise_threshold=25.0, w_speed=0.40, w_shift=0.40, w_turb=0.20,
    erode_ksize=3,
):
    """
    Computes decomposed motion (speed, angle, turbulence), normalizes them, fuses left and right hands,
    computes a dynamic sliding window threshold, and extracts segmentation boundaries.
    
    Returns:
        boundaries: list of integer frame indices.
        left_smooth_speeds: np.ndarray
        right_smooth_speeds: np.ndarray
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    N = len(flows)

    print("  [Fusion] Computing metrics for LEFT hand (with boundary erosion, RMS energy, and interpolation)...")
    l_spds, l_angs, l_turb = compute_decomposition(flows, left_masks, min_speed=min_speed, erode_ksize=erode_ksize)
    sl_spds = smooth_linear(l_spds, fps)
    sl_turb = smooth_linear(l_turb, fps)
    sl_angs = smooth_circular(l_angs, fps)
    
    # Normalize left hand
    nl_spds = normalize_linear(sl_spds)
    nl_turb = normalize_linear(sl_turb)
    nl_shift = normalize_angle_shift(sl_angs, noise_threshold=noise_threshold)
    l_likelihood, _ = calculate_boundary_likelihood(
        nl_spds, nl_turb, nl_shift, w_speed=w_speed, w_shift=w_shift, w_turb=w_turb
    )
    
    print("  [Fusion] Computing metrics for RIGHT hand (with boundary erosion, RMS energy, and interpolation)...")
    r_spds, r_angs, r_turb = compute_decomposition(flows, right_masks, min_speed=min_speed, erode_ksize=erode_ksize)
    sr_spds = smooth_linear(r_spds, fps)
    sr_turb = smooth_linear(r_turb, fps)
    sr_angs = smooth_circular(r_angs, fps)
    
    # Normalize right hand
    nr_spds = normalize_linear(sr_spds)
    nr_turb = normalize_linear(sr_turb)
    nr_shift = normalize_angle_shift(sr_angs, noise_threshold=noise_threshold)
    r_likelihood, _ = calculate_boundary_likelihood(
        nr_spds, nr_turb, nr_shift, w_speed=w_speed, w_shift=w_shift, w_turb=w_turb
    )

    # Fusion (Merge Left and Right)
    overall_likelihood = np.maximum(l_likelihood, r_likelihood)
    
    # Compute Dynamic Threshold (Sliding Window 4 seconds)
    window_size = int(fps * 4.0)
    if window_size == 0:
        window_size = 1
    local_mean = _uniform_filter1d(overall_likelihood, size=window_size)
    mean_sq = _uniform_filter1d(overall_likelihood**2, size=window_size)
    local_std = np.sqrt(np.maximum(mean_sq - local_mean**2, 0))
    
    dynamic_threshold = local_mean + k_std * local_std
    dynamic_threshold = np.maximum(dynamic_threshold, 0.25)
    
    dist_frames = max(1, int(fps * min_distance_sec))
    overall_peaks, _ = _find_peaks_1d(overall_likelihood, height=dynamic_threshold, distance=dist_frames)
    
    # Add start and end frames
    boundaries = [0] + overall_peaks.tolist() + [N - 1]
    # Remove duplicates if any
    boundaries = sorted(list(set(boundaries)))

    # Save numeric data
    out_npz = output_dir / "decomposed_motion.npz"
    np.savez_compressed(
        out_npz,
        left_global_speeds=l_spds, left_global_angles=l_angs, left_turbulences=l_turb,
        left_smooth_speeds=sl_spds, left_smooth_angles=sl_angs, left_smooth_turbulences=sl_turb,
        right_global_speeds=r_spds, right_global_angles=r_angs, right_turbulences=r_turb,
        right_smooth_speeds=sr_spds, right_smooth_angles=sr_angs, right_smooth_turbulences=sr_turb,
        left_norm_speeds=nl_spds, left_norm_turbulences=nl_turb, left_norm_shifts=nl_shift,
        right_norm_speeds=nr_spds, right_norm_turbulences=nr_turb, right_norm_shifts=nr_shift,
        overall_likelihood=overall_likelihood, dynamic_threshold=dynamic_threshold,
        fps=fps
    )
    print(f"  [Fusion] Data saved to {out_npz}")

    # Plot
    plot_path = output_dir / "motion_decomposition_smooth_plot.png"
    print(f"  [Fusion] Generating timeseries plot -> {plot_path}")
    plot_timeseries(l_spds, l_angs, l_turb, r_spds, r_angs, r_turb,
                    sl_spds, sl_angs, sl_turb, sr_spds, sr_angs, sr_turb, 
                    nl_spds, nl_turb, nl_shift, nr_spds, nr_turb, nr_shift,
                    l_likelihood, r_likelihood, overall_likelihood, overall_peaks,
                    dynamic_threshold,
                    fps, plot_path)
    
    return boundaries, sl_spds, sr_spds

def main():
    parser = argparse.ArgumentParser(description="Calculate decomposed motion metrics (Global Speed, Angle, Turbulence)")
    parser.add_argument("--flows", required=True, help="Path to flow .npz")
    parser.add_argument("--masks", required=True, help="Path to masks .npz")
    parser.add_argument("--output", required=True, help="Output directory to save plots and data")
    parser.add_argument("--min-speed", type=float, default=0.5, help="Minimum global speed to calculate angle (default: 0.5)")
    args = parser.parse_args()

    print("Loading data...")
    flows, left_masks, right_masks, fps = load_data(args.flows, args.masks)
    print(f"Loaded {len(flows)} frames.")
    
    boundaries, sl_spds, sr_spds = run_multimodal_dynamic_segmentation(
        flows, left_masks, right_masks, fps, args.output, min_speed=args.min_speed
    )
    
    print(f"Detected {len(boundaries)-1} segments.")
    print("Done!")

if __name__ == "__main__":
    main()
