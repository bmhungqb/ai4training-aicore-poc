import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from config import PipelineConfig

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: SAM3 imports
# ─────────────────────────────────────────────────────────────────────────────
try:
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    from segment import (
        normalize_mask,
        run_sam3_frame,
    )
except ImportError as e:
    print(f"Warning: SAM3 import error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: SEA-RAFT imports
# ─────────────────────────────────────────────────────────────────────────────
try:
    from searaft import (
        SeaRaftModel,
        safe_mask_array,
        resize_mask as searaft_resize_mask,
    )
except ImportError as e:
    print(f"Warning: SEA-RAFT import error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: action_segment_magnitude imports
# ─────────────────────────────────────────────────────────────────────────────
from action_segment_magnitude import (
    compute_hand_magnitudes,
    smooth_magnitudes,
    find_peaks_and_valleys,
    find_joint_boundaries,
    plot_magnitudes,
)

from calculate_direction_magnitude import run_multimodal_dynamic_segmentation

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: classify_action imports
# ─────────────────────────────────────────────────────────────────────────────
from classify_action import classify_segmented_actions


def load_roi_mask(mask_path, height: int, width: int):
    """Load a ROI mask image (white/nonzero=keep, black=ignore) and resize/binarize
    it to (height, width) bool. Returns None if mask_path is None or unreadable
    (caller should treat that as full-frame, no masking).

    Shared by Step 1 (SAM3 hand detection) and Step 2 (SEA-RAFT optical flow) so
    another worker/expert visible in the same frame doesn't get picked up as noise.
    """
    if mask_path is None:
        return None
    mask_path = Path(mask_path)
    if not mask_path.exists():
        print(f"  WARNING: mask file not found: {mask_path} — running on full frame")
        return None
    raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        print(f"  WARNING: could not read mask file: {mask_path} — running on full frame")
        return None
    if raw.shape[:2] != (height, width):
        raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST)
    return raw > 127


def apply_roi_mask(frame_rgb, roi_mask):
    """Zero out everything outside roi_mask in an RGB/BGR frame. No-op if roi_mask is None."""
    if roi_mask is None:
        return frame_rgb
    out = frame_rgb.copy()
    out[~roi_mask] = 0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: SAM3 segmentation
# ─────────────────────────────────────────────────────────────────────────────

def run_step1_segmentation(video_path: Path, output_dir: Path, config: PipelineConfig) -> Path:
    """Run SAM3 on the video and save masks .npz. Returns the masks path."""
    masks_path = output_dir / f"{video_path.stem}_masks.npz"

    if masks_path.exists() and not config.force:
        print(f"[Step 1] Masks already exist → skipping ({masks_path})")
        return masks_path

    print("\n" + "=" * 70)
    print("STEP 1: SAM3 Hand Segmentation")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Load SAM3
    print("Loading SAM3 model...")
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device=device, dtype=dtype)
    model.eval()
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

    def _create_fresh_session():
        sess = processor.init_video_session(
            inference_device=device, processing_device="cpu",
            video_storage_device="cpu", dtype=dtype,
        )
        sess = processor.add_text_prompt(
            inference_session=sess,
            text=["left human hand", "right human hand"],
        )
        return sess

    inference_session = _create_fresh_session()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    reset_interval = 1 if config.frame_by_frame else max(1, config.sam_reset_interval)
    mode_str = "Frame-by-Frame (Stateless)" if config.frame_by_frame else f"Sliding Memory (Reset every {reset_interval} frames)"
    print(f"Video: {width}x{height} | {frame_count} frames | {fps:.2f} FPS | SAM3 Mode: {mode_str}")

    roi_mask = load_roi_mask(getattr(config, "mask", None), height, width)
    print(f"ROI mask: {'applied (' + str(getattr(config, 'mask', None)) + ')' if roi_mask is not None else 'none (full frame)'}")

    left_masks, right_masks   = [], []
    left_scores, right_scores = [], []
    frame_indices             = []
    object_prompt_map         = {}

    frame_idx = 0
    processed = 0
    t0 = time.time()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        cur_idx = frame_idx
        frame_idx += 1

        if cur_idx % config.frame_step != 0:
            continue
        if config.max_frames is not None and processed >= config.max_frames:
            break

        # Periodic memory reset to prevent GPU VRAM accumulation on long videos
        if processed > 0 and processed % reset_interval == 0:
            del inference_session
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            inference_session = _create_fresh_session()
            object_prompt_map = {}

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = apply_roi_mask(frame_rgb, roi_mask)

        try:
            best_dets, object_prompt_map = run_sam3_frame(
                model=model, processor=processor,
                inference_session=inference_session,
                frame_rgb=frame_rgb, device=device,
                object_prompt_map=object_prompt_map,
                sam_threshold=config.sam_threshold,
            )
        except Exception as e:
            print(f"  Frame {cur_idx} failed: {e}")
            best_dets = {"left": None, "right": None}

        left_det  = best_dets.get("left")
        right_det = best_dets.get("right")

        l_mask = normalize_mask(left_det.get("mask"),  height, width) if left_det  else None
        r_mask = normalize_mask(right_det.get("mask"), height, width) if right_det else None
        l_score = float(left_det.get("score",  0.0)) if left_det  else 0.0
        r_score = float(right_det.get("score", 0.0)) if right_det else 0.0

        left_masks.append(l_mask);  right_masks.append(r_mask)
        left_scores.append(l_score); right_scores.append(r_score)
        frame_indices.append(cur_idx)
        processed += 1

        if processed % 20 == 0:
            elapsed = time.time() - t0
            print(f"  Frame {cur_idx:05d} | {processed}/{frame_count} | "
                  f"{processed/elapsed:.1f} fps | L={l_score:.2f} R={r_score:.2f}")

        del frame_rgb, frame_bgr

    cap.release()

    # Save
    np.savez_compressed(
        masks_path,
        left_masks=np.array(left_masks,   dtype=object),
        right_masks=np.array(right_masks,  dtype=object),
        left_scores=np.asarray(left_scores,  dtype=np.float32),
        right_scores=np.asarray(right_scores, dtype=np.float32),
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        fps=np.float32(fps), width=np.int32(width), height=np.int32(height),
    )
    print(f"[Step 1] Saved masks → {masks_path}")

    del inference_session, model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return masks_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: SEA-RAFT optical flow
# ─────────────────────────────────────────────────────────────────────────────

def run_step2_optical_flow(video_path: Path, masks_path: Path,
                           output_dir: Path, config: PipelineConfig) -> Path:
    """Run SEA-RAFT on consecutive frame pairs and save flow .npz. Returns flow path."""
    flow_path = output_dir / f"{video_path.stem}_flow.npz"

    if flow_path.exists() and not config.force:
        print(f"[Step 2] Flow already exists → skipping ({flow_path})")
        return flow_path

    print("\n" + "=" * 70)
    print("STEP 2: SEA-RAFT Optical Flow Extraction")
    print("=" * 70)

    # Load masks
    mask_data     = np.load(masks_path, allow_pickle=True)
    left_masks    = mask_data["left_masks"]
    right_masks   = mask_data["right_masks"]
    frame_indices = mask_data["frame_indices"].tolist()
    fps           = float(mask_data["fps"])

    # Load SEA-RAFT
    raft = SeaRaftModel(
        model_name=config.raft_model,
        device=config.raft_device,
        iters=config.raft_iters,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_mask_native = load_roi_mask(getattr(config, "mask", None), native_h, native_w)
    print(f"ROI mask: {'applied (' + str(getattr(config, 'mask', None)) + ')' if roi_mask_native is not None else 'none (full frame)'}")

    scale = config.resize_scale

    all_flows = []
    t0 = time.time()

    prev_frame_idx = -1
    prev_frame_img = None

    for i in range(len(frame_indices) - 1):
        fi_a = int(frame_indices[i])
        fi_b = int(frame_indices[i + 1])

        # Read frame A (reuse if consecutive)
        if prev_frame_idx == fi_a and prev_frame_img is not None:
            f_a = prev_frame_img
            ret_a = True
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi_a)
            ret_a, f_a = cap.read()

        # Read frame B
        if fi_b == fi_a + 1 and ret_a:
            ret_b, f_b = cap.read()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi_b)
            ret_b, f_b = cap.read()

        prev_frame_idx = fi_b
        prev_frame_img = f_b

        if not ret_a or not ret_b:
            all_flows.append(np.zeros((f_a.shape[0], f_a.shape[1], 2), np.float32)
                             if ret_a and f_a is not None else np.zeros((1, 1, 2), np.float32))
            continue

        # Resize if needed
        if scale != 1.0:
            new_w = int(f_a.shape[1] * scale)
            new_h = int(f_a.shape[0] * scale)
            f_a = cv2.resize(f_a, (new_w, new_h))
            f_b = cv2.resize(f_b, (new_w, new_h))

        H, W = f_a.shape[:2]

        # ROI mask, resized to this pair's working resolution — zero out pixels
        # outside the ROI so another worker/expert in frame can't contribute flow
        # even if their hand slipped past Step 1's (already ROI-restricted) detection
        roi_mask = None
        if roi_mask_native is not None:
            roi_mask = cv2.resize(roi_mask_native.astype(np.uint8), (W, H),
                                  interpolation=cv2.INTER_NEAREST) > 0
            f_a = apply_roi_mask(f_a, roi_mask)
            f_b = apply_roi_mask(f_b, roi_mask)

        # Combined mask for frame A (background masked out)
        l_mask = safe_mask_array(left_masks[i])
        r_mask = safe_mask_array(right_masks[i])
        combined_mask = None
        if l_mask is not None or r_mask is not None:
            combined_mask = np.zeros((H, W), dtype=bool)
            if l_mask is not None:
                resized_l = searaft_resize_mask(l_mask, W, H)
                if resized_l is not None:
                    combined_mask |= resized_l
            if r_mask is not None:
                resized_r = searaft_resize_mask(r_mask, W, H)
                if resized_r is not None:
                    combined_mask |= resized_r
        if roi_mask is not None and combined_mask is not None:
            combined_mask &= roi_mask

        try:
            # compute_masked_flow takes BGR frames and converts internally
            flow = raft.compute_masked_flow(f_a, f_b, combined_mask)
        except Exception as e:
            print(f"  SEA-RAFT failed on pair {fi_a}→{fi_b}: {e}")
            flow = np.zeros((f_a.shape[0], f_a.shape[1], 2), np.float32)

        all_flows.append(flow)

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  Pair {i+1}/{len(frame_indices)-1} | {(i+1)/elapsed:.1f} pairs/s")

    cap.release()

    # Stack + save
    # Pad last frame by duplicating last flow
    if all_flows:
        all_flows.append(all_flows[-1].copy())
    flow_arr = np.array(all_flows, dtype=np.float32)  # [N, H, W, 2]

    np.savez_compressed(flow_path, flow=flow_arr, fps=np.float32(fps))
    print(f"[Step 2] Saved flow → {flow_path}  shape={flow_arr.shape}")

    del raft
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return flow_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Action segmentation by magnitude
# ─────────────────────────────────────────────────────────────────────────────

def _load_flow_and_masks(flow_path: Path, masks_path: Path):
    """Helper: load flow .npz and masks .npz."""
    flow_data = np.load(flow_path)
    mask_data = np.load(masks_path, allow_pickle=True)
    flows       = flow_data["flow"]
    left_masks  = mask_data["left_masks"]
    right_masks = mask_data["right_masks"]
    fps         = float(flow_data["fps"])
    return flows, left_masks, right_masks, fps


def run_step3_segmentation(video_path: Path, flow_path: Path, masks_path: Path,
                           output_dir: Path, config: PipelineConfig):
    """Detect action boundaries using Dynamic Multi-Modal Fusion (Speed, Turbulence, Direction). Returns (boundaries, left_mags_s, right_mags_s, fps)."""
    boundaries_path = output_dir / "action_boundaries_dynamic.npy"

    should_recompute = config.force or config.recompute_segmentation
    if boundaries_path.exists() and not should_recompute:
        d = np.load(boundaries_path, allow_pickle=True).item()
        b = d.get("boundaries")
        if b:
            print(f"[Step 3] Dynamic multi-modal boundaries already exist → skipping ({len(b)-1} steps)")
            return b, d["left_mags"], d["right_mags"], d["fps"]

    print("\n" + "=" * 70)
    print("STEP 3: Action Segmentation by Dynamic Multi-Modal Fusion")
    print("=" * 70)

    flows, left_masks, right_masks, fps = _load_flow_and_masks(flow_path, masks_path)

    # Run New Dynamic Fusion (calculate_direction_magnitude.py)
    try:
        boundaries, left_mags_s, right_mags_s = run_multimodal_dynamic_segmentation(
            flows=flows,
            left_masks=left_masks,
            right_masks=right_masks,
            fps=fps,
            output_dir=output_dir,
            min_speed=config.min_speed
        )
        print(f"\n  ★ [Step 3] Fused {len(boundaries)-1} Segments using Dynamic Thresholds.")
    except Exception as e:
        print(f"  [Step 3] Dynamic Fusion failed: {e}")
        # Fallback empty
        boundaries = [0, len(flows)-1]
        left_mags_s, right_mags_s = np.zeros(len(flows)), np.zeros(len(flows))

    # Save
    np.save(boundaries_path, {
        "boundaries": boundaries,
        "left_mags":  left_mags_s,
        "right_mags": right_mags_s,
        "fps":        fps,
    })

    return boundaries, left_mags_s, right_mags_s, fps


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: VLM Classification per segment
# ─────────────────────────────────────────────────────────────────────────────

def run_step4_classify(video_path: Path, boundaries: list, total_frames: int, fps: float,
                       output_dir: Path, config: PipelineConfig) -> list:
    """
    For each action segment, sample frames and classify using classify_action.py.
    Returns list of classification dicts.
    """
    classify_path = output_dir / "classify_results.json"
    should_recompute = config.force or config.recompute_segmentation
    if classify_path.exists() and not should_recompute:
        print(f"[Step 4] Classifications already exist → skipping ({classify_path})")
        with open(classify_path, encoding="utf-8") as f:
            return json.load(f)

    if config.no_classify:
        print("[Step 4] Skipped (--no-classify)")
        return []

    print("\n" + "=" * 70)
    print("STEP 4: VLM Action Classification")
    print("=" * 70)

    min_f = config.classify_min_frames
    max_f = config.classify_max_frames
    
    motion_file = output_dir / "direction_boundaries.npy"
    motion_data = None
    if motion_file.exists():
        motion_data = np.load(motion_file, allow_pickle=True).item()

    results = classify_segmented_actions(
        video_path=video_path,
        boundaries=boundaries,
        motion_data=motion_data,
        fps=fps,
        min_frames=min_f,
        max_frames=max_f,
        model=config.vlm_model,
        output_file=classify_path,
    )
    return results
