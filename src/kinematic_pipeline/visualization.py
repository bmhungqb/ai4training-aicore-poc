import json
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from action_segment_magnitude import _safe_mask
except ImportError:
    def _safe_mask(mask):
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

CLASS_COLORS = {
    "move/transfer":       (0,   165, 255),  # orange
    "adjust/align":        (255, 255,   0),  # cyan
    "process/manipulate":  (0,   255,   0),  # green
    "check/end":           (0,     0, 255),  # red
    "unknown":             (180, 180, 180),  # grey
}

def _get_class_color(cls: str) -> tuple:
    if cls in CLASS_COLORS:
        return CLASS_COLORS[cls]
    import hashlib
    h = int(hashlib.md5(cls.encode('utf-8')).hexdigest(), 16)
    r = (h & 0xFF)
    g = ((h >> 8) & 0xFF)
    b = ((h >> 16) & 0xFF)
    # Ensure it's not too dark
    if r + g + b < 200:
        r, g, b = min(255, r + 100), min(255, g + 100), min(255, b + 100)
    return (b, g, r)  # OpenCV uses BGR


def run_step5_visualize(video_path: Path, masks_path: Path, flow_path: Path,
                        boundaries: list, left_mags: np.ndarray,
                        right_mags: np.ndarray, classifications: list,
                        fps: float, output_dir: Path, args) -> None:
    """Render annotated video + final JSON report."""

    print("\n" + "=" * 70)
    print("STEP 5: Visualization")
    print("=" * 70)

    # ── Build segment lookup ──────────────────────────────────────────────────
    total_frames = len(left_mags)
    segments = _build_segment_list(boundaries, total_frames, classifications, fps)

    # ── JSON report ───────────────────────────────────────────────────────────
    report_path = output_dir / "pipe1_report.json"
    report = {
        "video":      str(video_path),
        "fps":        fps,
        "n_segments": len(segments),
        "total_cost_usd": sum(r.get("estimated_cost_usd", 0) for r in classifications),
        "segments":   segments,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Report → {report_path}")

    if getattr(args, "no_video", False):
        print("  Video output skipped (--no-video)")
        return

    # ── Video annotator ───────────────────────────────────────────────────────
    viz_path = output_dir / f"{video_path.stem}_pipe1_viz.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Cannot open video for visualization: {video_path}")
        return

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(viz_path), fourcc, vid_fps, (W, H))
    if not writer.isOpened():
        print(f"  WARNING: Cannot create writer for {viz_path}")
        cap.release()
        return

    # Load masks
    mask_data   = np.load(masks_path, allow_pickle=True)
    left_masks  = mask_data["left_masks"]
    right_masks = mask_data["right_masks"]
    frame_ind   = mask_data["frame_indices"].tolist()

    # Frame → mask index map
    frame_to_mask_idx = {int(fi): mi for mi, fi in enumerate(frame_ind)}

    # Load rich direction telemetry if available (Legacy)
    dir_bound_file = output_dir / "direction_boundaries.npy"
    dir_data = {}
    if dir_bound_file.exists():
        try:
            dir_data = np.load(dir_bound_file, allow_pickle=True).item()
        except Exception:
            dir_data = {}

    left_vectors    = dir_data.get("left_vectors")
    right_vectors   = dir_data.get("right_vectors")
    left_raw_angles = dir_data.get("left_raw_angles")
    right_raw_angles= dir_data.get("right_raw_angles")
    boundary_details= dir_data.get("boundary_details", [])

    # Load Dynamic Fusion telemetry (New)
    decomposed_file = output_dir / "decomposed_motion.npz"
    overall_likelihood = None
    dynamic_threshold = None
    if decomposed_file.exists():
        try:
            dec_data = np.load(decomposed_file)
            overall_likelihood = dec_data.get("overall_likelihood")
            dynamic_threshold = dec_data.get("dynamic_threshold")
        except Exception:
            pass

    # Boundary lookup map
    boundary_lookup = {int(b["frame"]): b for b in boundary_details if "frame" in b}
    for b in boundaries:
        if int(b) not in boundary_lookup:
            boundary_lookup[int(b)] = {"frame": int(b), "type": "ACTION_BOUNDARY", "score": 0.5, "sources": []}

    boundary_set = set(int(b) for b in boundaries)

    print(f"  Writing {total_f} frames → {viz_path}")
    t0 = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        l_center, r_center = None, None

        # ── 1. Mask overlays & Centroids ──────────────────────────────────────
        mask_idx = frame_to_mask_idx.get(frame_idx)
        if mask_idx is not None:
            l_m = _safe_mask(left_masks[mask_idx])
            r_m = _safe_mask(right_masks[mask_idx])
            if l_m is not None and l_m.any():
                frame = _draw_colored_mask(frame, l_m, (0, 0, 200), 0.35)
                l_center = _get_mask_centroid(l_m, W, H)
            if r_m is not None and r_m.any():
                frame = _draw_colored_mask(frame, r_m, (0, 200, 0), 0.35)
                r_center = _get_mask_centroid(r_m, W, H)

        # ── 2. Hand Motion Vectors & Centroid Connection ───────────────────────
        if left_vectors is not None and frame_idx < len(left_vectors) and l_center is not None:
            lu, lv = float(left_vectors[frame_idx, 0]), float(left_vectors[frame_idx, 1])
            _draw_hand_motion_arrow(frame, l_center, lu, lv, (0, 0, 255))

        if right_vectors is not None and frame_idx < len(right_vectors) and r_center is not None:
            ru, rv = float(right_vectors[frame_idx, 0]), float(right_vectors[frame_idx, 1])
            _draw_hand_motion_arrow(frame, r_center, ru, rv, (0, 255, 0))

        if l_center is not None and r_center is not None:
            dist_px = float(np.linalg.norm(np.array(l_center) - np.array(r_center)))
            cv2.line(frame, l_center, r_center, (140, 140, 140), 1, cv2.LINE_AA)
            mid_pt = ((l_center[0] + r_center[0]) // 2, (l_center[1] + r_center[1]) // 2 - 8)
            cv2.putText(frame, f"D={dist_px:.0f}px", mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

        # ── 3. Magnitude bar (top of frame) ───────────────────────────────────
        l_spd = left_mags[frame_idx] if frame_idx < len(left_mags) else 0.0
        r_spd = right_mags[frame_idx] if frame_idx < len(right_mags) else 0.0
        frame = _draw_magnitude_bar(frame, l_spd, r_spd, W)

        # ── 4. Boundary flash border ──────────────────────────────────────────
        if frame_idx in boundary_set:
            cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 220, 255), 5)

        # ── 5. Boundary Notification Card (alert for 14 frames) ───────────────
        _draw_boundary_notification_card(frame, frame_idx, boundary_lookup, W, H, display_duration=14)

        # ── 6. Top Telemetry HUD (Glassmorphism card) ─────────────────────────
        l_ang = left_raw_angles[frame_idx] if left_raw_angles is not None and frame_idx < len(left_raw_angles) else np.nan
        r_ang = right_raw_angles[frame_idx] if right_raw_angles is not None and frame_idx < len(right_raw_angles) else np.nan
        _draw_top_hud(frame, frame_idx, total_f, fps, l_spd, r_spd, l_ang, r_ang, W)

        # ── 6.5 Dynamic Segmentation Engine HUD ───────────────────────────────
        if overall_likelihood is not None and dynamic_threshold is not None:
            l_val = overall_likelihood[frame_idx] if frame_idx < len(overall_likelihood) else 0.0
            t_val = dynamic_threshold[frame_idx] if frame_idx < len(dynamic_threshold) else 0.5
            _draw_segmentation_engine_hud(frame, l_val, t_val, W)

        # ── 7. Segment Progress Banner ────────────────────────────────────────
        seg = _get_segment_for_frame(frame_idx, segments)
        if seg:
            _draw_segment_banner_with_progress(frame, frame_idx, seg, fps, H, W)

        # ── 8. Bottom Mini Timeline Scrubber Strip ────────────────────────────
        _draw_bottom_timeline(frame, frame_idx, total_f, segments, W, H, bar_h=10)

        writer.write(frame)

        # ── 9. Pause effect at the end of each segment ────────────────────────
        if seg and frame_idx == seg["end_frame"] and frame_idx < total_f - 1:
            pause_duration_sec = 2.0
            pause_frames = int(vid_fps * pause_duration_sec)
            
            # Create a dimmed pause frame with text
            pause_frame = frame.copy()
            overlay = pause_frame.copy()
            cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, pause_frame, 0.6, 0, pause_frame)
            
            text = f"END OF {seg.get('class', 'UNKNOWN').upper()}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.65
            thickness = 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            tx = (W - text_size[0]) // 2
            ty = (H + text_size[1]) // 2
            
            cv2.putText(pause_frame, text, (max(tx, 10), ty), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
            cv2.putText(pause_frame, "Loading next step...", (max(tx, 10) + 20, ty + 40), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            
            for _ in range(pause_frames):
                writer.write(pause_frame)

        if (frame_idx + 1) % 60 == 0:
            el = time.time() - t0
            print(f"  {frame_idx+1}/{total_f} frames | {(frame_idx+1)/el:.1f} fps")

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[Step 5] Video → {viz_path}")


def _angle_to_direction_str(angle_deg: float, speed: float) -> str:
    """Convert angle into compass arrow and text."""
    if speed < 0.3 or not np.isfinite(angle_deg):
        return "IDLE (Stationary)"
    ang = (angle_deg + 180.0) % 360.0 - 180.0
    if -22.5 <= ang < 22.5:
        return f"{ang:+.0f}° (→ Right)"
    elif 22.5 <= ang < 67.5:
        return f"{ang:+.0f}° (↘ Down-Right)"
    elif 67.5 <= ang < 112.5:
        return f"{ang:+.0f}° (↓ Down)"
    elif 112.5 <= ang < 157.5:
        return f"{ang:+.0f}° (↙ Down-Left)"
    elif ang >= 157.5 or ang < -157.5:
        return f"{ang:+.0f}° (← Left)"
    elif -157.5 <= ang < -112.5:
        return f"{ang:+.0f}° (↖ Up-Left)"
    elif -112.5 <= ang < -67.5:
        return f"{ang:+.0f}° (↑ Up)"
    else:
        return f"{ang:+.0f}° (↗ Up-Right)"


def _get_mask_centroid(mask: np.ndarray | None, W: int, H: int) -> tuple[int, int] | None:
    """Compute (cx, cy) pixel centroid of boolean mask."""
    if mask is None or not mask.any():
        return None
    if mask.shape != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    y_idxs, x_idxs = np.where(mask)
    if len(x_idxs) == 0:
        return None
    return (int(np.mean(x_idxs)), int(np.mean(y_idxs)))


def _draw_hand_motion_arrow(frame: np.ndarray, center: tuple[int, int], u: float, v: float, color: tuple, scale: float = 8.0, min_speed: float = 0.3):
    """Draw live dynamic motion vector arrow from centroid."""
    spd = np.sqrt(u * u + v * v)
    if spd >= min_speed:
        H, W = frame.shape[:2]
        x2 = int(np.clip(center[0] + u * scale, 0, W - 1))
        y2 = int(np.clip(center[1] + v * scale, 0, H - 1))
        cv2.arrowedLine(frame, center, (x2, y2), (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.35)
        cv2.circle(frame, center, 4, color, -1, cv2.LINE_AA)


def _draw_top_hud(
    frame: np.ndarray, frame_idx: int, total_frames: int, fps: float,
    l_spd: float, r_spd: float, l_ang: float, r_ang: float, W: int
):
    """Sleek dark glassmorphism HUD panel in top-left."""
    t_sec = frame_idx / max(fps, 1.0)
    hud_w, hud_h = 360, 84
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 14), (10 + hud_w, 14 + hud_h), (10, 15, 22), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (10, 14), (10 + hud_w, 14 + hud_h), (80, 90, 100), 1, cv2.LINE_AA)

    # Line 1: Frame & Time
    cv2.putText(frame, f"Frame {frame_idx:04d}/{total_frames} | t={t_sec:05.2f}s (FPS:{fps:.1f})",
                (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # Line 2: Left hand
    l_dir_str = _angle_to_direction_str(l_ang, l_spd)
    cv2.putText(frame, f"L-Hand: {l_spd:4.2f} px/f | {l_dir_str}",
                (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 160, 255), 1, cv2.LINE_AA)

    # Line 3: Right hand
    r_dir_str = _angle_to_direction_str(r_ang, r_spd)
    cv2.putText(frame, f"R-Hand: {r_spd:4.2f} px/f | {r_dir_str}",
                (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 255, 160), 1, cv2.LINE_AA)


def _draw_segmentation_engine_hud(frame: np.ndarray, likelihood: float, threshold: float, W: int):
    """Dynamic HUD for Segmentation Engine (Likelihood vs Threshold) on top-right."""
    hud_w, hud_h = 280, 60
    margin_x, margin_y = 10, 14
    hud_x = W - hud_w - margin_x
    hud_y = margin_y
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (10, 15, 22), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    
    # Border: Red if Likelihood > Threshold (triggering cut), else normal grey
    is_trigger = likelihood >= threshold
    border_color = (0, 0, 255) if is_trigger else (80, 90, 100)
    cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), border_color, 2 if is_trigger else 1, cv2.LINE_AA)

    # Title
    title_col = (0, 100, 255) if is_trigger else (200, 200, 200)
    title_text = "SEGMENTATION TRIGGERED!" if is_trigger else "Segmentation Engine"
    cv2.putText(frame, title_text, (hud_x + 10, hud_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, title_col, 1 if not is_trigger else 2, cv2.LINE_AA)

    # Bar chart background
    bar_x = hud_x + 10
    bar_y = hud_y + 35
    bar_w = hud_w - 20
    bar_h_rect = 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_rect), (40, 40, 40), -1)
    
    # Likelihood Fill
    fill_w = int(np.clip(likelihood, 0, 1) * bar_w)
    fill_col = (0, 0, 255) if is_trigger else (0, 200, 255)  # Orange to Red
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h_rect), fill_col, -1)
    
    # Threshold Marker
    thresh_x = bar_x + int(np.clip(threshold, 0, 1) * bar_w)
    cv2.line(frame, (thresh_x, bar_y - 4), (thresh_x, bar_y + bar_h_rect + 4), (255, 255, 255), 2, cv2.LINE_AA)
    
    # Labels
    cv2.putText(frame, f"L:{likelihood:.2f}", (bar_x + 2, bar_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"T:{threshold:.2f}", (thresh_x - 12, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_boundary_notification_card(
    frame: np.ndarray, frame_idx: int, boundary_lookup: dict, W: int, H: int, display_duration: int = 14
):
    """If current frame is near a boundary trigger, render alert card."""
    active_b = None
    for b_idx in range(frame_idx, max(-1, frame_idx - display_duration), -1):
        if b_idx in boundary_lookup:
            active_b = boundary_lookup[b_idx]
            break

    if active_b is not None:
        b_type = active_b.get("type", "ACTION_BOUNDARY")
        score = active_b.get("score", 0.0)
        sources = active_b.get("sources", [])
        src_str = " + ".join(sources).replace("DIRECTION_", "DIR_").replace("SPEED_VALLEY", "SPEED")

        card_w, card_h = 440, 52
        card_x = (W - card_w) // 2
        card_y = 50

        overlay = frame.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (20, 25, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (0, 215, 255), 2, cv2.LINE_AA)
        title_text = f"★ BOUNDARY: [{b_type}] (Score: {score:.2f})"
        cv2.putText(frame, title_text, (card_x + 12, card_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2, cv2.LINE_AA)
        detail_text = f"Triggers: {src_str}"
        cv2.putText(frame, detail_text, (card_x + 12, card_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 220, 255), 1, cv2.LINE_AA)


def _draw_segment_banner_with_progress(
    frame: np.ndarray, frame_idx: int, seg: dict, fps: float, H: int, W: int, banner_h: int = 42
):
    """Draw rich bottom banner with action class, elapsed time, and progress bar."""
    y_top = H - 10 - banner_h  # above the 10px timeline strip
    cls = seg.get("class", "unknown")
    color = _get_class_color(cls)

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_top), (W, H - 10), (15, 15, 22), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Left color strip
    cv2.rectangle(frame, (0, y_top), (8, H - 10), color, -1)

    seg_num = seg.get("segment", 0) + 1
    s_f = seg["start_frame"]
    e_f = seg["end_frame"]
    seg_len_f = max(e_f - s_f + 1, 1)
    elapsed_in_seg_f = frame_idx - s_f
    progress = np.clip(elapsed_in_seg_f / seg_len_f, 0.0, 1.0)

    cur_t_s = elapsed_in_seg_f / max(fps, 1.0)
    dur_t_s = seg_len_f / max(fps, 1.0)

    label_main = f"[{cls.upper()}]"
    time_str = f"{cur_t_s:04.1f}s / {dur_t_s:04.1f}s"

    cv2.putText(frame, label_main, (20, y_top + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(frame, time_str, (W - 170, y_top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)

    # Mini Progress Bar
    prog_bar_w = 170
    prog_x1 = W - 180
    prog_y = y_top + 28
    prog_fill_w = int(progress * prog_bar_w)
    cv2.rectangle(frame, (prog_x1, prog_y), (prog_x1 + prog_bar_w, prog_y + 4), (60, 60, 60), -1)
    cv2.rectangle(frame, (prog_x1, prog_y), (prog_x1 + prog_fill_w, prog_y + 4), color, -1)


def _draw_bottom_timeline(frame: np.ndarray, frame_idx: int, total_frames: int, segments: list, W: int, H: int, bar_h: int = 10):
    """Draw a multi-colored SOP timeline strip at the very bottom with a live playhead cursor."""
    y_top = H - bar_h
    cv2.rectangle(frame, (0, y_top), (W, H), (25, 25, 30), -1)

    for seg in segments:
        s_f = seg["start_frame"]
        e_f = seg["end_frame"]
        cls = seg.get("class", "unknown")
        color = _get_class_color(cls)
        x1 = int((s_f / max(total_frames, 1)) * W)
        x2 = int(((e_f + 1) / max(total_frames, 1)) * W)
        x2 = max(x2, x1 + 1)
        cv2.rectangle(frame, (x1, y_top), (x2, H), color, -1)
        cv2.line(frame, (x1, y_top), (x1, H), (15, 15, 15), 1)

    # Live playhead cursor
    cur_x = int((frame_idx / max(total_frames, 1)) * W)
    cv2.rectangle(frame, (max(0, cur_x - 2), y_top - 2), (min(W - 1, cur_x + 2), H), (255, 255, 255), -1)


def _draw_colored_mask(frame: np.ndarray, mask: np.ndarray,
                       color: tuple, alpha: float) -> np.ndarray:
    """Blend a boolean mask as a translucent color overlay."""
    H, W = frame.shape[:2]
    mh, mw = mask.shape
    if (mh, mw) != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay = np.zeros_like(frame)
    overlay[mask] = color
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)


def _draw_magnitude_bar(frame: np.ndarray, l_mag: float, r_mag: float,
                        W: int, bar_h: int = 8, max_mag: float = 5.0) -> np.ndarray:
    """Draw two thin magnitude bars at the very top of the frame."""
    frame = frame.copy()
    half = W // 2
    l_px = int(np.clip(l_mag / max_mag, 0, 1) * half)
    r_px = int(np.clip(r_mag / max_mag, 0, 1) * half)
    cv2.rectangle(frame, (0, 0), (l_px, bar_h), (80, 80, 255), -1)        # left = blue-ish
    cv2.rectangle(frame, (half, 0), (half + r_px, bar_h), (80, 255, 80), -1)  # right = green-ish
    return frame


def _build_segment_list(boundaries: list, total_frames: int, classifications: list, fps: float) -> list:
    """Merge boundary info with classification results into seamless [0..total_frames] segments."""
    edges = [0] + sorted([int(b) for b in boundaries if 0 < int(b) < total_frames]) + [total_frames]
    edges = sorted(list(dict.fromkeys(edges)))
    segments = []
    for i in range(len(edges) - 1):
        start_f = edges[i]
        end_f   = edges[i + 1] - 1
        cls_entry = next((c for c in classifications if c.get("segment") == i), {})
        segments.append({
            "segment":      i,
            "start_frame":  start_f,
            "end_frame":    end_f,
            "start_time_s": round(start_f / fps, 3),
            "end_time_s":   round(end_f   / fps, 3),
            "duration_s":   round((end_f - start_f + 1) / fps, 3),
            "class":        cls_entry.get("class", "unknown"),
            "description":  cls_entry.get("description", ""),
        })
    return segments


def _get_segment_for_frame(frame_idx: int, segments: list) -> dict | None:
    """Return the segment dict that contains frame_idx, or None."""
    for seg in segments:
        if seg["start_frame"] <= frame_idx <= seg["end_frame"]:
            return seg
    return None
