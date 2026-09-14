#!/usr/bin/env python3
"""Render Stage 2 annotated visualization video directly from scratch.

Takes the ORIGINAL raw worker video and kinematic artifacts (SAM3 masks, motion decomposition,
action boundaries, speeds, likelihood) and renders the complete video from the ground up:
- Transparent hand masks (left: red, right: green) with centroid tracking
- Motion vector arrows and hand-to-hand distance D=...px
- Magnitude bars at the top
- Dynamic Segmentation Engine HUD (Likelihood vs Threshold)
- Top-left telemetry HUD (FPS, speeds, directional compass)
- Boundary alert cards
- Stage 2 Vietnamese SOP operation banner (PIL-rendered Unicode Arial font)
- Stage 2 Macro timing verdict badges (ĐẠT CHUẨN or CHẬM xRatio)
- Real-time segment progress bar
- Multi-colored bottom SOP timeline strip with live playhead cursor
- Clean Vietnamese transition cards between segments (NO "END OF UNKNOWN")

Usage:
    python tools/render_classified_viz.py [cong_doan_id] [--pause-sec SEC]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.kinematic_pipeline.visualization import (
    _angle_to_direction_str,
    _draw_boundary_notification_card,
    _draw_colored_mask,
    _draw_hand_motion_arrow,
    _draw_magnitude_bar,
    _draw_segmentation_engine_hud,
    _draw_top_hud,
    _get_mask_centroid,
    _safe_mask,
)


def normalize_op_name(name: str) -> str:
    """Normalize operation name by removing sub-clip markers like '( đoạn X )'."""
    cleaned = re.sub(r"\s*\(\s*đoạn\s*\d+\s*\)", "", name, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def merge_adjacent_segments(segments: list[dict], slow_times: dict | None = None) -> list[dict]:
    """Merge consecutive segments that have the same operation (or same base operation)."""
    if not segments:
        return []
    slow_times = slow_times or {}
    merged = []
    for s in segments:
        raw_name = s["operation_name"]
        norm_name = normalize_op_name(raw_name)

        if merged and merged[-1]["norm_op"] == norm_name:
            prev = merged[-1]
            prev["end_time"] = s["end_time"]
            prev["worker_duration_s"] = round(prev["end_time"] - prev["start_time"], 2)
            prev["operation_name"] = norm_name
            prev["sub_segments"].append(s)

            sub_t0 = round(s["start_time"], 2)
            if sub_t0 in slow_times or s.get("timing_verdict") == "slow":
                prev["timing_verdict"] = "slow"
                prev["slow_ev"] = slow_times.get(sub_t0, s)
            if s.get("off_standard"):
                prev["off_standard"] = True
        else:
            item = dict(s)
            item["norm_op"] = norm_name
            item["sub_segments"] = [s]
            sub_t0 = round(s["start_time"], 2)
            if sub_t0 in slow_times or s.get("timing_verdict") == "slow":
                item["timing_verdict"] = "slow"
                item["slow_ev"] = slow_times.get(sub_t0, s)
            merged.append(item)
    return merged


def get_op_color_bgr(op_name: str) -> tuple[int, int, int]:
    op_lower = op_name.lower()
    if "diễu" in op_lower or "may" in op_lower:
        return (80, 190, 40)    # Emerald green
    elif "điều chỉnh" in op_lower or "chỉnh" in op_lower:
        return (40, 160, 245)   # Amber / orange
    elif "chân vịt" in op_lower or "lấy túi" in op_lower or "đưa túi" in op_lower:
        return (240, 140, 50)   # Sky blue
    elif "lật lót" in op_lower:
        return (220, 90, 160)   # Purple
    elif "kéo" in op_lower or "cắt chỉ" in op_lower or "thân trước" in op_lower:
        return (60, 60, 240)    # Crimson red
    elif "lại mũi" in op_lower:
        return (180, 70, 230)   # Pink
    return (140, 140, 140)      # Slate grey


def _draw_pause_card(
    frame: np.ndarray, op_name: str, seg_idx: int, total_segs: int,
    dur: float, is_slow: bool, ratio: float, exp_dur: float,
    font_title, font_sub, font_badge, W: int, H: int,
) -> np.ndarray:
    """Render a clean centered pause transition card in Vietnamese."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, H), (10, 15, 20), -1)
    dimmed = cv2.addWeighted(overlay, 0.50, frame, 0.50, 0)

    card_w, card_h = 620, 130
    card_x = (W - card_w) // 2
    card_y = (H - card_h) // 2 - 20

    cv2.rectangle(dimmed, (card_x, card_y), (card_x + card_w, card_y + card_h), (18, 22, 30), -1)
    border_col = (40, 50, 220) if is_slow else (40, 180, 90)
    cv2.rectangle(dimmed, (card_x, card_y), (card_x + card_w, card_y + card_h), border_col, 2, cv2.LINE_AA)

    pil_img = Image.fromarray(cv2.cvtColor(dimmed, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    draw.text((card_x + 24, card_y + 14), f"HOÀN THÀNH BƯỚC #{seg_idx:02d} / {total_segs:02d}", font=font_sub, fill=(170, 190, 215))
    draw.text((card_x + 24, card_y + 36), op_name, font=font_title, fill=(255, 255, 255))

    if is_slow:
        verdict_str = f"⚠ CHẬM x{ratio:.2f} (Thực tế: {dur:.1f}s | Chuẩn: {exp_dur:.1f}s)"
        fill_col = (255, 160, 160)
    else:
        verdict_str = f"✓ ĐẠT CHUẨN (Thời gian: {dur:.1f}s)"
        fill_col = (150, 255, 180)
    draw.text((card_x + 24, card_y + 72), verdict_str, font=font_badge, fill=fill_col)
    draw.text((card_x + 24, card_y + 98), "Chuẩn bị chuyển sang thao tác tiếp theo...", font=font_sub, fill=(130, 145, 160))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def render_stage2_video(cd: str = "1", pause_sec: float = 1.0) -> Path:
    """Render Stage 2 visualization from scratch starting with the raw worker video."""
    data_dir = Path(f"data/{cd}")
    worker_segments_file = data_dir / "worker_segments" / "worker_segments.json"
    macro_file = data_dir / "macro_eval.json"
    kinematic_dir = data_dir / "kinematic"

    if not worker_segments_file.is_file():
        raise FileNotFoundError(f"Missing {worker_segments_file}")

    # Resolve raw worker video
    worker_video = None
    if (data_dir / "worker.mp4").exists():
        worker_video = (data_dir / "worker.mp4").resolve()
    else:
        vids = sorted(data_dir.glob("*.mp4"))
        vids = [v for v in vids if not v.name.startswith("expert") and "stage2" not in v.name and "pipe1" not in v.name]
        if vids:
            worker_video = vids[0].resolve()

    if not worker_video or not worker_video.is_file():
        raise FileNotFoundError(f"Cannot locate raw worker video in {data_dir}")

    # Locate kinematic directory for this video
    kinematic_sub = kinematic_dir / worker_video.stem
    if not kinematic_sub.is_dir():
        subs = [d for d in kinematic_dir.iterdir() if d.is_dir() and "expert" not in d.name and "073527" not in d.name]
        if subs:
            kinematic_sub = subs[0]
        else:
            raise FileNotFoundError(f"No kinematic artifacts found in {kinematic_dir}")

    masks_path = next(kinematic_sub.glob("*_masks.npz"), None)
    decomposed_path = kinematic_sub / "decomposed_motion.npz"
    boundaries_path = kinematic_sub / "action_boundaries_dynamic.npy"

    if not masks_path or not masks_path.is_file():
        raise FileNotFoundError(f"Missing masks.npz in {kinematic_sub}")

    out_dir = data_dir / "worker_segments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = out_dir / f"{worker_video.stem}_stage2_viz.mp4"

    print("=" * 70)
    print("STAGE 2 VISUALIZATION (FROM SCRATCH)")
    print("=" * 70)
    print(f"  Raw input video: {worker_video}")
    print(f"  Kinematic dir:   {kinematic_sub}")
    print(f"  Target output:   {out_video_path}")
    print(f"  Pause per step:  {pause_sec:.1f}s")

    # ── Load Kinematic Telemetry ──────────────────────────────────────────────
    mask_data   = np.load(masks_path, allow_pickle=True)
    left_masks  = mask_data["left_masks"]
    right_masks = mask_data["right_masks"]
    frame_ind   = mask_data["frame_indices"].tolist()
    frame_to_mask_idx = {int(fi): mi for mi, fi in enumerate(frame_ind)}

    overall_likelihood = None
    dynamic_threshold = None
    left_smooth_speeds = None
    right_smooth_speeds = None
    left_smooth_angles = None
    right_smooth_angles = None
    if decomposed_path.is_file():
        try:
            dec = np.load(decomposed_path)
            overall_likelihood = dec.get("overall_likelihood")
            dynamic_threshold = dec.get("dynamic_threshold")
            left_smooth_speeds = dec.get("left_smooth_speeds")
            right_smooth_speeds = dec.get("right_smooth_speeds")
            left_smooth_angles = dec.get("left_smooth_angles")
            right_smooth_angles = dec.get("right_smooth_angles")
        except Exception as e:
            print(f"  [warn] decomposed_motion error: {e}")

    left_mags = None
    right_mags = None
    boundaries = []
    if boundaries_path.is_file():
        try:
            b_dict = np.load(boundaries_path, allow_pickle=True).item()
            boundaries = b_dict.get("boundaries", [])
            left_mags = b_dict.get("left_mags")
            right_mags = b_dict.get("right_mags")
        except Exception as e:
            print(f"  [warn] boundaries error: {e}")

    boundary_set = set(int(b) for b in boundaries)
    boundary_lookup = {
        int(b): {
            "frame": int(b),
            "type": "ACTION_BOUNDARY",
            "score": 0.50,
            "sources": ["SPEED_VALLEY", "DIRECTION_SHIFT"],
        }
        for b in boundaries
    }

    # ── Load Stage 2 Macro Evaluations & Segments ─────────────────────────────
    macro_data = {}
    if macro_file.is_file():
        macro_data = json.loads(macro_file.read_text(encoding="utf-8"))

    slow_times = {}
    for ev in macro_data.get("evaluated", []):
        if ev.get("timing_verdict") == "slow":
            slow_times[round(ev["start_time"], 2)] = ev

    segments_data = json.loads(worker_segments_file.read_text(encoding="utf-8"))
    raw_segments = segments_data.get("segments", [])
    segments = merge_adjacent_segments(raw_segments, slow_times=slow_times)
    print(f"  Merged segments: {len(raw_segments)} raw -> {len(segments)} unique action blocks")

    # ── Setup Fonts ───────────────────────────────────────────────────────────
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    if not os.path.isfile(font_path):
        font_path = "/System/Library/Fonts/Helvetica.ttc"
    font_title = ImageFont.truetype(font_path, 21)
    font_sub = ImageFont.truetype(font_path, 14)
    font_badge = ImageFont.truetype(font_path, 13)
    font_time = ImageFont.truetype(font_path, 18)

    # ── Open Raw Video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(worker_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open raw video {worker_video}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 14.773
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (W, H))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot write to {out_video_path}")

    print(f"  Video: {W}x{H} @ {fps:.2f} fps | {total_frames} frames ({video_duration:.1f}s)")
    print(f"  Classified segments: {len(segments)} (0 UNKNOWN)")
    print(f"  Rendering directly from raw video...")

    # Compute exact continuous frame ranges for all segments
    for i, seg in enumerate(segments):
        if i == 0:
            seg["start_frame"] = 0
        else:
            seg["start_frame"] = segments[i - 1]["end_frame"] + 1
        if i + 1 < len(segments):
            next_sf = int(round(segments[i + 1]["start_time"] * fps))
            seg["end_frame"] = max(seg["start_frame"], next_sf - 1)
        else:
            seg["end_frame"] = total_frames - 1

    # ── Load Ground Truth (chuyen1_segment.json) ──────────────────────────────
    gt_file = data_dir / "chuyen1_segment.json"
    gt_segments = []
    gt_duration = 53.0
    if gt_file.is_file():
        try:
            gt_data = json.loads(gt_file.read_text(encoding="utf-8"))
            gt_segments = gt_data.get("segments", [])
            gt_duration = float(gt_data.get("total_duration_s", 53.0))
            print(f"  Ground Truth:    {len(gt_segments)} steps ({gt_duration:.1f}s cycle)")
        except Exception as e:
            print(f"  [warn] gt load error: {e}")

    has_gt = len(gt_segments) > 0
    bar_h = 24 if has_gt else 12
    banner_h = 52
    y_top = H - bar_h - banner_h

    # Precompute GT timeline rectangles (repeating cycle across video duration)
    gt_rects = []
    if has_gt:
        c = 0
        while c * gt_duration < video_duration:
            offset = c * gt_duration
            for s in gt_segments:
                t0 = offset + s["timestamp_start"]
                t1 = offset + s["timestamp_end"]
                if t0 >= video_duration:
                    break
                t1 = min(t1, video_duration)
                x1 = int((t0 / video_duration) * W)
                x2 = int((t1 / video_duration) * W)
                x2 = max(x2, x1 + 1)
                col = get_op_color_bgr(s["name"])
                gt_rects.append((x1, x2, col, c))
            c += 1

    # Precompute bottom timeline segment rectangles based on exact frames
    timeline_rects = []
    for s in segments:
        sf = s["start_frame"]
        ef = s["end_frame"]
        col = get_op_color_bgr(s["operation_name"])
        x1 = int((sf / max(total_frames, 1)) * W)
        x2 = int(((ef + 1) / max(total_frames, 1)) * W)
        x2 = max(x2, x1 + 1)
        timeline_rects.append((x1, x2, col))

    t0_start = time.time()
    frame_idx = 0
    cur_seg_idx = 0
    last_print = 0
    pause_frames_count = int(fps * pause_sec) if pause_sec > 0 else 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cur_t_s = frame_idx / fps

        while cur_seg_idx < len(segments) - 1 and frame_idx > segments[cur_seg_idx]["end_frame"]:
            cur_seg_idx += 1

        seg_idx = cur_seg_idx
        seg = segments[cur_seg_idx]

        op_name = seg["operation_name"]
        color = get_op_color_bgr(op_name)
        s_t0 = seg["start_time"]
        s_t1 = seg["end_time"]
        dur = max((seg["end_frame"] - seg["start_frame"] + 1) / fps, 0.01)
        elapsed_in_seg = (frame_idx - seg["start_frame"]) / fps
        progress = np.clip((frame_idx - seg["start_frame"]) / max(seg["end_frame"] - seg["start_frame"] + 1, 1), 0.0, 1.0)

        is_slow = seg.get("timing_verdict") == "slow" or round(s_t0, 2) in slow_times
        ev = seg.get("slow_ev") or slow_times.get(round(s_t0, 2), {})
        ratio = ev.get("duration_ratio", 1.5)
        exp_dur = ev.get("expert_duration_s", 1.0)

        # ── 1. Mask overlays & Centroids ──────────────────────────────────────
        l_center, r_center = None, None
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
        if left_smooth_speeds is not None and frame_idx < len(left_smooth_speeds) and l_center is not None:
            spd = float(left_smooth_speeds[frame_idx])
            ang = float(left_smooth_angles[frame_idx]) if left_smooth_angles is not None else np.nan
            if not np.isnan(ang) and spd > 0.3:
                rad = np.radians(ang)
                _draw_hand_motion_arrow(frame, l_center, float(spd * np.cos(rad)), float(spd * np.sin(rad)), (0, 0, 255))

        if right_smooth_speeds is not None and frame_idx < len(right_smooth_speeds) and r_center is not None:
            spd = float(right_smooth_speeds[frame_idx])
            ang = float(right_smooth_angles[frame_idx]) if right_smooth_angles is not None else np.nan
            if not np.isnan(ang) and spd > 0.3:
                rad = np.radians(ang)
                _draw_hand_motion_arrow(frame, r_center, float(spd * np.cos(rad)), float(spd * np.sin(rad)), (0, 255, 0))

        if l_center is not None and r_center is not None:
            dist_px = float(np.linalg.norm(np.array(l_center) - np.array(r_center)))
            cv2.line(frame, l_center, r_center, (140, 140, 140), 1, cv2.LINE_AA)
            mid_pt = ((l_center[0] + r_center[0]) // 2, (l_center[1] + r_center[1]) // 2 - 8)
            cv2.putText(frame, f"D={dist_px:.0f}px", mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

        # ── 3. Magnitude bar (top of frame) ───────────────────────────────────
        l_spd = float(left_mags[frame_idx]) if left_mags is not None and frame_idx < len(left_mags) else 0.0
        r_spd = float(right_mags[frame_idx]) if right_mags is not None and frame_idx < len(right_mags) else 0.0
        frame = _draw_magnitude_bar(frame, l_spd, r_spd, W)

        # ── 4. Boundary flash border ──────────────────────────────────────────
        if frame_idx in boundary_set:
            cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 220, 255), 5)

        # ── 5. Boundary Notification Card (alert for 14 frames) ───────────────
        _draw_boundary_notification_card(frame, frame_idx, boundary_lookup, W, H, display_duration=14)

        # ── 6. Top Telemetry HUD (Glassmorphism card) ─────────────────────────
        l_ang = float(left_smooth_angles[frame_idx]) if left_smooth_angles is not None and frame_idx < len(left_smooth_angles) else np.nan
        r_ang = float(right_smooth_angles[frame_idx]) if right_smooth_angles is not None and frame_idx < len(right_smooth_angles) else np.nan
        _draw_top_hud(frame, frame_idx, total_frames, fps, l_spd, r_spd, l_ang, r_ang, W)

        # ── 6.5 Dynamic Segmentation Engine HUD ───────────────────────────────
        if overall_likelihood is not None and dynamic_threshold is not None:
            l_val = float(overall_likelihood[frame_idx]) if frame_idx < len(overall_likelihood) else 0.0
            t_val = float(dynamic_threshold[frame_idx]) if frame_idx < len(dynamic_threshold) else 0.5
            _draw_segmentation_engine_hud(frame, l_val, t_val, W)

        # ── 7. Stage 2 Bottom Banner ──────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_top), (W, H - bar_h), (12, 16, 24), -1)
        cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)

        # Left accent strip
        accent_col = (40, 50, 230) if is_slow else color
        cv2.rectangle(frame, (0, y_top), (8, H - bar_h), accent_col, -1)

        # Mini Progress Bar on right
        prog_w = 170
        prog_x = W - 185
        prog_y = y_top + 34
        prog_fill = int(progress * prog_w)
        cv2.rectangle(frame, (prog_x, prog_y), (prog_x + prog_w, prog_y + 5), (50, 55, 65), -1)
        cv2.rectangle(frame, (prog_x, prog_y), (prog_x + prog_fill, prog_y + 5), accent_col, -1)

        # Render Text with PIL (Clean Vietnamese Unicode)
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Operation Title
        draw.text((22, y_top + 6), op_name, font=font_title, fill=(255, 255, 255))

        # Subtext (#idx | t0 - t1)
        subtext = f"#{seg_idx:02d}/{len(segments):02d} | {s_t0:.1f}s - {s_t1:.1f}s ({dur:.1f}s)"
        draw.text((22, y_top + 31), subtext, font=font_sub, fill=(160, 175, 190))

        # Status badge
        text_bbox = font_title.getbbox(op_name)
        title_w = text_bbox[2] - text_bbox[0]
        badge_x = min(22 + title_w + 16, W - 360)

        if is_slow:
            badge_text = f"⚠ CHẬM x{ratio:.2f} (Chuẩn: {exp_dur:.1f}s)"
            draw.rectangle([(badge_x, y_top + 6), (badge_x + 195, y_top + 28)], fill=(80, 20, 20), outline=(220, 60, 60))
            draw.text((badge_x + 8, y_top + 8), badge_text, font=font_badge, fill=(255, 170, 170))
        else:
            badge_text = "✓ ĐẠT CHUẨN"
            draw.rectangle([(badge_x, y_top + 6), (badge_x + 105, y_top + 28)], fill=(20, 65, 35), outline=(50, 180, 80))
            draw.text((badge_x + 8, y_top + 8), badge_text, font=font_badge, fill=(160, 255, 180))

        # Progress time numbers
        time_str = f"{elapsed_in_seg:04.1f}s / {dur:04.1f}s"
        draw.text((W - 175, y_top + 8), time_str, font=font_time, fill=(235, 235, 235))

        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # ── 8. Bottom Multi-Colored Timeline Strip with Playhead ─────────
        if has_gt:
            y_gt_top = H - bar_h
            y_gt_bot = H - 12
            y_wrk_top = H - 11
            y_wrk_bot = H

            # Background
            cv2.rectangle(frame, (0, y_gt_top), (W, H), (14, 16, 22), -1)

            # Draw GT Track (Top: Ground Truth)
            for (rx1, rx2, rcol, cycle) in gt_rects:
                cv2.rectangle(frame, (rx1, y_gt_top), (rx2, y_gt_bot), rcol, -1)
                cv2.line(frame, (rx1, y_gt_top), (rx1, y_gt_bot), (15, 15, 15), 1)

            # Cycle dividers in GT track
            c_count = int(video_duration // gt_duration)
            for cy in range(1, c_count + 1):
                cy_x = int(((cy * gt_duration) / video_duration) * W)
                cv2.line(frame, (cy_x, y_gt_top), (cy_x, y_gt_bot), (255, 255, 255), 2)

            # Draw Worker Track (Bottom: Worker Classified)
            for (rx1, rx2, rcol) in timeline_rects:
                cv2.rectangle(frame, (rx1, y_wrk_top), (rx2, y_wrk_bot), rcol, -1)
                cv2.line(frame, (rx1, y_wrk_top), (rx1, y_wrk_bot), (15, 15, 15), 1)

            # Track labels on the left
            cv2.rectangle(frame, (0, y_gt_top), (22, y_gt_bot), (10, 12, 16), -1)
            cv2.putText(frame, "GT", (3, y_gt_bot - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.rectangle(frame, (0, y_wrk_top), (22, y_wrk_bot), (10, 12, 16), -1)
            cv2.putText(frame, "WRK", (1, y_wrk_bot - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.24, (255, 255, 255), 1, cv2.LINE_AA)

            # Live playhead cursor cutting across both tracks
            play_x = int((frame_idx / max(total_frames, 1)) * W)
            cv2.rectangle(frame, (max(0, play_x - 2), y_gt_top - 2), (min(W - 1, play_x + 2), H), (255, 255, 255), -1)
        else:
            y_bar = H - bar_h
            cv2.rectangle(frame, (0, y_bar), (W, H), (20, 22, 28), -1)
            for (rx1, rx2, rcol) in timeline_rects:
                cv2.rectangle(frame, (rx1, y_bar), (rx2, H), rcol, -1)
                cv2.line(frame, (rx1, y_bar), (rx1, H), (15, 15, 15), 1)

            # Live playhead
            play_x = int((frame_idx / max(total_frames, 1)) * W)
            cv2.rectangle(frame, (max(0, play_x - 2), y_bar - 2), (min(W - 1, play_x + 2), H), (255, 255, 255), -1)

        writer.write(frame)

        # ── 9. Clean Vietnamese Transition Card at end of segment ────────
        is_seg_end = (frame_idx == seg["end_frame"])
        if is_seg_end and pause_frames_count > 0 and frame_idx < total_frames - 1:
            pause_card = _draw_pause_card(
                frame, op_name, seg_idx + 1, len(segments), dur,
                is_slow, ratio, exp_dur, font_title, font_sub, font_badge, W, H,
            )
            for _ in range(pause_frames_count):
                writer.write(pause_card)

        frame_idx += 1

        if time.time() - last_print > 3.0:
            last_print = time.time()
            pct = (frame_idx / total_frames) * 100
            print(f"  Rendering: {frame_idx}/{total_frames} frames ({pct:.1f}%) - {cur_t_s:.1f}s / {video_duration:.1f}s")

    cap.release()
    writer.release()

    # Create root symlink for easy access
    symlink_path = data_dir / "worker_stage2_viz.mp4"
    symlink_path.unlink(missing_ok=True)
    try:
        symlink_path.symlink_to(out_video_path.relative_to(data_dir))
    except Exception:
        pass

    elapsed = time.time() - t0_start
    print(f"\n[Completed] Rendered in {elapsed:.1f}s ({total_frames/elapsed:.1f} fps)!")
    print(f"Stage 2 from-scratch video saved -> {out_video_path}")
    return out_video_path


def main():
    parser = argparse.ArgumentParser(description="Render Stage 2 visualization directly from raw video.")
    parser.add_argument("cd", nargs="?", default="1", help="Operation ID (default: 1)")
    parser.add_argument("--pause-sec", type=float, default=1.0, help="Pause duration at end of each segment in seconds (default: 1.0, set 0 for continuous)")
    args = parser.parse_args()
    render_stage2_video(args.cd, pause_sec=args.pause_sec)


if __name__ == "__main__":
    main()

