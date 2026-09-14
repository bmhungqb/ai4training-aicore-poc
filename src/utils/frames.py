"""Frame selection and base64 encoding shared across phases."""
from __future__ import annotations

import base64
from pathlib import Path

import cv2

from src.config.common import (
    EXPERT_CROP_BOX, EXPERT_FRAME_WIDTH, WORKER_CROP_BOX, WORKER_FRAME_WIDTH)


def pick_evenly_spread(items: list, max_n: int) -> list:
    """Pick up to `max_n` items evenly spread across `items` (always keeps the
    first and last). Returns `items` unchanged if it already has <= max_n."""
    if len(items) <= max_n:
        return list(items)
    if max_n == 1:
        return [items[0]]
    step = (len(items) - 1) / (max_n - 1)
    idxs = sorted({round(i * step) for i in range(max_n)})
    return [items[i] for i in idxs]


def sharpness_score(frame_path: str | Path) -> float:
    """Blur/sharpness score for one frame: variance of the Laplacian on the
    grayscale image (a standard, model-free blur metric — sharp edges produce
    a high-variance second derivative, blurred/out-of-focus frames produce a
    low one). Higher = sharper. Returns 0.0 if the frame can't be read."""
    img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def pick_sharpest_spread(items: list, max_n: int, sharpness_pool_factor: int = 3) -> list:
    """Pick up to `max_n` items spread across `items` in time order, but for
    each of the `max_n` time slots, keep the SHARPEST frame within a small
    neighborhood around that slot instead of the frame exactly at that slot.

    `items` must be frame paths (str/Path) in time order. `sharpness_pool_factor`
    controls how wide each neighborhood is: pool size = `sharpness_pool_factor`
    consecutive frames worth of candidates per kept slot (higher = more chance
    to dodge a blurry frame, but drifts further from perfectly even spacing).

    Falls back to pick_evenly_spread() unchanged if `items` already has <= max_n.
    """
    if len(items) <= max_n:
        return list(items)
    if max_n == 1:
        return [max(items, key=sharpness_score)]

    n = len(items)
    step = (n - 1) / (max_n - 1)
    half_pool = max(1, round(step * sharpness_pool_factor / 2))
    chosen = []
    seen_idx = set()
    for i in range(max_n):
        center = round(i * step)
        lo, hi = max(0, center - half_pool), min(n - 1, center + half_pool)
        candidates = [j for j in range(lo, hi + 1) if j not in seen_idx]
        if not candidates:
            continue
        best_j = max(candidates, key=lambda j: sharpness_score(items[j]))
        seen_idx.add(best_j)
        chosen.append(best_j)
    return [items[j] for j in sorted(chosen)]


def sample_sharp_points_in_window(
    video_path: Path,
    t0: float,
    t1: float,
    points: list[float],
    scene_dir: Path,
    sharpness_pool_factor: int = 3,
) -> list[tuple[float, Path]]:
    """For each frac in `points` (each 0.0-1.0 as fraction of [t0, t1]),
    sample a neighborhood of frames around that point, pick the sharpest one,
    and write it to disk.

    Returns list of (timestamp_s, frame_path) sorted by timestamp.
    Caches to disk: reuses existing frame files if present.
    """
    import cv2

    out_dir = Path(scene_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = t1 - t0

    results: list[tuple[float, Path]] = []
    for frac in points:
        t_target = t0 + frac * duration
        half_frac = 1.0 / sharpness_pool_factor
        t_lo = max(t0, t_target - half_frac * duration)
        t_hi = min(t1, t_target + half_frac * duration)
        cand_step = 1.0 / (5.0 * src_fps)

        t_cands = []
        t = t_lo
        while t <= t_hi:
            t_cands.append(t)
            t += cand_step
        if t_cands and t_cands[-1] < t_hi:
            t_cands.append(t_hi)

        best_path, best_score = None, -1.0
        for tc in t_cands:
            out_path = out_dir / f"frame_{tc:07.2f}s.jpg"
            if not out_path.exists():
                frame_idx = int(round(tc * src_fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame_bgr = cap.read()
                if not ok:
                    continue
                cv2.imwrite(str(out_path), frame_bgr)
            score = sharpness_score(out_path)
            if score > best_score:
                best_score = score
                best_path = out_path
        if best_path is not None:
            ts = round(float(best_path.stem.removeprefix("frame_").removesuffix("s")), 2)
            results.append((ts, best_path))

    cap.release()
    return sorted(results, key=lambda x: x[0])


import numpy as np

_MASK_CACHE: dict[str, np.ndarray | None] = {}


def load_mask_cached(mask_path: str | Path | None) -> np.ndarray | None:
    """Load mask image as grayscale array and cache it."""
    if mask_path is None:
        return None
    key = str(Path(mask_path).resolve())
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]
    p = Path(mask_path)
    if not p.exists():
        _MASK_CACHE[key] = None
        return None
    mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    _MASK_CACHE[key] = mask
    return mask


def find_mask_for_video(video_path: str | Path | None = None,
                        action_segments_path: str | Path | None = None,
                        explicit_mask: str | Path | None = None) -> Path | None:
    """Auto-detect mask image (<video_stem>.mask.png) for a video.
    Checks:
    1. explicit_mask if provided and exists.
    2. Next to video_path or next to its resolved symlink target.
    3. Next to video in action_segments_path's directory structure (e.g. data/{cd}/{stem}.mask.png).
    """
    if explicit_mask is not None:
        p = Path(explicit_mask)
        if p.exists():
            return p

    if video_path is not None:
        vp = Path(video_path)
        candidates = [
            vp.with_name(f"{vp.stem}.mask.png"),
            vp.resolve().with_name(f"{vp.resolve().stem}.mask.png"),
        ]
        for c in candidates:
            if c.exists():
                return c

    if action_segments_path is not None:
        asp = Path(action_segments_path)
        if len(asp.parents) >= 3:
            cand = asp.parents[2] / f"{asp.parent.name}.mask.png"
            if cand.exists():
                return cand
        cand2 = asp.parent / f"{asp.parent.name}.mask.png"
        if cand2.exists():
            return cand2

    return None


def _encode_jpeg_b64(img) -> str:
    return base64.b64encode(cv2.imencode(".jpg", img)[1].tobytes()).decode()


def _encode_frame(frame_path: str, crop_box: tuple[int, int, int, int] | None,
                  target_width: int, save_dir: str | Path | None = None,
                  mask_path: str | Path | None = None) -> str:
    """Encode a frame for VLM inspection:
    1. If `mask_path` exists, zero out pixels outside the mask (>127) and crop
       to the bounding box of the kept area.
    2. Else if `crop_box` is given, crop to that (x1, y1, x2, y2).
    3. Otherwise, use the full frame uncropped.
    4. Resize to `target_width` maintaining aspect ratio.
    """
    img = cv2.imread(str(frame_path))
    if img is None:
        raise ValueError(f"Could not read frame from {frame_path}")

    mask = load_mask_cached(mask_path)
    if mask is not None:
        h, w = img.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        # Zero out pixels outside the mask
        img = cv2.bitwise_and(img, img, mask=(mask > 127).astype(np.uint8))
        # Crop tightly to the bounding box of the mask
        ys, xs = np.where(mask > 127)
        if len(xs) > 0:
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            img = img[y1:y2 + 1, x1:x2 + 1]
    elif crop_box is not None:
        x1, y1, x2, y2 = crop_box
        img = img[y1:y2, x1:x2]

    h, w = img.shape[:2]
    scale = target_width / w
    img_resized = cv2.resize(img, (target_width, int(round(h * scale))),
                             interpolation=cv2.INTER_CUBIC)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / Path(frame_path).name), img_resized)
    return _encode_jpeg_b64(img_resized)


def encode_expert_frame(frame_path: str, crop_box: tuple[int, int, int, int] | None = EXPERT_CROP_BOX,
                        target_width: int = EXPERT_FRAME_WIDTH,
                        save_dir: str | Path | None = None,
                        mask_path: str | Path | None = None) -> str:
    """Encode an expert frame with optional mask or crop box."""
    return _encode_frame(frame_path, crop_box, target_width, save_dir, mask_path=mask_path)


def encode_worker_frame(frame_path: str, crop_box: tuple[int, int, int, int] | None = WORKER_CROP_BOX,
                        target_width: int = WORKER_FRAME_WIDTH,
                        save_dir: str | Path | None = None,
                        mask_path: str | Path | None = None) -> str:
    """Encode a worker frame with optional mask or crop box."""
    return _encode_frame(frame_path, crop_box, target_width, save_dir, mask_path=mask_path)

