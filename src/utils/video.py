"""Video I/O primitives shared across phases: cutting clips with ffmpeg and
sampling frames with OpenCV."""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2


def cut_clip(video_path: str | Path, start: float, end: float, out_path: str | Path) -> None:
    """Cut [start, end) out of `video_path` into `out_path`.

    Re-encodes (does not stream-copy) so the cut points land exactly on
    start/end instead of snapping to the nearest keyframe.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "libx264", "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def extract_frames(video_path: str | Path, out_dir: str | Path,
                    sample_fps: float) -> list[tuple[float, Path]]:
    """Sample `video_path` uniformly at `sample_fps`, writing JPEGs to `out_dir`.

    Returns [(timestamp_s, frame_path), ...] in order.
    """
    return extract_frames_in_range(video_path, 0.0, None, out_dir, sample_fps)


def extract_frames_in_range(video_path: str | Path, start: float, end: float | None,
                            out_dir: str | Path, sample_fps: float) -> list[tuple[float, Path]]:
    """Sample `video_path` uniformly at `sample_fps`, restricted to [start, end)
    seconds (`end=None` means to the end of the video), writing JPEGs (named by
    absolute frame index in the whole video) to `out_dir`.

    Same use as extract_frames(), but scoped to one scene's time range instead
    of the whole video — for extracting per-scene curation candidates without
    cutting the scene into its own clip first.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps / sample_fps))

    start_idx = max(0, round(start * fps))
    end_idx = n_frames if end is None else min(n_frames, round(end * fps))

    frames: list[tuple[float, Path]] = []
    for frame_idx in range(start_idx, end_idx, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        ts = frame_idx / fps
        out_path = out_dir / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(out_path), frame_bgr)
        frames.append((ts, out_path))
    cap.release()
    return frames


def worker_frame_ts(frame_path: str | Path) -> float:
    """Timestamp encoded in a cached worker-frame filename
    (frame_<t>s.jpg, as written by sample_window_frames_cached())."""
    return float(Path(frame_path).stem.removeprefix("frame_").removesuffix("s"))


def sample_window_frames_cached(video_path: str | Path, t0: float, t1: float, fps: float,
                                out_dir: str | Path) -> list[tuple[float, Path]]:
    """Sample [t0, t1) at `fps`, writing JPEGs named by timestamp
    (frame_<t>s.jpg) to `out_dir` — reusing a file already on disk from an
    earlier, overlapping call instead of re-extracting it.

    Used by SegmentClassifier (Phase 2 / classify) to sample each action
    segment's frame window.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = 1.0 / fps

    frames: list[tuple[float, Path]] = []
    t = t0
    while t < t1:
        out_path = out_dir / f"frame_{t:07.2f}s.jpg"
        if not out_path.exists():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * src_fps)))
            ok, frame_bgr = cap.read()
            if ok:
                cv2.imwrite(str(out_path), frame_bgr)
            else:
                t += step
                continue
        frames.append((round(t, 2), out_path))
        t += step
    cap.release()
    return frames


def extract_frames_by_index(video_path: str | Path, frame_indices: list[int],
                            out_dir: str | Path) -> list[Path]:
    """Extract specific absolute frame indices (frame numbers in `video_path`,
    not scene-relative) as JPEGs into `out_dir`. Skips indices already written.

    Returns one Path per index in `frame_indices`, in the same order (a frame
    that fails to decode is skipped, so the result may be shorter).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))

    paths: list[Path] = []
    for idx in frame_indices:
        out_path = out_dir / f"frame_{idx:05d}.jpg"
        if not out_path.exists():
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            cv2.imwrite(str(out_path), frame_bgr)
        paths.append(out_path)
    cap.release()
    return paths
