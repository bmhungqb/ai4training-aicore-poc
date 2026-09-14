"""Motion visualization utilities: Motion History Image (MHI), optical flow overlay,
and motion-enhanced composite frames for VLM temporal understanding.
"""
from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path


def compute_mhi(frames: list[np.ndarray], tau: int = 15) -> np.ndarray:
    """Compute Motion History Image from a sequence of frames.

    Args:
        frames: List of BGR frames (numpy arrays).
        tau: Temporal decay constant — how many frames motion history persists.
             Higher = longer trails. Default 15 works well for ~25fps sewing video.

    Returns:
        BGR image where:
        - Bright/hot colors = recent motion
        - Dark/cold colors = old motion or static
        - Black = no motion detected
    """
    if len(frames) < 2:
        return _empty_mhi(frames[0] if frames else np.zeros((480, 640, 3), np.uint8))

    # Normalize all frames to the same size (OpenCV 5.0 requires exact match for absdiff)
    h, w = frames[0].shape[:2]
    c = frames[0].shape[2] if len(frames[0].shape) > 2 else 1
    resized = []
    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        resized.append(frame)

    mhi = np.zeros((h, w), np.float32)
    prev_gray = cv2.cvtColor(resized[0], cv2.COLOR_BGR2GRAY)

    for i, frame in enumerate(resized[1:], start=1):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        # Binary motion mask — pixels that changed more than threshold
        motion_mask = (diff > 25).astype(np.float32)
        # Update MHI: motion pixels get tau - (i % tau), others decay
        mhi = np.where(motion_mask > 0,
                       float(tau - (i % tau)),
                       np.maximum(mhi - 1.0, 0.0))
        prev_gray = gray

    # Normalize to 0-255
    mhi_norm = cv2.normalize(mhi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply colormap: JET makes recent motion yellow/white, old motion blue/purple
    mhi_color = cv2.applyColorMap(mhi_norm, cv2.COLORMAP_JET)
    return mhi_color


def compute_optical_flow_viz(frames: list[np.ndarray], step: int = 10) -> np.ndarray:
    """Compute dense optical flow and render as HSV color-wheel overlay.

    Args:
        frames: List of BGR frames (at least 2).
        step: Grid step for sparse flow visualization. Smaller = denser but slower.

    Returns:
        BGR image with HSV flow overlay on the last frame.
        Hue = flow direction (0-180 in OpenCV HSV), Saturation = 255, Value = flow magnitude.
    """
    if len(frames) < 2:
        return frames[0].copy() if frames else np.zeros((480, 640, 3), np.uint8)

    # Normalize to same size (OpenCV 5.0 requires exact match)
    h, w = frames[0].shape[:2]
    resized = []
    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        resized.append(frame)

    prev_gray = cv2.cvtColor(resized[0], cv2.COLOR_BGR2GRAY)
    base_frame = resized[-1].copy()

    # Compute Farneback optical flow (fast, good enough for motion viz)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        cv2.cvtColor(resized[-1], cv2.COLOR_BGR2GRAY),
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    # Convert flow to HSV
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = (angle * 180 / np.pi / 2).astype(np.uint8)  # Hue = direction
    hsv[..., 1] = 255  # Full saturation
    hsv[..., 2] = np.clip(magnitude * 4, 0, 255).astype(np.uint8)  # Value = speed

    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    overlay = cv2.addWeighted(base_frame, 0.6, flow_bgr, 0.4, 0)
    return overlay


def _empty_mhi(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return np.zeros((h, w, 3), np.uint8)


def create_motion_composite(
    key_frame: np.ndarray,
    mhi: np.ndarray,
    flow_viz: np.ndarray | None = None,
    motion_stats: dict | None = None,
    info_height: int = 80,
) -> np.ndarray:
    """Create a motion-enhanced composite frame: key frame + MHI + optional flow + info bar.

    This composite packs temporal motion information into a single image for VLM consumption,
    solving the temporal blindness problem.

    Args:
        key_frame: The representative key frame (BGR).
        mhi: Motion History Image from compute_mhi().
        flow_viz: Optional optical flow visualization.
        motion_stats: Dict with keys: speed_median, speed_rms, turbulence, direction_median.
        info_height: Height of the info bar at the bottom (pixels).

    Returns:
        Composite BGR image ready for base64 encoding.
    """
    # Resize MHI and flow to match key frame dimensions
    h, w = key_frame.shape[:2]
    mhi_resized = cv2.resize(mhi, (w, h))

    if flow_viz is not None:
        flow_resized = cv2.resize(flow_viz, (w, h))
        # Stack: key frame | MHI | flow
        if w * 3 < 1920:  # Only stack 3 if not too wide
            top_row = np.hstack([key_frame, mhi_resized, flow_resized])
        else:
            top_row = np.hstack([key_frame, mhi_resized])
    else:
        # Stack: key frame | MHI
        top_row = np.hstack([key_frame, mhi_resized])

    # Create info bar
    info_bar = np.zeros((info_height, top_row.shape[1], 3), dtype=np.uint8)

    if motion_stats:
        y_pos = 25
        col = (0, 255, 0)  # Green text

        speed = motion_stats.get("speed_median", 0)
        cv2.putText(info_bar, f"Speed: {speed:.3f}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        # Speed bar
        bar_w = int(min(speed * 300, top_row.shape[1] - 200))
        cv2.rectangle(info_bar, (130, y_pos - 18), (130 + bar_w, y_pos - 2), (0, 255, 0), -1)

        turbulence = motion_stats.get("turbulence", 0)
        cv2.putText(info_bar, f"Turbu: {turbulence:.3f}", (10, y_pos + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        bar_w = int(min(turbulence * 300, top_row.shape[1] - 200))
        cv2.rectangle(info_bar, (130, y_pos + 12), (130 + bar_w, y_pos + 28), (0, 165, 255), -1)

        direction = motion_stats.get("direction_median", 0)
        direction_deg = np.degrees(direction)
        cv2.putText(info_bar, f"Dir: {direction_deg:.0f}deg", (top_row.shape[1] - 150, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return np.vstack([top_row, info_bar])


def load_frames_from_paths(frame_paths: list[str | Path]) -> list[np.ndarray]:
    """Load frames from a list of file paths. Returns empty list on failure."""
    frames = []
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is not None:
            frames.append(img)
    return frames


def encode_motion_composite(
    frame_paths: list[str | Path],
    motion_stats: dict | None = None,
    target_width: int = 1280,
) -> str:
    """Create and encode a motion-enhanced composite from frame paths.

    Args:
        frame_paths: List of frame file paths in time order.
        motion_stats: Optional kinematic stats (speed_median, speed_rms, turbulence, direction_median).
        target_width: Resize output to this width.

    Returns:
        Base64-encoded JPEG string ready for VLM input.
    """
    import base64

    frames = load_frames_from_paths(frame_paths)
    if not frames:
        raise ValueError("No valid frames provided for motion composite")

    # Use middle frame as key frame
    key_idx = len(frames) // 2
    key_frame = frames[key_idx]

    # Compute MHI from all frames
    mhi = compute_mhi(frames)

    # Optionally compute flow viz (slower but more informative)
    flow_viz = compute_optical_flow_viz(frames) if len(frames) >= 2 else None

    # Create composite
    composite = create_motion_composite(key_frame, mhi, flow_viz, motion_stats)

    # Resize to target width
    h, w = composite.shape[:2]
    scale = target_width / w
    composite = cv2.resize(composite, (target_width, int(h * scale)),
                           interpolation=cv2.INTER_CUBIC)

    # Encode as JPEG
    _, buf = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()
