#!/usr/bin/env python3
"""
SEA-RAFT Optical Flow with Hand Masks

Input:
    - Video
    - Pre-computed SAM3 hand masks (.npz)

Output:
    - Optical flow .npz
    - Individual flow .npy files
    - Optional visualization video

Important fixes:
    1. Visualization writer uses ACTUAL resized frame dimensions.
    2. Robust handling of None/object masks.
    3. Robust flow_to_image() handling.
    4. Visualization frame dimensions are guaranteed to match VideoWriter.
    5. Avoids unnecessary CUDA tensor retention.
    6. Safer frame/mask indexing.
    7. Handles empty masks.
    8. Handles grayscale / malformed masks.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


# ============================================================
# SEA-RAFT: vendored inference code under src/kinematic_pipeline/searaft_core/
# (from github.com/princeton-vl/SEA-RAFT, BSD-3-Clause) instead of a separate
# SEA-RAFT checkout on the filesystem.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent / "searaft_core"

from searaft_core.raft import RAFT
from searaft_core.utils.flow_viz import flow_to_image


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="SEA-RAFT optical flow with SAM3 hand masks"
    )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    parser.add_argument(
        "--video",
        required=True,
        help="Input video path",
    )

    parser.add_argument(
        "--masks",
        required=True,
        help="SAM3 masks .npz file",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory",
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    parser.add_argument(
        "--model",
        default="MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
        help="SEA-RAFT pretrained model",
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=12,
        help="SEA-RAFT iterations",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device",
    )

    # --------------------------------------------------------
    # Processing
    # --------------------------------------------------------

    parser.add_argument(
        "--video-scale",
        type=float,
        default=0.5,
        help="Resize video before optical flow. 1.0 = original size.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Maximum number of frames to process. -1 = all.",
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization video",
    )

    parser.add_argument(
        "--vector-step",
        type=int,
        default=16,
        help="Spacing between flow vectors",
    )

    parser.add_argument(
        "--vector-scale",
        type=float,
        default=4.0,
        help="Visual scale for flow arrows",
    )

    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.30,
        help="Mask overlay alpha",
    )

    return parser.parse_args()


# ============================================================
# Load masks
# ============================================================

def load_masks(mask_path):

    print("\n" + "=" * 70)
    print("Loading SAM3 masks")
    print("=" * 70)

    data = np.load(mask_path, allow_pickle=True)

    required_keys = [
        "left_masks",
        "right_masks",
        "frame_indices",
        "width",
        "height",
        "fps",
    ]

    for key in required_keys:
        if key not in data:
            raise KeyError(
                f"Missing '{key}' in mask file: {mask_path}"
            )

    left_masks = data["left_masks"]
    right_masks = data["right_masks"]
    frame_indices = data["frame_indices"]

    width = int(data["width"])
    height = int(data["height"])
    fps = float(data["fps"])

    print(f"Mask frames : {len(left_masks)}")
    print(f"Resolution  : {width}x{height}")
    print(f"FPS         : {fps}")

    return (
        left_masks,
        right_masks,
        frame_indices,
        width,
        height,
        fps,
    )


# ============================================================
# Frame utilities
# ============================================================

def resize_frame(frame, scale):

    if frame is None:
        return None

    if scale == 1.0:
        return frame

    if scale <= 0:
        raise ValueError(
            f"video-scale must be > 0, got {scale}"
        )

    h, w = frame.shape[:2]

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    return cv2.resize(
        frame,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )


def frame_to_tensor(frame, device):

    if frame is None:
        raise ValueError("frame_to_tensor received None")

    if frame.ndim != 3:
        raise ValueError(
            f"Expected BGR frame [H,W,3], got shape={frame.shape}"
        )

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    tensor = torch.from_numpy(
        np.ascontiguousarray(rgb)
    )

    tensor = tensor.float()

    tensor = tensor.permute(
        2,
        0,
        1,
    )

    tensor = tensor.unsqueeze(0)

    return tensor.to(
        device,
        non_blocking=True,
    )


# ============================================================
# Mask utilities
# ============================================================

def safe_mask_array(mask):

    """
    Convert arbitrary NPZ mask object into a valid 2D numpy array.

    Handles:
        None
        numpy arrays
        object arrays
        empty arrays
        boolean arrays
        uint8 arrays
    """

    if mask is None:
        return None

    try:

        # Object scalar containing None
        if isinstance(mask, np.ndarray):

            if mask.ndim == 0:

                try:
                    value = mask.item()

                    if value is None:
                        return None

                    mask = value

                except Exception:
                    return None

        if mask is None:
            return None

        mask = np.asarray(mask)

        if mask.size == 0:
            return None

        # Remove singleton dimensions
        mask = np.squeeze(mask)

        if mask.ndim != 2:
            return None

        if mask.dtype != np.bool_:
            mask = mask > 0

        return np.ascontiguousarray(mask)

    except Exception as exc:

        print(
            f"Warning: invalid mask ignored: {exc}"
        )

        return None


def resize_mask(mask, width, height):

    mask = safe_mask_array(mask)

    if mask is None:
        return None

    if mask.shape == (height, width):
        return mask

    resized = cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    return resized.astype(bool)


def combine_masks(
    left_mask,
    right_mask,
    height,
    width,
):

    """
    Return:

        0 = background
        1 = left hand
        2 = right hand
    """

    combined = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    left_mask = resize_mask(
        left_mask,
        width,
        height,
    )

    right_mask = resize_mask(
        right_mask,
        width,
        height,
    )

    if left_mask is not None:
        combined[left_mask] = 1

    if right_mask is not None:
        combined[right_mask] = 2

    return combined


def apply_mask_to_flow(flow, mask):

    if flow is None:
        return None

    if mask is None:
        return flow

    h, w = flow.shape[:2]

    mask = resize_mask(
        mask,
        w,
        h,
    )

    if mask is None:
        return np.zeros_like(flow)

    mask_3d = mask[..., None].astype(
        flow.dtype
    )

    return flow * mask_3d


# ============================================================
# SEA-RAFT
# ============================================================

def load_config(cfg_path):

    cfg_path = Path(cfg_path)

    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config not found: {cfg_path}"
        )

    print(
        f"Reading config: {cfg_path}"
    )

    with open(
        cfg_path,
        "r",
    ) as f:

        config = json.load(f)

    return config


class SeaRaftModel:

    def __init__(
        self,
        model_name,
        device,
        iters,
    ):

        if device == "cuda" and not torch.cuda.is_available():

            print(
                "WARNING: CUDA requested but unavailable. "
                "Falling back to CPU."
            )

            device = "cpu"

        self.device = torch.device(device)

        self.iters = iters

        self.config = load_config(
            "config/eval/spring-M.json"
        )

        self.model = self._load_model(
            model_name,
            self.config,
        )

    def _load_model(
        self,
        model_name,
        config,
    ):

        print("\n" + "=" * 70)
        print("Loading SEA-RAFT")
        print("=" * 70)

        print(f"Model : {model_name}")
        print(f"Device: {self.device}")
        print(f"Iters : {self.iters}")

        from types import SimpleNamespace

        model_args = SimpleNamespace(
            **config
        )

        model = RAFT.from_pretrained(
            model_name,
            args=model_args,
        )

        model = model.to(
            self.device
        )

        model.eval()

        print("Model loaded successfully")

        if self.device.type == "cuda":

            print(
                "GPU   :",
                torch.cuda.get_device_name(0),
            )

        return model

    @torch.no_grad()
    def compute_flow(
        self,
        frame1,
        frame2,
    ):

        if frame1 is None:
            raise ValueError(
                "frame1 is None"
            )

        if frame2 is None:
            raise ValueError(
                "frame2 is None"
            )

        if frame1.shape != frame2.shape:

            raise ValueError(
                f"Frame size mismatch: "
                f"{frame1.shape} vs {frame2.shape}"
            )

        img1 = frame_to_tensor(
            frame1,
            self.device,
        )

        img2 = frame_to_tensor(
            frame2,
            self.device,
        )

        output = self.model(
            img1,
            img2,
            iters=self.iters,
            test_mode=True,
        )

        # ----------------------------------------------------
        # Extract flow
        # ----------------------------------------------------

        if isinstance(output, dict):

            if "flow" not in output:
                raise KeyError(
                    "SEA-RAFT output dict does not contain 'flow'. "
                    f"Keys: {list(output.keys())}"
                )

            flow_output = output["flow"]

            if isinstance(
                flow_output,
                (list, tuple),
            ):

                flow = flow_output[-1]

            else:

                flow = flow_output

        elif isinstance(
            output,
            (tuple, list),
        ):

            flow = output[-1]

        else:

            flow = output

        if flow is None:
            raise RuntimeError(
                "SEA-RAFT returned None flow"
            )

        # ----------------------------------------------------
        # Tensor -> numpy
        # ----------------------------------------------------

        if not torch.is_tensor(flow):

            raise TypeError(
                f"Unexpected flow type: "
                f"{type(flow)}"
            )

        if flow.ndim != 4:

            raise ValueError(
                f"Unexpected flow shape: "
                f"{flow.shape}"
            )

        flow_np = (
            flow[0]
            .permute(1, 2, 0)
            .contiguous()
            .float()
            .cpu()
            .numpy()
        )

        return flow_np.astype(
            np.float32,
            copy=False,
        )

    @torch.no_grad()
    def compute_masked_flow(
        self,
        frame1,
        frame2,
        mask,
    ):

        flow = self.compute_flow(
            frame1,
            frame2,
        )

        if mask is not None:

            flow = apply_mask_to_flow(
                flow,
                mask,
            )

        return flow


# ============================================================
# Visualization
# ============================================================

def safe_flow_visualization(flow):

    """
    Convert optical flow to BGR visualization.

    If SEA-RAFT's flow_to_image fails for any reason,
    fall back to a local HSV implementation.
    """

    if flow is None:
        raise ValueError(
            "Cannot visualize None flow"
        )

    flow = np.asarray(
        flow,
        dtype=np.float32,
    )

    if flow.ndim != 3 or flow.shape[2] != 2:

        raise ValueError(
            f"Expected flow [H,W,2], "
            f"got {flow.shape}"
        )

    # Remove NaN / Inf
    flow = np.nan_to_num(
        flow,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    try:

        result = flow_to_image(
            flow,
            convert_to_bgr=True,
        )

        result = np.asarray(
            result,
            dtype=np.uint8,
        )

        if result.ndim == 3:

            return result

    except Exception as exc:

        print(
            f"Warning: flow_to_image failed: {exc}"
        )

    # --------------------------------------------------------
    # Fallback HSV visualization
    # --------------------------------------------------------

    u = flow[..., 0]
    v = flow[..., 1]

    magnitude, angle = cv2.cartToPolar(
        u,
        v,
        angleInDegrees=True,
    )

    hsv = np.zeros(
        (flow.shape[0], flow.shape[1], 3),
        dtype=np.uint8,
    )

    hsv[..., 0] = (
        angle / 2
    ).astype(np.uint8)

    # Normalize magnitude
    if magnitude.max() > 0:

        mag_norm = cv2.normalize(
            magnitude,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        )

    else:

        mag_norm = np.zeros_like(
            magnitude
        )

    hsv[..., 1] = 255
    hsv[..., 2] = mag_norm.astype(
        np.uint8
    )

    return cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR,
    )


def draw_flow_vectors(
    frame,
    flow,
    step=16,
    vector_scale=4.0,
):

    if frame is None:
        raise ValueError(
            "draw_flow_vectors received None frame"
        )

    if flow is None:
        return frame.copy()

    result = frame.copy()

    h, w = flow.shape[:2]

    # Make sure flow and frame have the same size.
    if result.shape[:2] != (h, w):

        result = cv2.resize(
            result,
            (w, h),
            interpolation=cv2.INTER_AREA,
        )

    step = max(
        1,
        int(step),
    )

    vector_scale = float(
        vector_scale
    )

    for y in range(
        step // 2,
        h,
        step,
    ):

        for x in range(
            step // 2,
            w,
            step,
        ):

            u = float(
                flow[y, x, 0]
            )

            v = float(
                flow[y, x, 1]
            )

            if not np.isfinite(u) or not np.isfinite(v):
                continue

            magnitude = np.sqrt(
                u * u + v * v
            )

            if magnitude < 0.5:
                continue

            x2 = int(
                round(
                    x + u * vector_scale
                )
            )

            y2 = int(
                round(
                    y + v * vector_scale
                )
            )

            x2 = int(
                np.clip(
                    x2,
                    0,
                    w - 1,
                )
            )

            y2 = int(
                np.clip(
                    y2,
                    0,
                    h - 1,
                )
            )

            cv2.arrowedLine(
                result,
                (x, y),
                (x2, y2),
                (0, 255, 0),
                1,
                cv2.LINE_AA,
                tipLength=0.25,
            )

    return result


def draw_mask_overlay(
    frame,
    mask,
    alpha=0.30,
):

    result = frame.copy()

    if mask is None:
        return result

    h, w = frame.shape[:2]

    mask = resize_mask(
        mask,
        w,
        h,
    )

    if mask is None:
        return result

    # --------------------------------------------------------
    # BGR
    #
    # left  = red   [0,0,255]
    # right = green [0,255,0]
    # --------------------------------------------------------

    overlay = np.zeros_like(
        frame
    )

    overlay[mask == 1] = (
        0,
        0,
        255,
    )

    overlay[mask == 2] = (
        0,
        255,
        0,
    )

    alpha = float(
        np.clip(
            alpha,
            0.0,
            1.0,
        )
    )

    return cv2.addWeighted(
        frame,
        1.0 - alpha,
        overlay,
        alpha,
        0,
    )


def put_label(
    image,
    text,
    x=10,
    y=30,
):

    cv2.rectangle(
        image,
        (
            x - 5,
            y - 22,
        ),
        (
            x + len(text) * 11,
            y + 5,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return image


def visualize_flow_and_mask(
    frame,
    flow,
    mask,
    vector_step=16,
    vector_scale=4.0,
    mask_alpha=0.30,
):

    if frame is None:
        raise ValueError(
            "Visualization received None frame"
        )

    if flow is None:
        raise ValueError(
            "Visualization received None flow"
        )

    # --------------------------------------------------------
    # Normalize frame size to flow size
    # --------------------------------------------------------

    flow_h, flow_w = flow.shape[:2]

    if frame.shape[:2] != (
        flow_h,
        flow_w,
    ):

        frame = cv2.resize(
            frame,
            (flow_w, flow_h),
            interpolation=cv2.INTER_AREA,
        )

    # --------------------------------------------------------
    # 1. Mask overlay
    # --------------------------------------------------------

    mask_overlay = draw_mask_overlay(
        frame,
        mask,
        alpha=mask_alpha,
    )

    # --------------------------------------------------------
    # 2. Flow color visualization
    # --------------------------------------------------------

    flow_viz = safe_flow_visualization(
        flow
    )

    # Guarantee exact dimensions
    flow_viz = cv2.resize(
        flow_viz,
        (flow_w, flow_h),
        interpolation=cv2.INTER_NEAREST,
    )

    # --------------------------------------------------------
    # 3. Flow vectors
    # --------------------------------------------------------

    vectors = draw_flow_vectors(
        frame,
        flow,
        step=vector_step,
        vector_scale=vector_scale,
    )

    # --------------------------------------------------------
    # 4. Original frame
    # --------------------------------------------------------

    original = frame.copy()

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    mask_overlay = put_label(
        mask_overlay,
        "SAM3 Hand Mask",
    )

    flow_viz = put_label(
        flow_viz,
        "Optical Flow",
    )

    vectors = put_label(
        vectors,
        "Flow Vectors",
    )

    original = put_label(
        original,
        "Original",
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    frames = [
        mask_overlay,
        flow_viz,
        vectors,
        original,
    ]

    for idx, img in enumerate(frames):

        if img.shape[:2] != (
            flow_h,
            flow_w,
        ):

            raise RuntimeError(
                f"Visualization panel {idx} has "
                f"wrong size: {img.shape[:2]}, "
                f"expected {(flow_h, flow_w)}"
            )

    # --------------------------------------------------------
    # 2 x 2 visualization
    # --------------------------------------------------------

    top = np.hstack(
        [
            mask_overlay,
            flow_viz,
        ]
    )

    bottom = np.hstack(
        [
            vectors,
            original,
        ]
    )

    result = np.vstack(
        [
            top,
            bottom,
        ]
    )

    return np.ascontiguousarray(
        result,
        dtype=np.uint8,
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Load masks
    # --------------------------------------------------------

    (
        left_masks,
        right_masks,
        frame_indices,
        mask_width,
        mask_height,
        mask_fps,
    ) = load_masks(
        args.masks
    )

    # --------------------------------------------------------
    # Validate mask lengths
    # --------------------------------------------------------

    num_mask_frames = min(
        len(left_masks),
        len(right_masks),
    )

    if len(left_masks) != len(right_masks):

        print(
            "WARNING: left_masks and right_masks "
            f"have different lengths: "
            f"{len(left_masks)} vs "
            f"{len(right_masks)}"
        )

    if len(frame_indices) < num_mask_frames:

        print(
            "WARNING: frame_indices shorter "
            "than mask arrays. Missing indices "
            "will use sequential indices."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = SeaRaftModel(
        args.model,
        args.device,
        args.iters,
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        args.video
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: {args.video}"
        )

    video_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    video_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    video_fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    video_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if video_fps <= 0:
        video_fps = (
            mask_fps
            if mask_fps > 0
            else 25.0
        )

    print("\n" + "=" * 70)
    print("Video")
    print("=" * 70)

    print(
        f"Resolution : "
        f"{video_width}x{video_height}"
    )

    print(
        f"FPS        : {video_fps:.2f}"
    )

    print(
        f"Frames     : {video_frames}"
    )

    print(
        f"Scale      : {args.video_scale}"
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = Path(
        args.video
    ).stem

    flow_path = (
        output_dir
        / f"{stem}_flow.npz"
    )

    flow_frames_dir = (
        output_dir
        / f"{stem}_flow_frames"
    )

    flow_frames_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Visualization writer
    #
    # IMPORTANT:
    # Do NOT use original width/height.
    #
    # The video is resized by video_scale.
    # We initialize the writer after generating
    # the first visualization frame.
    # --------------------------------------------------------

    writer = None

    viz_path = (
        output_dir
        / f"{stem}_flow_viz.mp4"
    )

    # --------------------------------------------------------
    # Processing
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Processing optical flow")
    print("=" * 70)

    flows = []
    processed_frame_indices = []

    total_time = 0.0
    processed = 0

    # --------------------------------------------------------
    # Read first frame
    # --------------------------------------------------------

    ret, prev_frame_original = cap.read()

    if not ret:

        raise RuntimeError(
            "Cannot read first video frame"
        )

    prev_frame = resize_frame(
        prev_frame_original,
        args.video_scale,
    )

    prev_idx = 0

    # --------------------------------------------------------
    # Process mask frames
    # --------------------------------------------------------

    for i in range(
        num_mask_frames
    ):

        if (
            args.max_frames > 0
            and processed >= args.max_frames
        ):
            break

        # ----------------------------------------------------
        # Frame index
        # ----------------------------------------------------

        try:

            frame_idx = int(
                frame_indices[i]
            )

        except Exception:

            frame_idx = i

        # ----------------------------------------------------
        # Ignore invalid frame index
        # ----------------------------------------------------

        if frame_idx < 0:

            print(
                f"Warning: invalid frame index "
                f"{frame_idx}, skipping."
            )

            continue

        # ----------------------------------------------------
        # We need a valid previous/current pair.
        #
        # If mask frame == 0, there is no previous frame
        # for optical flow. Skip it.
        # ----------------------------------------------------

        if frame_idx == 0:

            print(
                "Skipping frame 0: "
                "no previous frame for optical flow."
            )

            continue

        # ----------------------------------------------------
        # Read requested frame
        # ----------------------------------------------------

        if frame_idx != prev_idx + 1:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_idx,
            )

            ret, curr_frame_original = cap.read()

            if not ret:

                print(
                    f"Frame {frame_idx} "
                    "not available, skipping."
                )

                continue

        else:

            ret, curr_frame_original = cap.read()

            if not ret:

                print(
                    f"Frame {frame_idx} "
                    "not available, skipping."
                )

                continue

        curr_frame = resize_frame(
            curr_frame_original,
            args.video_scale,
        )

        # ----------------------------------------------------
        # Combine masks
        # ----------------------------------------------------

        h, w = curr_frame.shape[:2]

        left_mask = (
            left_masks[i]
            if i < len(left_masks)
            else None
        )

        right_mask = (
            right_masks[i]
            if i < len(right_masks)
            else None
        )

        combined_mask = combine_masks(
            left_mask,
            right_mask,
            h,
            w,
        )

        # ----------------------------------------------------
        # Compute flow
        # ----------------------------------------------------

        try:

            start = time.perf_counter()

            flow = model.compute_masked_flow(
                prev_frame,
                curr_frame,
                combined_mask
                if np.any(combined_mask)
                else None,
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            total_time += elapsed

        except Exception as exc:

            print(
                f"\nERROR processing frame "
                f"{frame_idx}: {exc}"
            )

            # Do not destroy processing pipeline.
            prev_frame = curr_frame
            prev_idx = frame_idx

            continue

        # ----------------------------------------------------
        # Validate flow
        # ----------------------------------------------------

        if flow is None:

            print(
                f"WARNING: frame {frame_idx} "
                "returned None flow."
            )

            prev_frame = curr_frame
            prev_idx = frame_idx

            continue

        # ----------------------------------------------------
        # Sanitize flow
        # ----------------------------------------------------

        flow = np.nan_to_num(
            flow,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(
            np.float32,
            copy=False,
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        flows.append(
            flow.copy()
        )

        processed_frame_indices.append(
            frame_idx
        )

        processed += 1

        # ----------------------------------------------------
        # Save individual flow
        # ----------------------------------------------------

        flow_np = flow.astype(
            np.float16
        )

        np.save(
            flow_frames_dir
            / f"flow_{frame_idx:06d}.npy",
            flow_np,
        )

        # ----------------------------------------------------
        # Visualization
        # ----------------------------------------------------

        if args.visualize:

            try:

                viz = visualize_flow_and_mask(
                    curr_frame,
                    flow,
                    combined_mask,
                    vector_step=args.vector_step,
                    vector_scale=args.vector_scale,
                    mask_alpha=args.mask_alpha,
                )

                # ------------------------------------------------
                # Initialize writer using ACTUAL viz dimensions.
                #
                # This is the important fix.
                # ------------------------------------------------

                if writer is None:

                    viz_h, viz_w = (
                        viz.shape[:2]
                    )

                    fourcc = (
                        cv2.VideoWriter_fourcc(
                            *"mp4v"
                        )
                    )

                    writer = cv2.VideoWriter(
                        str(viz_path),
                        fourcc,
                        video_fps,
                        (viz_w, viz_h),
                    )

                    if not writer.isOpened():

                        raise RuntimeError(
                            "Could not open "
                            f"VideoWriter: {viz_path}"
                        )

                    print(
                        f"Visualization: "
                        f"{viz_path}"
                    )

                    print(
                        f"Visualization size: "
                        f"{viz_w}x{viz_h}"
                    )

                # ------------------------------------------------
                # Guarantee writer dimensions.
                # ------------------------------------------------

                expected_w = int(
                    writer.get(
                        cv2.CAP_PROP_FRAME_WIDTH
                    )
                )

                expected_h = int(
                    writer.get(
                        cv2.CAP_PROP_FRAME_HEIGHT
                    )
                )

                # Some OpenCV backends return 0,
                # so use actual viz dimensions in that case.
                if (
                    expected_w <= 0
                    or expected_h <= 0
                ):

                    expected_h, expected_w = (
                        viz.shape[:2]
                    )

                if viz.shape[:2] != (
                    expected_h,
                    expected_w,
                ):

                    viz = cv2.resize(
                        viz,
                        (
                            expected_w,
                            expected_h,
                        ),
                        interpolation=cv2.INTER_AREA,
                    )

                viz = np.ascontiguousarray(
                    viz,
                    dtype=np.uint8,
                )

                writer.write(
                    viz
                )

            except Exception as exc:

                print(
                    f"\nWARNING: visualization "
                    f"failed at frame {frame_idx}: "
                    f"{type(exc).__name__}: {exc}"
                )

                # Visualization failure should NOT
                # stop optical-flow extraction.

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            processed == 1
            or processed % 10 == 0
        ):

            avg_fps = (
                processed / total_time
                if total_time > 0
                else 0.0
            )

            mask_pixels = int(
                np.count_nonzero(
                    combined_mask
                )
            )

            print(
                f"Frame {frame_idx:06d} | "
                f"Processed {processed:06d} | "
                f"Flow {flow.shape} | "
                f"Mask px {mask_pixels} | "
                f"Avg FPS {avg_fps:.2f}"
            )

        # ----------------------------------------------------
        # Update previous frame
        # ----------------------------------------------------

        prev_frame = curr_frame
        prev_idx = frame_idx

    # ========================================================
    # Cleanup
    # ========================================================

    cap.release()

    if writer is not None:

        writer.release()

        print(
            f"\nVisualization saved: "
            f"{viz_path}"
        )

    # ========================================================
    # Save all flows
    # ========================================================

    if flows:

        try:

            flow_array = np.stack(
                flows,
                axis=0,
            )

            np.savez_compressed(
                flow_path,
                flow=flow_array,
                frame_indices=np.asarray(
                    processed_frame_indices,
                    dtype=np.int64,
                ),
                fps=video_fps,
                video_scale=args.video_scale,
                original_width=video_width,
                original_height=video_height,
            )

            print(
                f"\nFlow saved: "
                f"{flow_path}"
            )

            print(
                f"Flow shape: "
                f"{flow_array.shape}"
            )

        except Exception as exc:

            print(
                "\nWARNING: Could not stack "
                f"all flows: {exc}"
            )

            print(
                "Individual flow files were "
                "already saved."
            )

    else:

        print(
            "\nWARNING: No flow frames "
            "were successfully processed."
        )

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Processed     : {processed}"
    )

    print(
        f"Flow directory: {flow_frames_dir}"
    )

    if flows:

        print(
            f"Flow NPZ      : {flow_path}"
        )

    if args.visualize and writer is not None:

        print(
            f"Visualization : {viz_path}"
        )

    if total_time > 0:

        print(
            f"Average FPS   : "
            f"{processed / total_time:.2f}"
        )

    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()