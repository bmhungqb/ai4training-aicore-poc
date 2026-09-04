#!/usr/bin/env python3

import argparse
import gc
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SAM3 hand/arm segmentation -> masks for SEA-RAFT"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input video path",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory",
    )

    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "left human hand",
            "right human hand",
        ],
        help="Text prompts for SAM3",
    )

    parser.add_argument(
        "--sam-threshold",
        type=float,
        default=0.5,
        help="SAM3 detection score threshold",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of processed frames",
    )

    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every N-th frame",
    )

    parser.add_argument(
        "--sam-reset-interval",
        type=int,
        default=20,
        help="Frames per SAM3 session before resetting VRAM cache (default: 20)",
    )

    parser.add_argument(
        "--frame-by-frame",
        action="store_true",
        help="Pure frame-by-frame segmentation without holding cross-frame memory",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization video",
    )

    return parser.parse_args()


# ============================================================
# Utilities
# ============================================================

def to_numpy(value):
    """
    Convert torch tensor / list / numpy array to numpy array.

    None remains None.
    """
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        return value

    if torch.is_tensor(value):
        return value.detach().cpu().numpy()

    try:
        return np.asarray(value)
    except Exception:
        return None


def normalize_mask(mask, height, width):
    """
    Convert SAM mask into H x W boolean numpy array.
    """
    if mask is None:
        return None

    mask = to_numpy(mask)

    if mask is None:
        return None

    mask = np.squeeze(mask)

    if mask.ndim != 2:
        return None

    if mask.dtype != np.bool_:
        mask = mask > 0.5
    else:
        mask = mask.astype(bool)

    if mask.shape != (height, width):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    return mask


def normalize_prompt(prompt):
    """
    Convert arbitrary prompt text into a canonical side.

    Examples:
        left human hand          -> left
        left hand                -> left
        left human arm and hand  -> left

        right human hand         -> right
        right hand               -> right
        right human arm and hand -> right
    """

    if prompt is None:
        return None

    prompt = str(prompt).lower().strip()

    if "left" in prompt:
        return "left"

    if "right" in prompt:
        return "right"

    return None


def build_object_prompt_map(prompt_to_obj_ids):
    """
    Build:

        object_id -> "left" / "right"

    from SAM3 prompt_to_obj_ids.
    """

    mapping = {}

    if prompt_to_obj_ids is None:
        return mapping

    if not isinstance(prompt_to_obj_ids, dict):
        return mapping

    for prompt, ids in prompt_to_obj_ids.items():

        side = normalize_prompt(prompt)

        if side is None:
            continue

        if ids is None:
            continue

        # Make iterable if necessary
        if np.isscalar(ids):
            ids = [ids]

        for obj_id in ids:

            try:
                if hasattr(obj_id, "item"):
                    obj_id = obj_id.item()

                obj_id = int(obj_id)

                mapping[obj_id] = side

            except Exception:
                continue

    return mapping


# ============================================================
# Detection selection
# ============================================================

def select_best_arm_detections(
    object_ids,
    scores,
    boxes,
    masks,
    object_prompt_map,
    sam_threshold,
):
    """
    Select the highest-score detection for left/right.

    Returns:

        {
            "left": {
                "score": ...,
                "box": ...,
                "mask": ...
            } or None,

            "right": {
                "score": ...,
                "box": ...,
                "mask": ...
            } or None
        }
    """

    best = {
        "left": None,
        "right": None,
    }

    if object_ids is None:
        return best

    object_ids = np.asarray(object_ids).reshape(-1)

    if len(object_ids) == 0:
        return best

    # Safely flatten scores
    if scores is not None:
        scores = np.asarray(scores).reshape(-1)

    # Boxes
    if boxes is not None:
        boxes = np.asarray(boxes)

    # Masks
    if masks is not None:
        masks = np.asarray(masks)

    for i, obj_id in enumerate(object_ids):

        try:
            obj_id = int(obj_id)
        except Exception:
            continue

        side = object_prompt_map.get(obj_id)

        if side not in ("left", "right"):
            continue

        # -----------------------------
        # Score
        # -----------------------------

        if scores is not None and i < len(scores):
            try:
                score = float(scores[i])
            except Exception:
                score = 0.0
        else:
            score = 0.0

        if score < sam_threshold:
            continue

        # -----------------------------
        # Box
        # -----------------------------

        box = None

        if boxes is not None and i < len(boxes):
            box = boxes[i]

        # -----------------------------
        # Mask
        # -----------------------------

        mask = None

        if masks is not None and i < len(masks):
            mask = masks[i]

        # -----------------------------
        # Select best
        # -----------------------------

        if best[side] is None:
            best[side] = {
                "score": score,
                "box": box,
                "mask": mask,
            }

        elif score > best[side]["score"]:
            best[side] = {
                "score": score,
                "box": box,
                "mask": mask,
            }

    return best


# ============================================================
# Safe SAM3 inference
# ============================================================

def run_sam3_frame(
    model,
    processor,
    inference_session,
    frame_rgb,
    device,
    object_prompt_map,
    sam_threshold,
):
    """
    Run SAM3 on a single frame.

    IMPORTANT:
    This function NEVER raises because of malformed/None outputs.

    Returns:
        best_detections
        updated_object_prompt_map
    """

    height, width = frame_rgb.shape[:2]

    empty_result = {
        "left": None,
        "right": None,
    }

    inputs = None
    model_outputs = None

    try:

        # ----------------------------------------------------
        # Processor
        # ----------------------------------------------------

        inputs = processor(
            images=frame_rgb,
            device=device,
            return_tensors="pt",
        )

        # ----------------------------------------------------
        # SAM3 inference
        # ----------------------------------------------------

        with torch.inference_mode():

            model_outputs = model(
                inference_session=(
                    inference_session
                ),
                frame=(
                    inputs.pixel_values[0]
                ),
                reverse=False,
            )

        # ----------------------------------------------------
        # Postprocess
        # ----------------------------------------------------

        outputs = processor.postprocess_outputs(
            inference_session,
            model_outputs,
            original_sizes=(
                inputs.original_sizes
            ),
        )

        # SAM3 may return None
        if outputs is None:
            return empty_result, object_prompt_map

        # Some versions can return non-dict-like objects.
        # Try converting safely.
        try:
            object_ids_raw = outputs.get("object_ids")
            scores_raw = outputs.get("scores")
            boxes_raw = outputs.get("boxes")
            masks_raw = outputs.get("masks")
            prompt_to_obj_ids = outputs.get("prompt_to_obj_ids")
        except Exception:
            return empty_result, object_prompt_map

        # ----------------------------------------------------
        # Convert
        # ----------------------------------------------------

        object_ids = to_numpy(object_ids_raw)
        scores = to_numpy(scores_raw)
        boxes = to_numpy(boxes_raw)
        masks = to_numpy(masks_raw)

        # ----------------------------------------------------
        # Update prompt mapping
        # ----------------------------------------------------

        if prompt_to_obj_ids is not None:

            new_mapping = build_object_prompt_map(
                prompt_to_obj_ids
            )

            if new_mapping:
                object_prompt_map.update(new_mapping)

        # ----------------------------------------------------
        # Select best
        # ----------------------------------------------------

        best = select_best_arm_detections(
            object_ids=object_ids,
            scores=scores,
            boxes=boxes,
            masks=masks,
            object_prompt_map=object_prompt_map,
            sam_threshold=sam_threshold,
        )

        return best, object_prompt_map

    except Exception as e:

        # Do not crash the whole video.
        print(f"    SAM3 warning: {type(e).__name__}: {e}")

        return empty_result, object_prompt_map

    finally:

        # Release temporary tensors
        if inputs is not None:
            del inputs

        if model_outputs is not None:
            del model_outputs


# ============================================================
# Visualization
# ============================================================

def overlay_mask(
    image,
    mask,
    color,
    alpha=0.5,
):
    """
    Overlay binary mask on BGR image.
    """

    if mask is None:
        return image

    if not np.any(mask):
        return image

    color_layer = np.zeros_like(image)
    color_layer[:, :] = color

    image[mask] = cv2.addWeighted(
        image[mask],
        1.0 - alpha,
        color_layer[mask],
        alpha,
        0,
    )

    return image


def draw_detection(
    image,
    detection,
    label,
):
    """
    Draw bbox + score.
    """

    if detection is None:
        return image

    box = detection.get("box")

    if box is None:
        return image

    try:
        box = np.asarray(box).reshape(-1)

        if len(box) < 4:
            return image

        x1, y1, x2, y2 = map(
            lambda x: int(round(float(x))),
            box[:4],
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        score = detection.get("score", 0.0)

        text = f"{label}: {score:.2f}"

        cv2.putText(
            image,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    except Exception:
        pass

    return image


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if args.frame_step < 1:
        raise ValueError("--frame-step must be >= 1")

    if args.sam_threshold < 0.0 or args.sam_threshold > 1.0:
        raise ValueError("--sam-threshold must be between 0 and 1")

    input_path = Path(args.input)
    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available()
        else torch.float32
    )

    print("=" * 70)
    print("SAM3 Hand/Arm Segmentation")
    print("=" * 70)

    print(f"Input : {input_path}")
    print(f"Output: {output_dir}")
    print(f"Prompts: {args.prompts}")
    print(f"Threshold: {args.sam_threshold}")
    print(f"Frame step: {args.frame_step}")
    print(f"Device: {device}")
    print(f"Dtype : {dtype}")

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {input_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print()
    print(
        f"Video: {width}x{height} | "
        f"{frame_count} frames | "
        f"{fps:.2f} FPS"
    )

    # --------------------------------------------------------
    # Load SAM3
    # --------------------------------------------------------

    print()
    print("Loading SAM3...")

    model = Sam3VideoModel.from_pretrained(
        "facebook/sam3"
    ).to(
        device=device,
        dtype=dtype,
    )

    model.eval()

    processor = Sam3VideoProcessor.from_pretrained(
        "facebook/sam3"
    )

    print("SAM3 loaded.")

    # --------------------------------------------------------
    # Init video session helper
    # --------------------------------------------------------

    def _create_fresh_session():
        sess = processor.init_video_session(
            inference_device=device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=dtype,
        )
        sess = processor.add_text_prompt(
            inference_session=sess,
            text=args.prompts,
        )
        return sess

    inference_session = _create_fresh_session()
    reset_interval = 1 if getattr(args, "frame_by_frame", False) else max(1, getattr(args, "sam_reset_interval", 20))
    mode_str = "Frame-by-Frame (Stateless)" if getattr(args, "frame_by_frame", False) else f"Sliding Memory (Reset every {reset_interval} frames)"
    print(f"Session ready. SAM3 Mode: {mode_str}")

    # --------------------------------------------------------
    # Visualization writer
    # --------------------------------------------------------

    writer = None

    if args.visualize:

        viz_path = (
            output_dir
            / f"{input_path.stem}_viz.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(viz_path),
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():
            writer.release()
            writer = None
            print(
                f"WARNING: Cannot create visualization: "
                f"{viz_path}"
            )
        else:
            print(
                f"Visualization: {viz_path}"
            )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    left_masks = []
    right_masks = []

    left_scores = []
    right_scores = []

    frame_indices = []

    # IMPORTANT:
    # object IDs are persistent across frames.
    object_prompt_map = {}

    # --------------------------------------------------------
    # Processing
    # --------------------------------------------------------

    frame_idx = 0
    processed_count = 0
    failed_count = 0

    print()
    print("Processing video...")
    print("-" * 70)

    while True:

        ret, frame_bgr = cap.read()

        if not ret:
            break

        current_frame_idx = frame_idx

        frame_idx += 1

        # ----------------------------------------------------
        # Frame step
        # ----------------------------------------------------

        if current_frame_idx % args.frame_step != 0:
            continue

        # ----------------------------------------------------
        # Max frames
        # ----------------------------------------------------

        if (
            args.max_frames is not None
            and processed_count >= args.max_frames
        ):
            break

        # Periodic memory reset to prevent GPU VRAM accumulation on long videos
        if processed_count > 0 and processed_count % reset_interval == 0:
            del inference_session
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            inference_session = _create_fresh_session()
            object_prompt_map = {}

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        # ----------------------------------------------------
        # SAM3
        # ----------------------------------------------------

        try:

            best_detections, object_prompt_map = (
                run_sam3_frame(
                    model=model,
                    processor=processor,
                    inference_session=inference_session,
                    frame_rgb=frame_rgb,
                    device=device,
                    object_prompt_map=object_prompt_map,
                    sam_threshold=args.sam_threshold,
                )
            )

        except Exception as e:

            failed_count += 1

            print(
                f"Frame {current_frame_idx} failed: "
                f"{type(e).__name__}: {e}"
            )

            best_detections = {
                "left": None,
                "right": None,
            }

        # ----------------------------------------------------
        # Left
        # ----------------------------------------------------

        left_mask = None
        left_score = 0.0

        left_det = best_detections.get("left")

        if left_det is not None:

            left_mask = normalize_mask(
                left_det.get("mask"),
                height,
                width,
            )

            left_score = float(
                left_det.get("score", 0.0)
            )

        # ----------------------------------------------------
        # Right
        # ----------------------------------------------------

        right_mask = None
        right_score = 0.0

        right_det = best_detections.get("right")

        if right_det is not None:

            right_mask = normalize_mask(
                right_det.get("mask"),
                height,
                width,
            )

            right_score = float(
                right_det.get("score", 0.0)
            )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        left_masks.append(left_mask)
        right_masks.append(right_mask)

        left_scores.append(left_score)
        right_scores.append(right_score)

        frame_indices.append(
            current_frame_idx
        )

        processed_count += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            processed_count == 1
            or processed_count % 10 == 0
        ):

            print(
                f"Frame {current_frame_idx:06d} "
                f"({processed_count:06d}) | "
                f"L={left_score:.3f} "
                f"R={right_score:.3f} | "
                f"Lmask={left_mask is not None} "
                f"Rmask={right_mask is not None}"
            )

        # ----------------------------------------------------
        # Visualization
        # ----------------------------------------------------

        if writer is not None:

            vis = frame_bgr.copy()

            # Left mask
            if left_mask is not None:

                vis = overlay_mask(
                    vis,
                    left_mask,
                    color=(0, 0, 255),
                    alpha=0.5,
                )

            # Right mask
            if right_mask is not None:

                vis = overlay_mask(
                    vis,
                    right_mask,
                    color=(0, 255, 0),
                    alpha=0.5,
                )

            # BBoxes
            vis = draw_detection(
                vis,
                left_det,
                "LEFT",
            )

            vis = draw_detection(
                vis,
                right_det,
                "RIGHT",
            )

            cv2.putText(
                vis,
                (
                    f"Frame {current_frame_idx} | "
                    f"L={left_score:.2f} "
                    f"R={right_score:.2f}"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

            writer.write(vis)

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        del frame_rgb
        del frame_bgr

        if (
            processed_count % 10 == 0
            and torch.cuda.is_available()
        ):

            gc.collect()
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Release video
    # --------------------------------------------------------

    cap.release()

    if writer is not None:
        writer.release()

    # --------------------------------------------------------
    # Save NPZ
    # --------------------------------------------------------

    masks_path = (
        output_dir
        / f"{input_path.stem}_masks.npz"
    )

    print()
    print("Saving masks...")

    np.savez_compressed(
        masks_path,

        left_masks=np.array(
            left_masks,
            dtype=object,
        ),

        right_masks=np.array(
            right_masks,
            dtype=object,
        ),

        left_scores=np.asarray(
            left_scores,
            dtype=np.float32,
        ),

        right_scores=np.asarray(
            right_scores,
            dtype=np.float32,
        ),

        frame_indices=np.asarray(
            frame_indices,
            dtype=np.int64,
        ),

        fps=np.float32(fps),

        width=np.int32(width),

        height=np.int32(height),
    )

    print(
        f"Saved: {masks_path}"
    )

    # --------------------------------------------------------
    # Save frames + masks
    # --------------------------------------------------------

    print()
    print("Saving frames with masks for SEA-RAFT...")

    frames_dir = (
        output_dir
        / f"{input_path.stem}_frames"
    )

    frames_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # IMPORTANT:
    # We seek to the exact frame index instead of
    # reading sequentially. This fixes frame-step alignment.

    cap = cv2.VideoCapture(
        str(input_path)
    )

    if not cap.isOpened():
        print(
            "WARNING: Cannot reopen video for frame export."
        )
    else:

        for i, (
            frame_number,
            left_mask,
            right_mask,
        ) in enumerate(
            zip(
                frame_indices,
                left_masks,
                right_masks,
            )
        ):

            frame_number = int(
                frame_number
            )

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number,
            )

            ret, frame = cap.read()

            if not ret:
                print(
                    f"WARNING: Cannot read frame "
                    f"{frame_number}"
                )
                continue

            # ------------------------------------------------
            # Save original frame
            # ------------------------------------------------

            frame_path = (
                frames_dir
                / f"frame_{frame_number:06d}.jpg"
            )

            cv2.imwrite(
                str(frame_path),
                frame,
            )

            # ------------------------------------------------
            # Combined mask
            #
            # 0 = background
            # 1 = left
            # 2 = right
            # ------------------------------------------------

            if (
                left_mask is not None
                or right_mask is not None
            ):

                combined_mask = np.zeros(
                    (height, width),
                    dtype=np.uint8,
                )

                if left_mask is not None:

                    combined_mask[
                        left_mask
                    ] = 1

                if right_mask is not None:

                    combined_mask[
                        right_mask
                    ] = 2

                mask_path = (
                    frames_dir
                    / f"mask_{frame_number:06d}.png"
                )

                cv2.imwrite(
                    str(mask_path),
                    combined_mask,
                )

        cap.release()

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del inference_session
    del model
    del processor

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Input frames       : {frame_count}"
    )

    print(
        f"Processed frames    : {processed_count}"
    )

    print(
        f"Failed frames       : {failed_count}"
    )

    print(
        f"Left detections     : "
        f"{sum(x is not None for x in left_masks)}"
    )

    print(
        f"Right detections    : "
        f"{sum(x is not None for x in right_masks)}"
    )

    print(
        f"NPZ                 : {masks_path}"
    )

    print(
        f"Frames              : {frames_dir}"
    )

    if writer is not None:
        print(
            f"Visualization       : "
            f"{output_dir / f'{input_path.stem}_viz.mp4'}"
        )

    print()
    print("SEA-RAFT input:")
    print(
        f"  frames = {frames_dir}"
    )
    print(
        f"  masks  = {frames_dir}/mask_*.png"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()