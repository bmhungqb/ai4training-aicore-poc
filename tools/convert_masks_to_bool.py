#!/usr/bin/env python3
"""
convert_masks_to_bool.py

Convert old object-dtype masks .npz files (shape=(N,) object array containing None or bool 2D arrays)
into the new bool-dtype format (shape=(N, H, W) bool) so that stream_mask_frames() can stream
them frame-by-frame without loading the full array into RAM.

Usage:
    python tools/convert_masks_to_bool.py                  # convert all masks in data/
    python tools/convert_masks_to_bool.py --dry-run        # preview what would be converted
    python tools/convert_masks_to_bool.py --cd 8           # only convert công đoạn 8
"""
from __future__ import annotations
import argparse
import gc
import zipfile
from pathlib import Path

import numpy as np


def needs_conversion(masks_path: Path) -> tuple[bool, str]:
    """Check if a masks.npz needs conversion. Returns (needs_convert, reason)."""
    try:
        with zipfile.ZipFile(masks_path, "r") as z:
            with z.open("left_masks.npy") as fp:
                magic = fp.read(6)
                if magic != b"\x93NUMPY":
                    return True, "invalid magic"
                major = fp.read(1)[0]
                fp.read(1)
                hlen = int.from_bytes(fp.read(2 if major == 1 else 4), "little")
                hdr = eval(fp.read(hlen).decode("ascii"))
                shape = hdr["shape"]
                dtype = np.dtype(hdr["descr"])
                if dtype == object:
                    return True, f"object dtype, shape={shape}"
                if len(shape) != 3:
                    return True, f"unexpected shape ndim={len(shape)}, shape={shape}"
                return False, f"already bool/numeric, shape={shape}"
    except Exception as e:
        return True, f"error reading: {e}"


def convert_masks(masks_path: Path, dry_run: bool = False) -> bool:
    """Convert a single masks.npz from object-dtype to (N, H, W) bool dtype.
    Returns True on success."""
    needs, reason = needs_conversion(masks_path)
    if not needs:
        print(f"  [SKIP] {masks_path.parent.name} — {reason}")
        return True

    print(f"  [CONVERT] {masks_path.parent.name} — {reason}")
    if dry_run:
        return True

    # Load with allow_pickle to get the object arrays
    try:
        data = np.load(masks_path, allow_pickle=True)
        left_obj = data["left_masks"]    # shape (N,) object or (N, H, W)
        right_obj = data["right_masks"]  # shape (N,) object or (N, H, W)
        left_scores = data.get("left_scores", None)
        right_scores = data.get("right_scores", None)
        frame_indices = data.get("frame_indices", None)
        fps_val = data.get("fps", np.float32(25.0))
        width_val = data.get("width", None)
        height_val = data.get("height", None)
    except Exception as e:
        print(f"    ERROR loading {masks_path}: {e}")
        return False

    N = len(left_obj)
    if N == 0:
        print(f"    WARNING: empty masks file {masks_path}, skipping")
        return True

    # Detect H, W from the first non-None mask
    H, W = None, None
    for i in range(N):
        m = left_obj[i] if left_obj[i] is not None else right_obj[i]
        if m is not None and isinstance(m, np.ndarray) and m.ndim == 2:
            H, W = m.shape
            break

    if H is None or W is None:
        # Try from metadata
        if height_val is not None and width_val is not None:
            H, W = int(height_val), int(width_val)
        else:
            print(f"    ERROR: cannot determine H, W for {masks_path}")
            return False

    print(f"    N={N}, H={H}, W={W} → allocating {N * H * W * 2 / 1024 / 1024:.1f} MB bool arrays")

    # Build dense bool arrays
    left_arr = np.zeros((N, H, W), dtype=bool)
    right_arr = np.zeros((N, H, W), dtype=bool)

    for i in range(N):
        lm = left_obj[i]
        rm = right_obj[i]
        if lm is not None and isinstance(lm, np.ndarray) and lm.ndim == 2:
            if lm.shape == (H, W):
                left_arr[i] = lm.astype(bool)
            else:
                import cv2
                left_arr[i] = cv2.resize(lm.astype(np.uint8), (W, H),
                                          interpolation=cv2.INTER_NEAREST).astype(bool)
        if rm is not None and isinstance(rm, np.ndarray) and rm.ndim == 2:
            if rm.shape == (H, W):
                right_arr[i] = rm.astype(bool)
            else:
                import cv2
                right_arr[i] = cv2.resize(rm.astype(np.uint8), (W, H),
                                           interpolation=cv2.INTER_NEAREST).astype(bool)

    del left_obj, right_obj
    gc.collect()

    # Build extra arrays with fallbacks
    kwargs: dict = dict(
        left_masks=left_arr,
        right_masks=right_arr,
    )
    if left_scores is not None:
        kwargs["left_scores"] = np.asarray(left_scores, dtype=np.float32)
    if right_scores is not None:
        kwargs["right_scores"] = np.asarray(right_scores, dtype=np.float32)
    if frame_indices is not None:
        kwargs["frame_indices"] = np.asarray(frame_indices, dtype=np.int64)
    kwargs["fps"] = np.float32(fps_val)
    kwargs["width"] = np.int32(W)
    kwargs["height"] = np.int32(H)

    # Write back to same path (atomic via temp file)
    tmp_path = masks_path.with_suffix(".tmp.npz")
    try:
        np.savez_compressed(tmp_path, **kwargs)
        del left_arr, right_arr
        gc.collect()
        tmp_path.replace(masks_path)
        print(f"    ✓ Converted → {masks_path}")
        return True
    except Exception as e:
        print(f"    ERROR saving {masks_path}: {e}")
        tmp_path.unlink(missing_ok=True)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", help="Root data directory (default: data)")
    ap.add_argument("--cd", "--cong-doan", dest="cd", default=None,
                    help="Only convert a specific công đoạn (e.g. --cd 8)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview conversions without writing anything")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    if args.cd:
        masks_files = sorted((data_dir / str(args.cd)).glob("kinematic/*/*_masks.npz"))
    else:
        masks_files = sorted(data_dir.glob("*/kinematic/*/*_masks.npz"))

    if not masks_files:
        print("No masks files found.")
        return

    mode = "DRY RUN — " if args.dry_run else ""
    print(f"{mode}Found {len(masks_files)} masks file(s) to check.\n")

    ok, failed, skipped = 0, 0, 0
    for mf in masks_files:
        needs, _ = needs_conversion(mf)
        if not needs:
            skipped += 1
            print(f"  [SKIP] {mf.parent.name} — already streaming-compatible")
            continue
        success = convert_masks(mf, dry_run=args.dry_run)
        if success:
            ok += 1
        else:
            failed += 1

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Done: {ok} converted, {skipped} skipped, {failed} failed.")
    if failed > 0:
        raise SystemExit(f"{failed} file(s) failed to convert.")


if __name__ == "__main__":
    main()
