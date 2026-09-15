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
import io
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def _write_array(zf: zipfile.ZipFile, name: str, arr: np.ndarray) -> None:
    """Write a single small ndarray into an open zip as a .npy entry (in-memory, for
    small metadata arrays only — do not use for full (N,H,W) mask arrays)."""
    buf = io.BytesIO()
    np.save(buf, arr)
    zf.writestr(name, buf.getvalue())
    buf.close()


def _build_and_write_dense_side(zf: zipfile.ZipFile, name: str, obj_arr, N: int, H: int, W: int) -> None:
    """Build a dense (N, H, W) bool array from one side's object array of 2D masks/None.

    The dense array is backed by a disk-mapped temp .npy file (via np.lib.format.open_memmap)
    instead of a fresh in-RAM np.zeros allocation, so peak RAM only holds the source object
    array (already-decoded per-frame bool masks) plus small per-frame temporaries -- not a
    second full (N, H, W) copy in memory. The temp file is then streamed into the output zip
    in chunks and deleted.
    """
    tmp_fd, tmp_npy_path = tempfile.mkstemp(suffix=".npy")
    os.close(tmp_fd)
    try:
        dense = np.lib.format.open_memmap(tmp_npy_path, mode="w+", dtype=bool, shape=(N, H, W))
        for i in range(N):
            m = obj_arr[i]
            if m is not None and isinstance(m, np.ndarray) and m.ndim == 2:
                if m.shape == (H, W):
                    dense[i] = m.astype(bool)
                else:
                    import cv2
                    dense[i] = cv2.resize(m.astype(np.uint8), (W, H),
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
        dense.flush()
        del dense
        gc.collect()

        # Stream the memmapped .npy file straight into the zip without loading it whole.
        # force_zip64=True is required since these dense mask arrays commonly exceed 2GB.
        zinfo = zipfile.ZipInfo(name)
        zinfo.compress_type = zf.compression
        with open(tmp_npy_path, "rb") as f, zf.open(zinfo, "w", force_zip64=True) as zdst:
            chunk = f.read(1024 * 1024 * 16)
            while chunk:
                zdst.write(chunk)
                chunk = f.read(1024 * 1024 * 16)
    finally:
        try:
            os.unlink(tmp_npy_path)
        except OSError:
            pass


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

    # Open lazily; np.load on a .npz is a lazy zipfile reader, but each array key
    # is only decoded from pickle once accessed. We deliberately access left_masks
    # and right_masks one at a time (never both alive simultaneously) to keep peak
    # RAM to roughly one side's worth of object-array + dense-array data.
    try:
        data = np.load(masks_path, allow_pickle=True)
        left_scores = data.get("left_scores", None)
        right_scores = data.get("right_scores", None)
        frame_indices = data.get("frame_indices", None)
        fps_val = data.get("fps", np.float32(25.0))
        width_val = data.get("width", None)
        height_val = data.get("height", None)
    except Exception as e:
        print(f"    ERROR loading {masks_path}: {e}")
        return False

    # Detect N, H, W by peeking at left_masks only (freed right after).
    try:
        left_obj = data["left_masks"]  # shape (N,) object or (N, H, W)
    except Exception as e:
        print(f"    ERROR loading left_masks from {masks_path}: {e}")
        return False

    N = len(left_obj)
    if N == 0:
        print(f"    WARNING: empty masks file {masks_path}, skipping")
        del left_obj
        gc.collect()
        return True

    H, W = None, None
    for i in range(N):
        m = left_obj[i]
        if m is not None and isinstance(m, np.ndarray) and m.ndim == 2:
            H, W = m.shape
            break

    if (H is None or W is None) and height_val is not None and width_val is not None:
        H, W = int(height_val), int(width_val)

    print(f"    N={N}, H={H if H else '?'}, W={W if W else '?'} → streaming conversion (one side at a time)")

    # Extra arrays with fallbacks (small, safe to build eagerly)
    extras: dict = {}
    if left_scores is not None:
        extras["left_scores"] = np.asarray(left_scores, dtype=np.float32)
    if right_scores is not None:
        extras["right_scores"] = np.asarray(right_scores, dtype=np.float32)
    if frame_indices is not None:
        extras["frame_indices"] = np.asarray(frame_indices, dtype=np.int64)
    extras["fps"] = np.float32(fps_val)

    tmp_path = masks_path.with_suffix(".tmp.npz")
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # --- LEFT side: already loaded above ---
            if H is None or W is None:
                # Still unknown (no mask found in left); try right_masks to detect.
                right_obj_peek = data["right_masks"]
                for i in range(len(right_obj_peek)):
                    m = right_obj_peek[i]
                    if m is not None and isinstance(m, np.ndarray) and m.ndim == 2:
                        H, W = m.shape
                        break
                del right_obj_peek
                gc.collect()
                if H is None or W is None:
                    print(f"    ERROR: cannot determine H, W for {masks_path}")
                    return False

            _build_and_write_dense_side(zf, "left_masks.npy", left_obj, N, H, W)
            del left_obj
            gc.collect()

            # --- RIGHT side: load only now, after left is fully freed ---
            right_obj = data["right_masks"]
            _build_and_write_dense_side(zf, "right_masks.npy", right_obj, N, H, W)
            del right_obj
            gc.collect()

            extras["width"] = np.int32(W)
            extras["height"] = np.int32(H)
            for name, arr in extras.items():
                _write_array(zf, f"{name}.npy", arr)

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
