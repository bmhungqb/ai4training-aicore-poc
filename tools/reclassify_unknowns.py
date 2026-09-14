#!/usr/bin/env python3
"""Re-classify UNKNOWN segments in worker_segments.json.

1. Fixes segments where VLM provided a valid operation name but string matching failed
   due to parentheses spacing.
2. Re-runs VLM for segments that had upstream 502 errors, with auto-retry.
3. Merges consecutive identical operations, updates worker_segments.json, timeline_debug.json,
   and re-cuts clips in cuts/.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.analysis.segment_classify import (
    SegmentClassifier,
    cut_worker_segments,
    dump_timeline_debug,
)
from src.manifest import scene_op_name


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\(\)\[\]]", " ", s)).strip().lower()


def resolve_unknowns(cd: str = "1") -> None:
    data_dir = Path(f"data/{cd}")
    manifest_path = data_dir / "expert_scenes" / "selected_frames.json"
    worker_video = data_dir / "worker.mp4"
    out_dir = data_dir / "worker_segments"
    frames_dir = data_dir / "worker_frames"
    cuts_dir = out_dir / "cuts"
    segments_path = out_dir / "worker_segments.json"
    timeline_path = out_dir / "timeline_debug.json"
    action_segments_path = data_dir / "kinematic" / "cam-03_20260808_023207_cut_1_28-4_03" / "action_segments.json"

    runner = SegmentClassifier(
        manifest_path=manifest_path,
        video_path=worker_video,
        out_dir=out_dir,
        frames_dir=frames_dir,
        action_segments_path=action_segments_path,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ordered_ops = [scene_op_name(sd) for _, sd in runner.ordered_scenes]

    # Get dedup expert ref frames
    all_expert_ref_paths: list[str] = []
    for _, sd in runner.ordered_scenes:
        all_expert_ref_paths.extend(sd.get("frames", []))
    seen = set()
    dedup_expert_refs = []
    for p in all_expert_ref_paths:
        if p not in seen:
            seen.add(p)
            dedup_expert_refs.append(p)

    report = runner._load_action_segments()

    # Load current segments
    current_data = json.loads(segments_path.read_text(encoding="utf-8"))
    segs = current_data.get("segments", [])

    print(f"Loaded {len(segs)} segments. Inspecting UNKNOWNs...")
    updated_count = 0

    for i, s in enumerate(segs):
        if s.get("operation_name") != "UNKNOWN":
            continue

        t0, t1 = s["start_time"], s["end_time"]
        mo = s.get("model_output", {})
        raw_op = mo.get("operation_name", "UNKNOWN")

        # Step 1: Check if raw_op is already a known operation via normalized matching
        matched_op = "UNKNOWN"
        norm_raw = _norm(raw_op)
        for op in ordered_ops:
            norm_op = _norm(op)
            if norm_raw == norm_op or norm_raw in norm_op or norm_op in norm_raw:
                matched_op = op
                break

        if matched_op != "UNKNOWN":
            print(f"Segment [{t0:.1f}s - {t1:.1f}s]: Resolved via normalization: {raw_op!r} -> {matched_op!r}")
            s["operation_name"] = matched_op
            s["off_standard"] = False
            s["off_standard_desc"] = ""
            updated_count += 1
            continue

        # Step 2: Truly unknown / upstream error -> re-run VLM with fallback
        print(f"\nSegment [{t0:.1f}s - {t1:.1f}s]: Needs VLM re-run (previous raw_op was {raw_op!r})...")
        k_seg_candidates = [
            ks for ks in report.segments if abs(ks.start_time_s - t0) < 0.2
        ]
        if not k_seg_candidates:
            print(f"  Warning: No kinematic segment matching {t0:.1f}s, skipping.")
            continue

        k_seg = k_seg_candidates[0]
        # Attempt classify with retries and smaller frame count if needed
        res = None
        for attempt in range(1, 4):
            print(f"  Attempt {attempt}/3 calling VLM for [{t0:.1f}s - {t1:.1f}s]...")
            try:
                # Use fewer expert ref frames to prevent prompt token bloat
                expert_refs = dedup_expert_refs[:4]
                res = runner._classify_segment(k_seg, expert_refs)
                if res and res.get("operation_name") != "UNKNOWN":
                    print(f"  -> Successfully classified as: {res['operation_name']}")
                    break
            except Exception as e:
                print(f"  Attempt {attempt} error: {e}")

        if res and res.get("operation_name") != "UNKNOWN":
            s.update(res)
            updated_count += 1
        else:
            print(f"  Could not classify segment [{t0:.1f}s - {t1:.1f}s]. Keeping as {s.get('operation_name')}.")

    print(f"\nTotal segments updated: {updated_count}")

    # Re-merge consecutive segments with identical operation names (except UNKNOWN)
    merged_segments: list[dict] = []
    for s in segs:
        if (
            merged_segments
            and merged_segments[-1]["operation_name"] == s["operation_name"]
            and s["operation_name"] != "UNKNOWN"
        ):
            prev = merged_segments[-1]
            prev["end_time"] = s["end_time"]
            prev["worker_duration_s"] = round(prev["end_time"] - prev["start_time"], 2)
            prev["n_vlm_calls"] += s.get("n_vlm_calls", 1)
            prev["cost_usd"] = round(prev.get("cost_usd", 0.0) + s.get("cost_usd", 0.0), 5)
            prev["worker_frame_count"] = prev.get("worker_frame_count", 0) + s.get("worker_frame_count", 0)
            if s.get("off_standard"):
                prev["off_standard"] = True
                if s.get("off_standard_desc"):
                    prev["off_standard_desc"] = (
                        (prev.get("off_standard_desc", "") + "; " + s["off_standard_desc"]).strip("; ")
                    )
        else:
            merged_segments.append(dict(s))

    current_data["segments"] = merged_segments
    current_data["total_cost_usd"] = round(sum(s.get("cost_usd", 0.0) for s in merged_segments), 5)
    current_data["total_vlm_calls"] = sum(s.get("n_vlm_calls", 1) for s in merged_segments)

    # Save updated worker_segments.json
    segments_path.write_text(json.dumps(current_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(merged_segments)} merged segments to {segments_path}")

    # Dump timeline debug
    dump_timeline_debug(current_data, timeline_path)
    print(f"Updated timeline debug -> {timeline_path}")

    # Re-cut worker segments
    print("Re-cutting clips into cuts/...")
    cut_worker_segments(segments_path, video_path=worker_video, out_dir=cuts_dir)
    print("Done!")


if __name__ == "__main__":
    resolve_unknowns("1")
