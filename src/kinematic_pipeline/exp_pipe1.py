#!/usr/bin/env python3
"""
exp_pipe1.py  —  Experiment Pipeline 1

Steps:
    1. SAM3       → segment hand masks per frame
    2. SEA-RAFT   → extract optical-flow vectors on hand masks
    3. action_segment_magnitude & direction → detect action boundaries using speed and direction fusion
    4. classify_action (VLM)    → classify each segment into 4 category types
    5. Visualize                → annotated video + summary plot + JSON report

Usage:
    python exp_pipe1.py --video videos/video_10s.mp4 --output output/pipe1/ [options]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Setup path for SEA-RAFT imports
SEGMENT_FLOW_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SEGMENT_FLOW_DIR))

from config import PipelineConfig
from pipeline_steps import (
    run_step1_segmentation,
    run_step2_optical_flow,
    run_step3_segmentation,
    run_step4_classify,
)
from visualization import run_step5_visualize

def parse_args():
    p = argparse.ArgumentParser(description="Pipeline 1: SAM3 → SEA-RAFT → Segment → Classify → Visualize")

    p.add_argument("--video",        required=True, help="Input video path")
    p.add_argument("--output",       required=True, help="Output directory")
    p.add_argument("--force",        action="store_true",
                   help="Re-run all steps even if intermediate files already exist")
    p.add_argument("--only-step12", "--stop-after-step2", dest="only_step12", action="store_true",
                   help="Only run Step 1 (SAM3) and Step 2 (SEA-RAFT Flow) and exit")
    p.add_argument("--recompute-segmentation", "--force-step3", dest="recompute_segmentation", action="store_true",
                   help="Force re-running Step 3 (magnitude segmentation) and Step 4+ while reusing cached masks & flow")
    p.add_argument("--mask", default=None,
                   help="Path to a ROI mask image (white=keep, black=ignore), same resolution as "
                        "--video. Restricts Step 1 (SAM3) hand-detection and Step 2 (SEA-RAFT) flow "
                        "to the masked region only — useful when the frame also shows another "
                        "worker/expert whose hands would otherwise be picked up as noise. If not "
                        "given, falls back to '<video_stem>.mask.png' next to --video if it exists, "
                        "otherwise runs on the full frame.")

    # SAM3
    p.add_argument("--sam-threshold", type=float, default=0.5)
    p.add_argument("--frame-step",    type=int,   default=1,
                   help="Process every Nth frame (1 = every frame)")
    p.add_argument("--max-frames",    type=int,   default=None)
    p.add_argument("--sam-reset-interval", type=int, default=20,
                   help="Frames per SAM3 session before resetting VRAM cache (default: 20)")
    p.add_argument("--frame-by-frame", action="store_true",
                   help="Pure frame-by-frame segmentation without holding cross-frame memory")

    # SEA-RAFT
    p.add_argument("--raft-model",    default="MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
                   help="SEA-RAFT HuggingFace model name")
    p.add_argument("--raft-iters",    type=int, default=12)
    p.add_argument("--raft-device",   default="cuda")
    p.add_argument("--resize-scale",  type=float, default=0.5,
                   help="Scale factor for flow inference (0.5 = 2x faster, standard in searaft.py)")

    # Segmentation & Fusion
    p.add_argument("--smooth-window", type=int,   default=5)
    p.add_argument("--min-distance",  type=float, default=0.2,
                   help="Min seconds between action boundaries")
    p.add_argument("--prominence",    type=float, default=0.1,
                   help="Relative prominence threshold (fraction of signal range)")
    p.add_argument("--margin",        type=int,   default=2)
    p.add_argument("--angle-tolerance", type=float, default=40.0,
                   help="Direction change threshold in degrees (default: 40). Higher = fewer direction boundaries (e.g. 60-90 for macro steps)")
    p.add_argument("--min-segment-len", type=int, default=8,
                   help="Minimum frames per segment before creating a boundary (default: 8 ≈ 0.27s@30fps). Higher = fewer, longer segments")
    p.add_argument("--min-speed", type=float, default=0.5,
                   help="Minimum global speed to calculate angle (default: 0.5)")
    
    # Weighted Multi-Modal Scoring (Normalized in [0.0, 1.0])
    p.add_argument("--fusion-threshold", type=float, default=0.40,
                   help="Minimum total evidence score in [0.0, 1.0] to accept a boundary cluster (default: 0.40). Set lower (e.g. 0.20) for micro actions, higher (e.g. 0.50) for macro steps")
    p.add_argument("--w-speed", type=float, default=0.25,
                   help="Weight for speed valley / slowdown pause (default: 0.25)")
    p.add_argument("--w-dir-left", type=float, default=0.18,
                   help="Weight for left hand direction shift (default: 0.18)")
    p.add_argument("--w-dir-right", type=float, default=0.18,
                   help="Weight for right hand direction shift (default: 0.18)")
    p.add_argument("--w-both-hands", type=float, default=0.10,
                   help="Bonus weight when both left and right hands shift direction together (default: 0.10)")
    p.add_argument("--w-idle", type=float, default=0.20,
                   help="Weight for IDLE->MOVING transition (default: 0.20)")
    p.add_argument("--w-accel", type=float, default=0.09,
                   help="Weight for kinematic acceleration peak factor (default: 0.09)")

    # Legacy presets / shortcuts
    p.add_argument("--require-both", "--strict-joint", dest="require_both", action="store_true", default=None,
                   help="Legacy preset: enforce fusion-threshold=0.40 (macro steps)")
    p.add_argument("--union-fusion", dest="require_both", action="store_false",
                   help="Legacy preset: enforce fusion-threshold=0.18 (micro steps)")

    # Classification
    p.add_argument("--classify-min-frames", type=int, default=5,
                   help="Minimum frames per segment for VLM classification (default: 5)")
    p.add_argument("--classify-max-frames", type=int, default=15,
                   help="Maximum frames per segment for VLM classification (default: 15)")
    p.add_argument("--no-classify",   action="store_true",
                   help="Skip VLM classification step (step 4)")
    p.add_argument("--phase2-json", type=str, default=None,
                   help="Path to phase 2 kinematic_segmentation.json to override classifications")
    p.add_argument("--vlm-model",     default="qwen/qwen3.7-flash")

    # Visualization
    p.add_argument("--no-video",      action="store_true",
                   help="Skip video output (step 5)")

    return p.parse_args()


def main():
    args = parse_args()
    config = PipelineConfig.from_args(args)
    
    video_path = config.video
    output_dir = config.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    if config.mask is None:
        default_mask = video_path.with_suffix("").with_suffix(".mask.png")
        if default_mask.exists():
            config.mask = default_mask
    if config.mask is not None and not config.mask.exists():
        print(f"WARNING: --mask {config.mask} not found, running on full frame")
        config.mask = None

    print("\n" + "=" * 70)
    print("PIPELINE 1: SAM3 → SEA-RAFT → Segment → Classify → Visualize")
    print("=" * 70)
    print(f"  Video  : {video_path}")
    print(f"  Output : {output_dir}")
    print(f"  Force  : {config.force}")
    print(f"  Mask   : {config.mask if config.mask else '(none — full frame)'}")

    t_total = time.time()

    # ── Step 1 ────────────────────────────────────────────────────────────────
    masks_path = run_step1_segmentation(video_path, output_dir, config)

    # ── Step 2 ────────────────────────────────────────────────────────────────
    flow_path = run_step2_optical_flow(video_path, masks_path, output_dir, config)

    if config.only_step12:
        print("\n" + "=" * 70)
        print(f"STEP 1 & 2 COMPLETE in {time.time() - t_total:.1f}s")
        print(f"  Masks: {masks_path}")
        print(f"  Flow : {flow_path}")
        print("=" * 70)
        return

    # ── Step 3 ────────────────────────────────────────────────────────────────
    boundaries, left_mags, right_mags, fps = run_step3_segmentation(
        video_path, flow_path, masks_path, output_dir, config)

    total_frames = len(left_mags)

    # ── Step 4 ────────────────────────────────────────────────────────────────
    if getattr(args, "phase2_json", None) and Path(args.phase2_json).exists():
        print(f"\n  [Info] Loading Phase 2 JSON for visualization: {args.phase2_json}")
        import json
        with open(args.phase2_json, "r", encoding="utf-8") as f:
            p2_data = json.load(f)
        
        p2_segments = p2_data.get("segments", [])
        classifications = []
        
        edges = [0] + sorted([int(b) for b in boundaries if 0 < int(b) < total_frames]) + [total_frames]
        edges = sorted(list(dict.fromkeys(edges)))
        n_segs = len(edges) - 1
        
        for i in range(n_segs):
            start_f = edges[i]
            end_f = edges[i + 1] - 1
            mid_t = ((start_f + end_f) / 2) / fps
            
            matched_p2 = None
            for p2_seg in p2_segments:
                if p2_seg.get("start_time", 0.0) <= mid_t <= p2_seg.get("end_time", 0.0):
                    matched_p2 = p2_seg
                    break
            
            if matched_p2:
                cls_name = matched_p2.get("operation_name", "unknown")
                desc = matched_p2.get("evidence", "")
                cost = matched_p2.get("cost_usd", 0.0)
            else:
                cls_name = "Unmerged / Unknown"
                desc = "This segment was not merged into a main operation in Phase 2."
                cost = 0.0
                
            classifications.append({
                "segment": i,
                "start_f": start_f,
                "end_f": end_f,
                "start_timestamp_sec": start_f / fps,
                "end_timestamp_sec": end_f / fps,
                "class": cls_name,
                "description": desc,
                "estimated_cost_usd": cost
            })
            
        # Merge adjacent segments with the same class
        merged_classifications = []
        new_boundaries = []
        
        for c in classifications:
            if not merged_classifications:
                c["segment"] = 0
                merged_classifications.append(c)
            else:
                last_c = merged_classifications[-1]
                if last_c["class"] == c["class"]:
                    last_c["end_f"] = c["end_f"]
                    last_c["end_timestamp_sec"] = c["end_timestamp_sec"]
                    last_c["estimated_cost_usd"] += c["estimated_cost_usd"]
                else:
                    new_boundaries.append(c["start_f"])
                    c["segment"] = len(merged_classifications)
                    merged_classifications.append(c)
                    
        classifications = merged_classifications
        boundaries = new_boundaries
    else:
        classifications = run_step4_classify(video_path, boundaries, total_frames, fps, output_dir, config)

    # ── Step 5 ────────────────────────────────────────────────────────────────
    run_step5_visualize(
        video_path, masks_path, flow_path,
        boundaries, left_mags, right_mags,
        classifications, fps, output_dir, config,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    edges = [0] + sorted([int(b) for b in boundaries if 0 < int(b) < total_frames]) + [total_frames]
    n_segs  = len(dict.fromkeys(edges)) - 1
    print("\n" + "=" * 70)
    print("PIPELINE 1 COMPLETE")
    print("=" * 70)
    print(f"  Total time  : {elapsed:.1f}s")
    print(f"  Segments    : {n_segs}")
    if classifications:
        for r in classifications:
            st = r.get("start_timestamp_sec", r.get("start_time_s", 0.0))
            et = r.get("end_timestamp_sec", r.get("end_time_s", 0.0))
            cls_name = r.get("class", "unknown")
            desc = r.get("description", "")
            seg_num = r.get("segment", 0) + 1
            print(f"  S{seg_num:02d} [{st:.2f}s-{et:.2f}s] {cls_name:22s}  {desc[:60]}")
        total_cost = sum(r.get("estimated_cost_usd", 0) for r in classifications)
        print(f"\n  Total VLM cost: ${total_cost:.4f}")
    print(f"\n  Outputs in:  {output_dir}")


if __name__ == "__main__":
    main()
