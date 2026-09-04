#!/usr/bin/env python3
"""CLI entry point for the 2-phase sewing-skill evaluation pipeline.

    python pipeline.py segment                  # Phase 1: worker action segmentation
    python pipeline.py analyze                  # Phase 2: VLM analysis (expert -> classify -> macro -> micro)
    python pipeline.py all                       # both phases

Each phase can be narrowed to one sub-step with --step, and re-runs its
sub-steps' inputs from disk instead of needing the whole phase re-run:

    python pipeline.py segment --step kinematic
    python pipeline.py analyze --step expert
    python pipeline.py analyze --step classify
    python pipeline.py analyze --step macro
    python pipeline.py analyze --step micro
"""
from __future__ import annotations

import argparse
import json

from src.analysis import expert_analysis, macro_eval, micro_eval
from src.analysis.segment_classify import SegmentClassifier, cut_worker_segments, dump_timeline_debug
from src.config import phase1_segmentation as cfg1
from src.config import phase2_classify as cfg2c
from src.config import phase2_expert as cfg2e
from src.config import phase2_macro as cfg2m
from src.config import phase2_micro as cfg2u
from src.manifest import ordered_scene_items, scene_op_name
from src.segmentation.kinematic import KinematicSegmenter
from src.vlm_client import OpenRouterClient

SEGMENT_STEPS = ["kinematic"]
ANALYZE_STEPS = ["expert", "classify", "macro", "micro"]


def _scenes_by_name(manifest: dict) -> dict:
    """Return a dict mapping operation_name -> scene dict from the manifest."""
    result = {}
    for _, sd in ordered_scene_items(manifest):
        result.setdefault(scene_op_name(sd), sd)
    return result


# ---------------------------------------------------------------------------
# Phase 1: worker action segmentation
# ---------------------------------------------------------------------------
def run_segment(step: str | None = None, force: bool = False, visualize: bool = False,
                mask_path: str | None = None, video_path: str | Path | None = None,
                out_dir: str | Path | None = None, cong_doan: str | int | None = None,
                all_data: bool = False, resize_scale: float | None = None,
                frame_step: int | None = None, frame_by_frame: bool = False) -> None:
    """Phase 1: find where the worker's actions start/stop. No VLM, no expert
    knowledge.

    mask_path: ROI mask image restricting SAM3/SEA-RAFT to one worker's area
    (see tools/mask_editor for drawing one). Defaults to auto-detecting
    "<worker video>.mask.png" next to the video if not given; full frame if
    neither exists.

    cong_doan: operation sequence number (e.g. 1 or 'all'). When provided, all .mp4 videos
    under data/{cong_doan}/ are processed sequentially into data/{cong_doan}/kinematic/{stem}/.

    all_data: if True, processes every operation folder containing .mp4 videos in data/.
    """
    steps = [step] if step else SEGMENT_STEPS
    for s in steps:
        if s == "kinematic":
            print("Phase 1 / kinematic: action boundary detection")
            print("-" * 70)
            if all_data or (cong_doan is not None and str(cong_doan).strip().lower() == "all"):
                subdirs = [p for p in cfg1.DATA_DIR.iterdir() if p.is_dir() and any(p.glob("*.mp4"))]
                subdirs.sort(key=lambda p: (0, int(p.name)) if p.name.isdigit() else (1, p.name))
                if not subdirs:
                    raise SystemExit(f"No operation folders with .mp4 videos found in {cfg1.DATA_DIR}")
                total_videos = sum(len(list(p.glob("*.mp4"))) for p in subdirs)
                print(f"Scanning data/: Found {len(subdirs)} operation folder(s) with {total_videos} video(s) total.")

                count = 0
                for idx_cd, cd_dir in enumerate(subdirs, 1):
                    videos = sorted(cd_dir.glob("*.mp4"))
                    print(f"\n[{idx_cd}/{len(subdirs)}] === Công đoạn {cd_dir.name} ({len(videos)} video) ===")
                    for idx_v, v in enumerate(videos, 1):
                        count += 1
                        v_out = cd_dir / "kinematic" / v.stem
                        print(f"  ({idx_v}/{len(videos)}) [Video {count}/{total_videos}] {v.name} -> {v_out}")
                        v_mask = Path(mask_path) if mask_path else (v.with_suffix(".mask.png") if v.with_suffix(".mask.png").exists() else None)
                        segmenter = KinematicSegmenter(video_path=v, out_dir=v_out, mask_path=v_mask)
                        report = segmenter.run(force=force, visualize=visualize, resize_scale=resize_scale,
                                               frame_step=frame_step, frame_by_frame=frame_by_frame)
                        print(f"    -> Ready: {report.n_segments} segments -> {v_out / 'action_segments.json'}")
            elif cong_doan is not None:
                cd_dir = cfg1.DATA_DIR / str(cong_doan)
                if not cd_dir.is_dir():
                    raise SystemExit(f"Operation directory not found: {cd_dir}")
                videos = sorted(cd_dir.glob("*.mp4"))
                if not videos:
                    raise SystemExit(f"No .mp4 videos found in {cd_dir}")
                print(f"Found {len(videos)} video(s) for công đoạn {cong_doan} in {cd_dir}:")
                for i, v in enumerate(videos, 1):
                    print(f"  {i}. {v.name}")

                for i, v in enumerate(videos, 1):
                    v_out = cd_dir / "kinematic" / v.stem
                    print(f"\n[{i}/{len(videos)}] Processing: {v.name} -> {v_out}")
                    # auto-detect mask if not explicitly passed
                    v_mask = Path(mask_path) if mask_path else (v.with_suffix(".mask.png") if v.with_suffix(".mask.png").exists() else None)
                    segmenter = KinematicSegmenter(video_path=v, out_dir=v_out, mask_path=v_mask)
                    report = segmenter.run(force=force, visualize=visualize, resize_scale=resize_scale,
                                           frame_step=frame_step, frame_by_frame=frame_by_frame)
                    print(f"Action segmentation ready: {report.n_segments} segments -> {v_out / 'action_segments.json'}")
            else:
                target_video = Path(video_path) if video_path else cfg1.WORKER_VIDEO
                target_out = Path(out_dir) if out_dir else cfg1.KINEMATIC_OUT_DIR
                segmenter = KinematicSegmenter(video_path=target_video, out_dir=target_out, mask_path=mask_path)
                report = segmenter.run(force=force, visualize=visualize, resize_scale=resize_scale,
                                       frame_step=frame_step, frame_by_frame=frame_by_frame)
                print(f"Action segmentation ready: {report.n_segments} segments -> "
                      f"{target_out / 'action_segments.json'}")
        else:
            raise SystemExit(f"Unknown segment step: {s!r} (choices: {SEGMENT_STEPS})")


# ---------------------------------------------------------------------------
# Phase 2: VLM-based analysis (expert -> classify -> macro -> micro)
# ---------------------------------------------------------------------------
def run_expert(vlm_model: str = cfg2e.VLM_MODEL, force_kinematic: bool = False,
               mask_path: str | Path | None = None) -> None:
    print("Phase 2 / expert: expert analysis")
    print("-" * 70)
    print(f"Reference frames auto-selected from kinematic action segmentation on "
          f"{cfg2e.EXPERT_VIDEO} (no manual curation).")
    expert_analysis.run(cfg2e.EXPERT_JSON, cfg2e.EXPERT_SCENES_DIR, vlm_model=vlm_model,
                        force_kinematic=force_kinematic, mask_path=mask_path)


def run_classify(model: str = cfg2c.MODEL, cut: bool = False,
                 save_crop_frames: bool = False, visualize: bool = False,
                 action_segments_path=cfg1.ACTION_SEGMENTS_PATH,
                 mask_path: str | Path | None = None,
                 expert_mask_path: str | Path | None = None) -> tuple[dict, dict]:
    print("Phase 2 / classify: worker segment classification")
    print("-" * 70)
    runner = SegmentClassifier(manifest_path=cfg2e.MANIFEST_PATH, video_path=cfg2c.WORKER_VIDEO,
                               out_dir=cfg2c.OUT_DIR, frames_dir=cfg2c.WORKER_FRAMES_DIR, model=model,
                               save_crop_frames=save_crop_frames,
                               action_segments_path=action_segments_path,
                               mask_path=mask_path, expert_mask_path=expert_mask_path)
    print(f"task: {runner.task_name} | {len(runner.ordered_scenes)} scene(s) | "
          f"worker duration: {runner.video_duration:.1f}s")
    if not runner.client.available:
        raise SystemExit("Missing OPENROUTER_API_KEY (env or .env)")
    result = runner.run()
    if visualize:
        dump_timeline_debug(result, cfg2c.TIMELINE_DEBUG_PATH)
    if cut:
        cut_worker_segments(cfg2c.SEGMENTS_PATH, video_path=cfg2c.WORKER_VIDEO, out_dir=cfg2c.CUTS_DIR)
    manifest = dict(runner.ordered_scenes)  # not the raw manifest dict, but summarize() only needs "scenes"
    manifest = {"task_name": runner.task_name, "scenes": manifest}
    return result, manifest


def run_macro(result: dict | None = None, manifest: dict | None = None) -> macro_eval.MacroSummary:
    print("\nPhase 2 / macro: macro evaluation")
    print("-" * 70)
    if result is None:
        result = json.loads(cfg2c.SEGMENTS_PATH.read_text(encoding="utf-8"))
        manifest = json.loads(cfg2e.MANIFEST_PATH.read_text(encoding="utf-8"))

    summary = macro_eval.summarize(result, manifest)
    macro_eval.print_report(summary)
    macro_eval.save(summary, cfg2m.MACRO_EVAL_PATH)
    return summary


def run_micro(summary: macro_eval.MacroSummary | None = None, manifest: dict | None = None,
             model: str = cfg2u.MODEL,
             expert_mask_path: str | Path | None = None,
             worker_mask_path: str | Path | None = None) -> None:
    print("\nPhase 2 / micro: micro evaluation")
    print("-" * 70)
    if summary is None:
        macro_data = json.loads(cfg2m.MACRO_EVAL_PATH.read_text(encoding="utf-8"))
        summary = macro_eval.MacroSummary(
            task_name=macro_data["task_name"], n_scenes=macro_data["n_scenes"],
            evaluated=macro_data["evaluated"], missing=macro_data["missing"],
            extra=macro_data["extra"],
            slow_segments=[s for s in macro_data["evaluated"] if s["timing_verdict"] == "slow"])
        manifest = json.loads(cfg2e.MANIFEST_PATH.read_text(encoding="utf-8"))

    vlm_client = OpenRouterClient(model=model)
    if not vlm_client.available:
        raise SystemExit("Missing OPENROUTER_API_KEY (env or .env)")
    scenes_by_name = _scenes_by_name(manifest)
    micro_results = micro_eval.evaluate_slow_segments(
        summary.slow_segments, scenes_by_name, vlm_client, worker_frames_dir=cfg2u.WORKER_FRAMES_DIR,
        expert_mask_path=expert_mask_path, worker_mask_path=worker_mask_path)
    micro_eval.save(micro_results, cfg2u.MICRO_EVAL_PATH)


def run_analyze(step: str | None = None, vlm_model: str = cfg2e.VLM_MODEL, model: str = cfg2c.MODEL,
               cut: bool = False, save_crop_frames: bool = False, visualize: bool = False,
               action_segments_path=cfg1.ACTION_SEGMENTS_PATH,
               force_kinematic: bool = False,
               mask_path: str | Path | None = None,
               expert_mask_path: str | Path | None = None) -> None:
    """Phase 2: everything VLM-based — learn the standard from the expert
    video, classify Phase 1's action segments against it, then macro/micro
    evaluate. Default runs all 4 sub-steps in order, keeping state in memory."""
    if step:
        if step == "expert":
            run_expert(vlm_model=vlm_model, force_kinematic=force_kinematic, mask_path=expert_mask_path or mask_path)
        elif step == "classify":
            run_classify(model=model, cut=cut, save_crop_frames=save_crop_frames, visualize=visualize,
                        action_segments_path=action_segments_path,
                        mask_path=mask_path, expert_mask_path=expert_mask_path)
        elif step == "macro":
            run_macro()
        elif step == "micro":
            run_micro(model=model, expert_mask_path=expert_mask_path, worker_mask_path=mask_path)
        else:
            raise SystemExit(f"Unknown analyze step: {step!r} (choices: {ANALYZE_STEPS})")
        return

    run_expert(vlm_model=vlm_model, force_kinematic=force_kinematic, mask_path=expert_mask_path or mask_path)
    result, manifest = run_classify(model=model, cut=cut, save_crop_frames=save_crop_frames,
                                    visualize=visualize, action_segments_path=action_segments_path,
                                    mask_path=mask_path, expert_mask_path=expert_mask_path)
    summary = run_macro(result=result, manifest=manifest)
    run_micro(summary=summary, manifest=manifest, model=model,
              expert_mask_path=expert_mask_path, worker_mask_path=mask_path)


def run_all(vlm_model: str = cfg2e.VLM_MODEL, model: str = cfg2c.MODEL,
           cut: bool = False, save_crop_frames: bool = False, visualize: bool = False,
           force_segment: bool = False, force_kinematic_expert: bool = False,
           mask_path: str | None = None,
           expert_mask_path: str | None = None) -> None:
    run_segment(force=force_segment, visualize=visualize, mask_path=mask_path)
    run_analyze(vlm_model=vlm_model, model=model, cut=cut, save_crop_frames=save_crop_frames,
               visualize=visualize, force_kinematic=force_kinematic_expert,
               mask_path=mask_path, expert_mask_path=expert_mask_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["segment", "analyze", "all"])
    ap.add_argument("--step", default=None,
                    help=f"run only one sub-step of the phase (segment: {SEGMENT_STEPS}; "
                         f"analyze: {ANALYZE_STEPS}) — reads that step's inputs from disk")
    ap.add_argument("--vlm-model", default=cfg2e.VLM_MODEL,
                    help="model used for the expert-analysis sub-step")
    ap.add_argument("--model", default=cfg2c.MODEL,
                    help="model used for the classify/macro/micro sub-steps")
    ap.add_argument("--cut", action="store_true",
                    help="also cut worker.mp4 into one clip per segment (classify sub-step)")
    ap.add_argument("--save-crop-frames", action="store_true",
                    help="save the cropped+resized worker frames actually sent to the VLM "
                         "(to <frames-dir>_cropped) for inspection (classify sub-step)")
    ap.add_argument("--mask", default=None,
                    help="ROI mask image restricting SAM3/SEA-RAFT to one worker's area (segment "
                         "phase) and masking/cropping worker frames sent to VLM (analyze phase); "
                         "defaults to auto-detecting '<worker video>.mask.png' next to the video")
    ap.add_argument("--expert-mask", default=None,
                    help="ROI mask image for expert reference video (analyze phase); "
                         "defaults to auto-detecting '<expert video>.mask.png' next to the video")
    ap.add_argument("--visualize", action="store_true",
                    help="write extra debug artifacts: annotated boundary video/plots "
                         "(segment phase) and a timeline debug dump (analyze/classify step)")
    ap.add_argument("--cong-doan", "--cd", dest="cong_doan", default=None,
                    help="operation number (e.g. 1) or 'all' to segment all videos under data/{cong_doan}/ (Phase 1)")
    ap.add_argument("--all-data", "--all-cong-doan", "--all-cd", dest="all_data", action="store_true",
                    help="segment all videos across all operation folders under data/ (Phase 1)")
    ap.add_argument("--video", default=None,
                    help="path to a specific video file to segment (Phase 1)")
    ap.add_argument("--out-dir", default=None,
                    help="output directory for segmentation results (Phase 1)")
    ap.add_argument("--force-segment", action="store_true",
                    help="force re-running Phase 1 segmentation even if an existing report exists")
    ap.add_argument("--action-segments", default=cfg1.ACTION_SEGMENTS_PATH,
                    help="path to Phase 1's action_segments.json (classify sub-step input)")
    ap.add_argument("--resize-scale", type=float, default=None,
                    help="scale factor for optical flow inference (default: 0.5; recommended 0.25 on 2.5K video to prevent OOM)")
    ap.add_argument("--frame-step", type=int, default=None,
                    help="process every Nth frame in Phase 1 (default: 1; use 2 to halve memory and 2x speed)")
    ap.add_argument("--frame-by-frame", action="store_true",
                    help="run SAM3 in stateless frame-by-frame mode (saves RAM on long videos)")
    args = ap.parse_args()

    if args.phase == "segment":
        run_segment(step=args.step, force=args.force_segment, visualize=args.visualize,
                   mask_path=args.mask, video_path=args.video, out_dir=args.out_dir,
                   cong_doan=args.cong_doan, all_data=args.all_data,
                   resize_scale=args.resize_scale, frame_step=args.frame_step,
                   frame_by_frame=args.frame_by_frame)
    elif args.phase == "analyze":
        run_analyze(step=args.step, vlm_model=args.vlm_model, model=args.model, cut=args.cut,
                   save_crop_frames=args.save_crop_frames, visualize=args.visualize,
                   action_segments_path=args.action_segments,
                   force_kinematic=args.force_kinematic_expert,
                   mask_path=args.mask, expert_mask_path=args.expert_mask)
    elif args.phase == "all":
        run_all(vlm_model=args.vlm_model, model=args.model, cut=args.cut,
               save_crop_frames=args.save_crop_frames, visualize=args.visualize,
               force_segment=args.force_segment,
               force_kinematic_expert=args.force_kinematic_expert,
               mask_path=args.mask, expert_mask_path=args.expert_mask)


if __name__ == "__main__":
    main()
