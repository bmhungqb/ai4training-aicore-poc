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
from pathlib import Path

from src.analysis import expert_analysis, macro_eval, micro_eval
from src.analysis.segment_classify import (
    SegmentClassifier, BatchedSegmentClassifier, cut_worker_segments, dump_timeline_debug)
from src.config.common import DATA_DIR
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
def _get_video_duration(video_path: Path) -> float:
    """Returns video duration in seconds using cv2 (fast probe)."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0.0
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        return float(n_frames / fps) if fps > 0 else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
def run_segment(step: str | None = None, force: bool = False, visualize: bool = False,
                mask_path: str | None = None, video_path: str | Path | None = None,
                out_dir: str | Path | None = None, cong_doan: str | int | None = None,
                all_data: bool = False, resize_scale: float | None = None,
                frame_step: int | None = None, frame_by_frame: bool = False,
                data_dir: str | Path | None = None, result_dir: str | Path | None = None,
                ignore_large: bool = False, max_duration: float | None = None,
                sort_by_duration: bool = False) -> None:
    """Phase 1: find where the worker's actions start/stop. No VLM, no expert
    knowledge.

    mask_path: ROI mask image restricting SAM3/SEA-RAFT to one worker's area
    (see tools/mask_editor for drawing one). Defaults to auto-detecting
    "<worker video>.mask.png" next to the video if not given; full frame if
    neither exists.

    cong_doan: operation sequence number (e.g. 1 or 'all'). When provided, all .mp4 videos
    under data/{cong_doan}/ are processed sequentially into data/{cong_doan}/kinematic/{stem}/.

    all_data: if True, processes every operation folder containing .mp4 videos in data/.

    ignore_large: if True, skips videos longer than 3 minutes (180s) by default.

    max_duration: maximum video duration in seconds; videos longer than this are skipped.

    sort_by_duration: if True, sorts videos ascending by length so shortest run first.
    """
    base_data_dir = Path(data_dir) if data_dir else cfg1.DATA_DIR
    base_result_dir = Path(result_dir) if result_dir else (Path(out_dir) if out_dir else base_data_dir)

    if ignore_large and max_duration is None:
        max_duration = 180.0  # 3 minutes default

    steps = [step] if step else SEGMENT_STEPS
    for s in steps:
        if s == "kinematic":
            print("Phase 1 / kinematic: action boundary detection")
            if max_duration is not None:
                print(f"[Filter] Max duration filter active: skipping videos > {max_duration:.1f}s ({max_duration/60:.1f}m)")
            print("-" * 70)
            if all_data or (cong_doan is not None and str(cong_doan).strip().lower() == "all"):
                subdirs = [p for p in base_data_dir.iterdir() if p.is_dir() and any(p.glob("*.mp4"))]
                subdirs.sort(key=lambda p: (0, int(p.name)) if p.name.isdigit() else (1, p.name))
                if not subdirs:
                    raise SystemExit(f"No operation folders with .mp4 videos found in {base_data_dir}")
                total_videos = sum(len(list(p.glob("*.mp4"))) for p in subdirs)
                print(f"Scanning {base_data_dir}/: Found {len(subdirs)} operation folder(s) with {total_videos} video(s) total.")

                count = 0
                for idx_cd, cd_dir in enumerate(subdirs, 1):
                    videos = sorted(cd_dir.glob("*.mp4"))
                    if sort_by_duration:
                        videos.sort(key=lambda v: _get_video_duration(v))
                    print(f"\n[{idx_cd}/{len(subdirs)}] === Công đoạn {cd_dir.name} ({len(videos)} video) ===")
                    for idx_v, v in enumerate(videos, 1):
                        count += 1
                        dur_s = _get_video_duration(v)
                        if max_duration is not None and dur_s > max_duration:
                            print(f"  ({idx_v}/{len(videos)}) [Video {count}/{total_videos}] [SKIP - LARGE] {v.name} ({dur_s:.1f}s > {max_duration:.1f}s)")
                            continue
                        v_out = base_result_dir / cd_dir.name / "kinematic" / v.stem
                        print(f"  ({idx_v}/{len(videos)}) [Video {count}/{total_videos}] {v.name} ({dur_s:.1f}s) -> {v_out}")
                        v_mask = Path(mask_path) if mask_path else (v.with_suffix(".mask.png") if v.with_suffix(".mask.png").exists() else None)
                        segmenter = KinematicSegmenter(video_path=v, out_dir=v_out, mask_path=v_mask)
                        report = segmenter.run(force=force, visualize=visualize, resize_scale=resize_scale,
                                               frame_step=frame_step, frame_by_frame=frame_by_frame)
                        print(f"    -> Ready: {report.n_segments} segments -> {v_out / 'action_segments.json'}")
            elif cong_doan is not None:
                cd_dir = base_data_dir / str(cong_doan)
                if not cd_dir.is_dir():
                    raise SystemExit(f"Operation directory not found: {cd_dir}")
                videos = sorted(cd_dir.glob("*.mp4"))
                if not videos:
                    raise SystemExit(f"No .mp4 videos found in {cd_dir}")
                if sort_by_duration:
                    videos.sort(key=lambda v: _get_video_duration(v))
                print(f"Found {len(videos)} video(s) for công đoạn {cong_doan} in {cd_dir}:")
                for i, v in enumerate(videos, 1):
                    dur_s = _get_video_duration(v)
                    print(f"  {i}. {v.name} ({dur_s:.1f}s)")

                for i, v in enumerate(videos, 1):
                    dur_s = _get_video_duration(v)
                    if max_duration is not None and dur_s > max_duration:
                        print(f"\n[{i}/{len(videos)}] [SKIP - LARGE] {v.name} ({dur_s:.1f}s > {max_duration:.1f}s)")
                        continue
                    v_out = base_result_dir / str(cong_doan) / "kinematic" / v.stem
                    print(f"\n[{i}/{len(videos)}] Processing: {v.name} ({dur_s:.1f}s) -> {v_out}")
                    # auto-detect mask if not explicitly passed
                    v_mask = Path(mask_path) if mask_path else (v.with_suffix(".mask.png") if v.with_suffix(".mask.png").exists() else None)
                    segmenter = KinematicSegmenter(video_path=v, out_dir=v_out, mask_path=v_mask)
                    report = segmenter.run(force=force, visualize=visualize, resize_scale=resize_scale,
                                           frame_step=frame_step, frame_by_frame=frame_by_frame)
                    print(f"Action segmentation ready: {report.n_segments} segments -> {v_out / 'action_segments.json'}")
            else:
                target_video = Path(video_path) if video_path else cfg1.WORKER_VIDEO
                target_out = Path(out_dir) if out_dir else (base_result_dir / "kinematic" if result_dir else cfg1.KINEMATIC_OUT_DIR)
                segmenter = KinematicSegmenter(video_path=target_video, out_dir=target_out, mask_path=mask_path)
                report = segmenter.run(force=force, visualize=visualize, resize_scale=resize_scale,
                                       frame_step=frame_step, frame_by_frame=frame_by_frame)
                print(f"Action segmentation ready: {report.n_segments} segments -> "
                      f"{target_out / 'action_segments.json'}")
        else:
            raise SystemExit(f"Unknown segment step: {s!r} (choices: {SEGMENT_STEPS})")


# ---------------------------------------------------------------------------
## Phase 2: VLM-based analysis (expert -> classify -> macro -> micro)
# ---------------------------------------------------------------------------
def _resolve_analyze_paths(cong_doan: str | int | None = None,
                           video_path: str | Path | None = None,
                           action_segments_path: str | Path | None = None,
                           out_dir: str | Path | None = None) -> dict:
    cd_dir = DATA_DIR / str(cong_doan) if cong_doan else cfg2e.DEFAULT_CD_DIR
    expert_json = cd_dir / "expert.json"
    expert_scenes_dir = cd_dir / "expert_scenes"
    manifest_path = expert_scenes_dir / "selected_frames.json"
    worker_frames_dir = cd_dir / "worker_frames"
    worker_segments_dir = Path(out_dir) if out_dir else (cd_dir / "worker_segments")
    macro_eval_path = cd_dir / "macro_eval.json"
    micro_eval_path = cd_dir / "micro_eval.json"

    # Worker video resolution
    if video_path:
        worker_video = Path(video_path)
    elif (cd_dir / "worker.mp4").exists():
        worker_video = cd_dir / "worker.mp4"
    else:
        vids = sorted(cd_dir.glob("*.mp4"))
        expert_video_name = ""
        if (cd_dir / "chuyen1_segment.json").is_file():
            try:
                ch1 = json.loads((cd_dir / "chuyen1_segment.json").read_text(encoding="utf-8"))
                expert_video_name = ch1.get("video_file", "")
            except Exception:
                pass
        worker_vids = [v for v in vids if v.name != expert_video_name and not v.name.startswith("expert")]
        worker_video = worker_vids[0] if worker_vids else cfg2c.WORKER_VIDEO

    # Action segments resolution
    if action_segments_path and Path(action_segments_path).is_file():
        act_path = Path(action_segments_path)
    else:
        worker_stem = worker_video.resolve().stem
        cand = cd_dir / "kinematic" / worker_stem / "action_segments.json"
        if cand.is_file():
            act_path = cand
        else:
            cands = list((cd_dir / "kinematic").glob("*/action_segments.json")) if (cd_dir / "kinematic").is_dir() else []
            worker_cands = [c for c in cands if not any(x in c.parent.name for x in ("073527", "expert"))]
            act_path = worker_cands[0] if worker_cands else (cands[0] if cands else cfg1.ACTION_SEGMENTS_PATH)

    return {
        "cd_dir": cd_dir,
        "expert_json": expert_json,
        "expert_scenes_dir": expert_scenes_dir,
        "manifest_path": manifest_path,
        "worker_video": worker_video,
        "worker_frames_dir": worker_frames_dir,
        "worker_segments_dir": worker_segments_dir,
        "action_segments_path": act_path,
        "macro_eval_path": macro_eval_path,
        "micro_eval_path": micro_eval_path,
    }


def run_expert(vlm_model: str = cfg2e.VLM_MODEL, force_kinematic: bool = False,
               mask_path: str | Path | None = None,
               expert_json: str | Path | None = None,
               expert_scenes_dir: str | Path | None = None,
               cd_dir: str | Path | None = None) -> None:
    print("Phase 2 / expert: expert analysis")
    print("-" * 70)
    target_cd_dir = Path(cd_dir) if cd_dir else cfg2e.DEFAULT_CD_DIR
    target_json = Path(expert_json) if expert_json else (target_cd_dir / "expert.json")
    target_scenes = Path(expert_scenes_dir) if expert_scenes_dir else (target_cd_dir / "expert_scenes")
    print(f"Working directory: {target_cd_dir}")
    expert_analysis.run(target_json, target_scenes, vlm_model=vlm_model,
                        force_kinematic=force_kinematic, mask_path=mask_path)


def run_classify(model: str = cfg2c.MODEL, cut: bool = False,
                 save_crop_frames: bool = False, visualize: bool = False,
                 action_segments_path: str | Path | None = None,
                 mask_path: str | Path | None = None,
                 expert_mask_path: str | Path | None = None,
                 video_path: str | Path | None = None,
                 manifest_path: str | Path | None = None,
                 out_dir: str | Path | None = None,
                 frames_dir: str | Path | None = None,
                 cd_dir: str | Path | None = None,
                 batched: bool = False,
                 max_workers: int = 15,
                 max_window_duration: float = 4.0,
                 use_motion_composite: bool = True,
                 max_frames_per_call: int = 4) -> tuple[dict, dict]:
    print("Phase 2 / classify: worker segment classification")
    print("-" * 70)
    target_cd_dir = Path(cd_dir) if cd_dir else cfg2c.DEFAULT_CD_DIR
    target_manifest = Path(manifest_path) if manifest_path else (target_cd_dir / "expert_scenes" / "selected_frames.json")
    target_video = Path(video_path) if video_path else (target_cd_dir / "worker.mp4")
    target_out = Path(out_dir) if out_dir else (target_cd_dir / "worker_segments")
    target_frames = Path(frames_dir) if frames_dir else (target_cd_dir / "worker_frames")

    if batched:
        print("[classify] Using BATCHED mode (macro-windows + motion composite + concurrent calls)")
        runner = BatchedSegmentClassifier(
            manifest_path=target_manifest, video_path=target_video,
            out_dir=target_out, frames_dir=target_frames, model=model,
            save_crop_frames=save_crop_frames,
            action_segments_path=action_segments_path,
            mask_path=mask_path, expert_mask_path=expert_mask_path,
            max_workers=max_workers,
            max_window_duration=max_window_duration,
            use_motion_composite=use_motion_composite,
            max_frames_per_call=max_frames_per_call,
        )
    else:
        runner = SegmentClassifier(manifest_path=target_manifest, video_path=target_video,
                                 out_dir=target_out, frames_dir=target_frames, model=model,
                                 save_crop_frames=save_crop_frames,
                                 action_segments_path=action_segments_path,
                                 mask_path=mask_path, expert_mask_path=expert_mask_path)
    print(f"task: {runner.task_name} | {len(runner.ordered_scenes)} scene(s) | "
          f"worker duration: {runner.video_duration:.1f}s")
    # Check availability: BatchedSegmentClassifier uses batch_client, SegmentClassifier uses client
    client_attr = getattr(runner, "batch_client", None) or getattr(runner, "client", None)
    if client_attr and not client_attr.available:
        raise SystemExit("Missing OPENROUTER_API_KEY (env or .env)")
    result = runner.run()
    if visualize:
        dump_timeline_debug(result, target_out / "timeline_debug.json")
        try:
            from tools.generate_timeline_html import generate_html_report
            from tools.render_classified_viz import render_stage2_video
            cd_name = str(target_cd_dir.name) if target_cd_dir else "1"
            generate_html_report(cd_name)
            render_stage2_video(cd_name)
        except Exception as e:
            print(f"[visualize] Note: {e}")
    if cut:
        cut_worker_segments(target_out / "worker_segments.json", video_path=target_video, out_dir=target_out / "cuts")
    manifest = dict(runner.ordered_scenes)
    manifest = {"task_name": runner.task_name, "scenes": manifest}
    return result, manifest


def run_macro(result: dict | None = None, manifest: dict | None = None,
              cd_dir: str | Path | None = None,
              segments_path: str | Path | None = None,
              manifest_path: str | Path | None = None,
              macro_eval_path: str | Path | None = None) -> macro_eval.MacroSummary:
    print("\nPhase 2 / macro: macro evaluation")
    print("-" * 70)
    target_cd_dir = Path(cd_dir) if cd_dir else cfg2m.DEFAULT_CD_DIR
    target_segments = Path(segments_path) if segments_path else (target_cd_dir / "worker_segments" / "worker_segments.json")
    target_manifest = Path(manifest_path) if manifest_path else (target_cd_dir / "expert_scenes" / "selected_frames.json")
    target_macro_out = Path(macro_eval_path) if macro_eval_path else (target_cd_dir / "macro_eval.json")

    if result is None:
        result = json.loads(target_segments.read_text(encoding="utf-8"))
        manifest = json.loads(target_manifest.read_text(encoding="utf-8"))

    summary = macro_eval.summarize(result, manifest)
    macro_eval.print_report(summary)
    macro_eval.save(summary, target_macro_out)
    return summary


def run_micro(summary: macro_eval.MacroSummary | None = None, manifest: dict | None = None,
              model: str = cfg2u.MODEL,
              expert_mask_path: str | Path | None = None,
              worker_mask_path: str | Path | None = None,
              cd_dir: str | Path | None = None,
              macro_eval_path: str | Path | None = None,
              manifest_path: str | Path | None = None,
              worker_frames_dir: str | Path | None = None,
              micro_eval_path: str | Path | None = None) -> None:
    print("\nPhase 2 / micro: micro evaluation")
    print("-" * 70)
    target_cd_dir = Path(cd_dir) if cd_dir else cfg2u.DEFAULT_CD_DIR
    target_macro = Path(macro_eval_path) if macro_eval_path else (target_cd_dir / "macro_eval.json")
    target_manifest = Path(manifest_path) if manifest_path else (target_cd_dir / "expert_scenes" / "selected_frames.json")
    target_frames = Path(worker_frames_dir) if worker_frames_dir else (target_cd_dir / "worker_frames")
    target_micro_out = Path(micro_eval_path) if micro_eval_path else (target_cd_dir / "micro_eval.json")

    if summary is None:
        macro_data = json.loads(target_macro.read_text(encoding="utf-8"))
        summary = macro_eval.MacroSummary(
            task_name=macro_data["task_name"], n_scenes=macro_data["n_scenes"],
            evaluated=macro_data["evaluated"], missing=macro_data["missing"],
            extra=macro_data["extra"],
            slow_segments=[s for s in macro_data["evaluated"] if s["timing_verdict"] == "slow"])
        manifest = json.loads(target_manifest.read_text(encoding="utf-8"))

    vlm_client = OpenRouterClient(model=model)
    if not vlm_client.available:
        raise SystemExit("Missing OPENROUTER_API_KEY (env or .env)")
    scenes_by_name = _scenes_by_name(manifest)
    micro_results = micro_eval.evaluate_slow_segments(
        summary.slow_segments, scenes_by_name, vlm_client, worker_frames_dir=target_frames,
        expert_mask_path=expert_mask_path, worker_mask_path=worker_mask_path)
    micro_eval.save(micro_results, target_micro_out)


def run_analyze(step: str | None = None, vlm_model: str = cfg2e.VLM_MODEL, model: str = cfg2c.MODEL,
                cut: bool = False, save_crop_frames: bool = False, visualize: bool = False,
                action_segments_path=None,
                force_kinematic: bool = False,
                mask_path: str | Path | None = None,
                expert_mask_path: str | Path | None = None,
                cong_doan: str | int | None = None,
                video_path: str | Path | None = None,
                out_dir: str | Path | None = None,
                batched: bool = False,
                max_workers: int = 15,
                max_window_duration: float = 4.0,
                use_motion_composite: bool = True,
                max_frames_per_call: int = 4) -> None:
    """Phase 2: everything VLM-based — learn the standard from the expert
    video, classify Phase 1's action segments against it, then macro/micro
    evaluate. Default runs all 4 sub-steps in order, keeping state in memory."""
    paths = _resolve_analyze_paths(cong_doan=cong_doan, video_path=video_path,
                                   action_segments_path=action_segments_path, out_dir=out_dir)
    cd_dir = paths["cd_dir"]
    act_path = paths["action_segments_path"]
    worker_video = paths["worker_video"]

    if step:
        if step == "expert":
            run_expert(vlm_model=vlm_model, force_kinematic=force_kinematic,
                       mask_path=expert_mask_path or mask_path,
                       expert_json=paths["expert_json"],
                       expert_scenes_dir=paths["expert_scenes_dir"],
                       cd_dir=cd_dir)
        elif step == "classify":
            run_classify(model=model, cut=cut, save_crop_frames=save_crop_frames, visualize=visualize,
                         action_segments_path=act_path,
                         mask_path=mask_path, expert_mask_path=expert_mask_path,
                         video_path=worker_video, manifest_path=paths["manifest_path"],
                         out_dir=paths["worker_segments_dir"], frames_dir=paths["worker_frames_dir"],
                         cd_dir=cd_dir,
                         batched=batched, max_workers=max_workers,
                         max_window_duration=max_window_duration,
                         use_motion_composite=use_motion_composite,
                         max_frames_per_call=max_frames_per_call)
        elif step == "macro":
            run_macro(cd_dir=cd_dir, segments_path=paths["worker_segments_dir"] / "worker_segments.json",
                      manifest_path=paths["manifest_path"], macro_eval_path=paths["macro_eval_path"])
        elif step == "micro":
            run_micro(model=model, expert_mask_path=expert_mask_path, worker_mask_path=mask_path,
                      cd_dir=cd_dir, macro_eval_path=paths["macro_eval_path"],
                      manifest_path=paths["manifest_path"], worker_frames_dir=paths["worker_frames_dir"],
                      micro_eval_path=paths["micro_eval_path"])
        else:
            raise SystemExit(f"Unknown analyze step: {step!r} (choices: {ANALYZE_STEPS})")
        return

    run_expert(vlm_model=vlm_model, force_kinematic=force_kinematic,
               mask_path=expert_mask_path or mask_path,
               expert_json=paths["expert_json"],
               expert_scenes_dir=paths["expert_scenes_dir"],
               cd_dir=cd_dir)
    result, manifest = run_classify(model=model, cut=cut, save_crop_frames=save_crop_frames,
                                    visualize=visualize, action_segments_path=act_path,
                                    mask_path=mask_path, expert_mask_path=expert_mask_path,
                                    video_path=worker_video, manifest_path=paths["manifest_path"],
                                    out_dir=paths["worker_segments_dir"], frames_dir=paths["worker_frames_dir"],
                                    cd_dir=cd_dir,
                                    batched=batched, max_workers=max_workers,
                                    max_window_duration=max_window_duration,
                                    use_motion_composite=use_motion_composite,
                                    max_frames_per_call=max_frames_per_call)
    summary = run_macro(result=result, manifest=manifest, cd_dir=cd_dir,
                        segments_path=paths["worker_segments_dir"] / "worker_segments.json",
                        manifest_path=paths["manifest_path"], macro_eval_path=paths["macro_eval_path"])
    run_micro(summary=summary, manifest=manifest, model=model,
              expert_mask_path=expert_mask_path, worker_mask_path=mask_path,
              cd_dir=cd_dir, macro_eval_path=paths["macro_eval_path"],
              manifest_path=paths["manifest_path"], worker_frames_dir=paths["worker_frames_dir"],
              micro_eval_path=paths["micro_eval_path"])


def run_all(vlm_model: str = cfg2e.VLM_MODEL, model: str = cfg2c.MODEL,
            cut: bool = False, save_crop_frames: bool = False, visualize: bool = False,
            force_segment: bool = False, force_kinematic_expert: bool = False,
            mask_path: str | None = None,
            expert_mask_path: str | None = None,
            cong_doan: str | int | None = None,
            video_path: str | Path | None = None,
            out_dir: str | Path | None = None,
            batched: bool = False,
            max_workers: int = 15,
            max_window_duration: float = 4.0,
            use_motion_composite: bool = True,
            max_frames_per_call: int = 4,
            ignore_large: bool = False,
            max_duration: float | None = None,
            sort_by_duration: bool = False) -> None:
    run_segment(force=force_segment, visualize=visualize, mask_path=mask_path,
                cong_doan=cong_doan, video_path=video_path, out_dir=out_dir,
                ignore_large=ignore_large, max_duration=max_duration,
                sort_by_duration=sort_by_duration)
    run_analyze(vlm_model=vlm_model, model=model, cut=cut, save_crop_frames=save_crop_frames,
                visualize=visualize, force_kinematic=force_kinematic_expert,
                mask_path=mask_path, expert_mask_path=expert_mask_path,
                cong_doan=cong_doan, video_path=video_path, out_dir=out_dir,
                batched=batched, max_workers=max_workers,
                max_window_duration=max_window_duration,
                use_motion_composite=use_motion_composite,
                max_frames_per_call=max_frames_per_call)


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
    ap.add_argument("--force-kinematic-expert", action="store_true",
                    help="force re-running Phase 1 kinematic segmentation on expert.mp4")
    ap.add_argument("--action-segments", default=None,
                    help="path to Phase 1's action_segments.json (classify sub-step input; defaults to auto-detected under kinematic/)")
    ap.add_argument("--resize-scale", type=float, default=None,
                    help="scale factor for optical flow inference (default: 0.5; recommended 0.25 on 2.5K video to prevent OOM)")
    ap.add_argument("--frame-step", type=int, default=None,
                    help="process every Nth frame in Phase 1 (default: 1; use 2 to halve memory and 2x speed)")
    ap.add_argument("--frame-by-frame", action="store_true",
                    help="run SAM3 in stateless frame-by-frame mode (saves RAM on long videos)")
    ap.add_argument("--ignore-large", action="store_true",
                    help="skip large videos (> 3 minutes / 180s by default) during Phase 1 segmentation")
    ap.add_argument("--max-duration", type=float, default=None,
                    help="maximum video duration in seconds to process in Phase 1 (longer videos are skipped)")
    ap.add_argument("--sort-by-duration", action="store_true",
                    help="process shorter videos first within each operation folder")
    # Batched classification (classify step)
    ap.add_argument("--batched", action="store_true",
                    help="use BatchedSegmentClassifier with macro-window + motion composite + concurrent VLM calls (classify step)")
    ap.add_argument("--max-workers", type=int, default=15,
                    help="max concurrent VLM workers for batched mode (default: 15)")
    ap.add_argument("--max-window-duration", type=float, default=4.0,
                    help="max duration in seconds for one macro-window (default: 4.0s)")
    ap.add_argument("--no-motion-composite", action="store_true",
                    help="disable motion-enhanced composite frames in batched mode")
    ap.add_argument("--data-dir", default=None,
                    help="root folder containing operation folders (default: data)")
    ap.add_argument("--result-dir", default=None,
                    help="root folder for output results (default: data_result if it exists, else same as data-dir)")
    args = ap.parse_args()

    if args.phase == "segment":
        run_segment(step=args.step, force=args.force_segment, visualize=args.visualize,
                   mask_path=args.mask, video_path=args.video, out_dir=args.out_dir,
                   cong_doan=args.cong_doan, all_data=args.all_data,
                   resize_scale=args.resize_scale, frame_step=args.frame_step,
                   frame_by_frame=args.frame_by_frame,
                   data_dir=args.data_dir, result_dir=args.result_dir,
                   ignore_large=args.ignore_large, max_duration=args.max_duration,
                   sort_by_duration=args.sort_by_duration)
    elif args.phase == "analyze":
        run_analyze(step=args.step, vlm_model=args.vlm_model, model=args.model, cut=args.cut,
                   save_crop_frames=args.save_crop_frames, visualize=args.visualize,
                   action_segments_path=args.action_segments,
                   force_kinematic=args.force_kinematic_expert,
                   mask_path=args.mask, expert_mask_path=args.expert_mask,
                   cong_doan=args.cong_doan, video_path=args.video, out_dir=args.out_dir,
                   batched=args.batched, max_workers=args.max_workers,
                   max_window_duration=args.max_window_duration,
                   use_motion_composite=not args.no_motion_composite,
                   max_frames_per_call=args.max_frames_per_call)
    elif args.phase == "all":
        run_all(vlm_model=args.vlm_model, model=args.model, cut=args.cut,
               save_crop_frames=args.save_crop_frames, visualize=args.visualize,
               force_segment=args.force_segment,
               force_kinematic_expert=args.force_kinematic_expert,
               mask_path=args.mask, expert_mask_path=args.expert_mask,
               cong_doan=args.cong_doan, video_path=args.video, out_dir=args.out_dir,
               batched=args.batched, max_workers=args.max_workers,
               max_window_duration=args.max_window_duration,
               use_motion_composite=not args.no_motion_composite,
               max_frames_per_call=args.max_frames_per_call,
               ignore_large=args.ignore_large, max_duration=args.max_duration,
               sort_by_duration=args.sort_by_duration)


if __name__ == "__main__":
    main()
