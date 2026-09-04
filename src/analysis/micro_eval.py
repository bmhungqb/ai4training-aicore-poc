"""Phase 2 (micro step): micro evaluation
Evaluate each worker segment flagged "slow" in Phase 3, by comparing the worker's video frames
against the expert's frames and guideline for that scene. Ask the VLM to identify which small
step is slow, why, and how to improve it.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.config.phase2_micro import (
    MAX_EXPERT_FRAMES, MAX_WORKER_FRAMES, WINDOW_PAD_SEC, WORKER_FRAMES_DIR)
from src.manifest import format_product_state
from src.prompts.evaluation_prompts import SYSTEM_MICRO_COMPARE, USER_MICRO_COMPARE
from src.utils.frames import encode_expert_frame, encode_worker_frame, find_mask_for_video, pick_evenly_spread
from src.utils.message_content import labeled_frames, render_template_content
from src.utils.video import worker_frame_ts
from src.vlm_client import OpenRouterClient


def _format_expert_guideline(scene_data: dict) -> str:
    """Format the expert guideline for a scene into a readable string."""
    g = scene_data.get("guideline") or {}
    if not g:
        return "(no guideline)"
    lines = [f"- Operation description: {g.get('operation_description', '')}",
             format_product_state(g)]
    for i, step in enumerate(g.get("how_to_steps", []), 1):
        lines.append(f"  {i}. {step}")
    return "\n".join(lines)


def _format_worker_evidence(seg: dict) -> str:
    """Format the worker's evidence/actions into a readable string."""
    lines = [
        f"- Observation: {seg.get('evidence', '')}",
        f"- Action evidence: {seg.get('action_evidence', '')}",
        f"- Product state: {seg.get('product_state_evidence', '')}",
    ]
    if seg.get("off_standard"):
        lines.append(f"- Off-standard technique flagged: {seg.get('off_standard_desc', '')}")
    return "\n".join(lines)


def _worker_frames_for_window(start_time: float, end_time: float,
                             worker_frames_dir: Path | str) -> list[Path]:
    """Find worker frames in [start_time - WINDOW_PAD_SEC, end_time + WINDOW_PAD_SEC], evenly spread."""
    worker_frames_dir = Path(worker_frames_dir)
    all_frames = sorted(worker_frames_dir.glob("frame_*s.jpg"))
    t_lo = max(0.0, start_time - WINDOW_PAD_SEC)
    t_hi = end_time + WINDOW_PAD_SEC

    window_frames = [p for p in all_frames if t_lo <= worker_frame_ts(p) <= t_hi]
    if not window_frames:
        window_frames = [p for p in all_frames if start_time <= worker_frame_ts(p) <= end_time]
    return pick_evenly_spread(window_frames, MAX_WORKER_FRAMES)


def _build_micro_compare_messages(seg: dict, scene_data: dict, worker_frames_dir: Path | str,
                                  expert_mask_path: str | Path | None = None,
                                  worker_mask_path: str | Path | None = None) -> list[dict]:
    """Build the messages to send to the VLM for a single slow segment.
    Args:
        seg: worker segment dict (from Phase 3)
        scene_data: scene dict from the selection manifest
        worker_frames_dir: directory containing the worker video frames (frame_*.jpg)
        expert_mask_path: optional ROI mask for expert frames
        worker_mask_path: optional ROI mask for worker frames
    Returns:
        List of messages (dicts) to send to the VLM.
    """
    worker_dur = seg["end_time"] - seg["start_time"]
    subs = dict(
        operation_name=seg["operation_name"],
        expert_duration=seg["expert_duration_s"],
        worker_duration=round(worker_dur, 2),
        ratio=seg["duration_ratio"],
        expert_guideline_text=_format_expert_guideline(scene_data),
        worker_evidence_text=_format_worker_evidence(seg),
    )
    expert_frame_paths = pick_evenly_spread(scene_data["frames"], MAX_EXPERT_FRAMES)
    expert_frame_content = labeled_frames(
        (f"[Expert frame {i}/{len(expert_frame_paths)}] {Path(fp).name}:", encode_expert_frame(fp, mask_path=expert_mask_path))
        for i, fp in enumerate(expert_frame_paths, 1))

    worker_frame_paths = _worker_frames_for_window(seg["start_time"], seg["end_time"], worker_frames_dir)
    worker_frame_content = labeled_frames(
        (f"[Worker frame {i}/{len(worker_frame_paths)}, t={worker_frame_ts(fp):.1f}s] {Path(fp).name}:",
         encode_worker_frame(fp, mask_path=worker_mask_path))
        for i, fp in enumerate(worker_frame_paths, 1))

    user_content = render_template_content(
        USER_MICRO_COMPARE, subs,
        {"expert_frames_content": expert_frame_content,
         "worker_frames_content": worker_frame_content})
    return [
        {"role": "system", "content": SYSTEM_MICRO_COMPARE},
        {"role": "user", "content": user_content},
    ]


def evaluate_slow_segments(slow_segments: list[dict], scenes_by_name: dict,
                          vlm_client: OpenRouterClient,
                          worker_frames_dir: str | Path = WORKER_FRAMES_DIR,
                          expert_mask_path: str | Path | None = None,
                          worker_mask_path: str | Path | None = None) -> list[dict]:
    """Evaluate each slow segment step of the worker.
    Args:
        slow_segments: list of worker segments flagged "slow" (from Phase 3)
        scenes_by_name: dict mapping operation_name to scene data (from selected_frames.json)
        vlm_client: OpenRouterClient instance for sending messages to the VLM
        worker_frames_dir: directory containing the worker video frames (frame_*.jpg)
        expert_mask_path: optional ROI mask for expert frames
        worker_mask_path: optional ROI mask for worker frames
    Returns:
        List of dicts containing the VLM's micro evaluation for each slow segment.
    """
    results = []
    for seg in slow_segments:
        scene_data = scenes_by_name[seg["operation_name"]]
        messages = _build_micro_compare_messages(
            seg, scene_data, worker_frames_dir,
            expert_mask_path=expert_mask_path, worker_mask_path=worker_mask_path)
        try:
            reply = vlm_client.chat_json(messages)
        except Exception as e:
            print(f"{seg['operation_name']}: ERROR - {e}")
            continue

        worker_dur = round(seg["end_time"] - seg["start_time"], 2)
        reply["expert_duration_s"] = seg["expert_duration_s"]
        reply["worker_duration_s"] = worker_dur
        reply["duration_ratio"] = seg["duration_ratio"]
        results.append(reply)

        print(f"\n{'=' * 70}")
        print(f"{seg['operation_name']} — worker {worker_dur}s vs standard {seg['expert_duration_s']}s "
              f"(x{seg['duration_ratio']})")
        print(f"{'=' * 70}")
        print(f"Slow step       : {reply.get('slow_step')}")
        print(f"Evidence        : {reply.get('comparison_evidence')}")
        print(f"Probable cause  : {reply.get('probable_cause')}")
        print(f"Improvement     : {reply.get('improvement_suggestion')}")
        print(f"Confidence      : {reply.get('confidence_level')}")

    return results


def save(results: list[dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(results)} detailed micro evaluations to {out_path}")
