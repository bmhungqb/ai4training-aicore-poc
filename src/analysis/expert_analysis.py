"""Phase 2 / expert: analyze the expert video to build the "process knowledge"
that the classify sub-step uses as its reference standard.

Scene boundaries (`timestamp_start`/`timestamp_end`/`operations` per scene in
expert.json) are ground truth and stay hand-authored — that's the human
definition of the process being taught, not something to infer from the video.

Reference frame SELECTION within each scene is automatic:
auto_select_frames_from_kinematic — run kinematic action segmentation on
expert.mp4 (same engine as Phase 1 for worker.mp4, no VLM), assign each
action segment to the scene whose time range contains it, then sample
FRAMES_PER_ACTION_SEGMENT frames per action segment (sharpest frame in a
neighborhood around each uniform time step — see pick_sharpest_spread() in
src/utils/frames.py) and write scene["selected_frame_indices"]. No manual
frame curation.

From there the pipeline does:
1. build_selection_manifest — extract each scene's selected reference frames
   (by absolute frame index, from expert.json's
   scene["selected_frame_indices"]) and build selected_frames.json
2. generate_scene_guidelines — ask the VLM to describe how each scene is
   performed + the product state before/during/after
3. synthesize_process_knowledge — merge all scene guidelines into one
   process-wide reference (per operation: observable traits, start/end
   cues, easily-confused neighbors)
"""
from __future__ import annotations

import json
from pathlib import Path

from src.config.phase1_segmentation import (
    ANGLE_TOLERANCE, MIN_DISTANCE, MIN_SEGMENT_LEN, PROMINENCE, REQUIRE_BOTH, SMOOTH_WINDOW)
from src.config.phase2_expert import (
    EXPERT_ACTION_SEGMENTS_PATH, EXPERT_JSON, EXPERT_KINEMATIC_OUT_DIR, EXPERT_SCENES_DIR,
    FRAMES_PER_ACTION_SEGMENT, MAX_REF_FRAMES_PER_SCENE, SHARPNESS_POOL_FACTOR, VLM_MODEL)
from src.manifest import format_product_state, ordered_scene_items, scene_op_name
from src.prompts.expert_analysis_prompts import (
    SYSTEM_LEARNING_PHASE, SYSTEM_SYNTHESIS_PHASE, USER_LEARNING_PHASE, USER_SYNTHESIS_PHASE)
from src.segmentation.kinematic import KinematicSegmenter
from src.utils.frames import (
    encode_expert_frame, find_mask_for_video, pick_evenly_spread, pick_sharpest_spread)
from src.utils.message_content import labeled_frames, render_template_content
from src.utils.video import extract_frames_by_index, sample_window_frames_cached
from src.vlm_client import OpenRouterClient


def auto_select_frames_from_kinematic(
        expert_json_path: str | Path = EXPERT_JSON,
        frames_dir: str | Path = None,
        kinematic_out_dir: str | Path = EXPERT_KINEMATIC_OUT_DIR,
        action_segments_path: str | Path = EXPERT_ACTION_SEGMENTS_PATH,
        frames_per_segment: int = FRAMES_PER_ACTION_SEGMENT,
        sharpness_pool_factor: int = SHARPNESS_POOL_FACTOR,
        force_kinematic: bool = False) -> dict:
    """Automatically pick reference frames for every scene in expert.json,
    replacing the old dense-sample + manual-delete workflow.

    1. Run (or reuse) kinematic action segmentation on expert.mp4 — the same
       no-VLM engine Phase 1 uses for worker.mp4 — producing plain action
       segments (start/end time, no labels).
    2. Assign each action segment to the scene whose [timestamp_start,
       timestamp_end] contains the segment's midpoint. A scene may own
       multiple action segments (it's one operation performed across several
       physical motions).
    3. For each scene, sample `frames_per_segment` frames from EVERY action
       segment assigned to it (uniformly spaced in time within the segment,
       picking the sharpest nearby frame via pick_sharpest_spread() to avoid
       motion-blurred frames), then write the resulting absolute frame
       indices into expert.json as scene["selected_frame_indices"].

    Falls back to a single evenly-spread uniform sample over the scene's
    whole time range (frames_per_segment frames) for any scene that ends up
    with zero action segments assigned (e.g. a very short or static scene the
    kinematic engine didn't split).
    """
    expert_json_path = Path(expert_json_path)
    expert_data = json.loads(expert_json_path.read_text(encoding="utf-8"))
    expert_video = expert_data["expert_video"][0]
    video_path = Path(expert_video["file_path"])
    frames_dir = Path(frames_dir) if frames_dir else Path(EXPERT_SCENES_DIR) / "frames"

    segmenter = KinematicSegmenter(
        video_path=video_path,
        out_dir=Path(kinematic_out_dir),
        report_path=Path(kinematic_out_dir) / "pipe1_report.json",
        boundaries_path=Path(kinematic_out_dir) / "action_boundaries_dynamic.npy")
    report = segmenter.run(
        force=force_kinematic, angle_tolerance=ANGLE_TOLERANCE, min_segment_len=MIN_SEGMENT_LEN,
        min_distance=MIN_DISTANCE, prominence=PROMINENCE, smooth_window=SMOOTH_WINDOW,
        require_both=REQUIRE_BOTH, action_segments_path=Path(action_segments_path))
    fps = report.fps or 25.0

    scenes = expert_video["scenes"]
    for scene in scenes:
        idx = scene["scene_index"]
        t0, t1 = scene["timestamp_start"], scene["timestamp_end"]
        owned = [s for s in report.segments if t0 <= (s.start_time_s + s.end_time_s) / 2 < t1]

        scene_dir = frames_dir / f"scene_{idx:02d}"
        indices: set[int] = set()

        if owned:
            for seg in owned:
                seg_frames = sample_window_frames_cached(
                    video_path, seg.start_time_s, seg.end_time_s,
                    fps=max(2.0, frames_per_segment * sharpness_pool_factor / max(seg.duration_s, 0.1)),
                    out_dir=scene_dir)
                paths = [p for _, p in seg_frames]
                for p in pick_sharpest_spread(paths, frames_per_segment, sharpness_pool_factor):
                    frame_idx = round(float(Path(p).stem.removeprefix("frame_").removesuffix("s")) * fps)
                    indices.add(frame_idx)
            print(f"scene {idx:02d}: {len(owned)} action segment(s) -> {len(indices)} frame(s)")
        else:
            # fallback: no kinematic segment landed in this scene's time range
            candidate_fps = max(2.0, frames_per_segment * sharpness_pool_factor / max(t1 - t0, 0.1))
            seg_frames = sample_window_frames_cached(video_path, t0, t1, fps=candidate_fps, out_dir=scene_dir)
            paths = [p for _, p in seg_frames]
            for p in pick_sharpest_spread(paths, frames_per_segment, sharpness_pool_factor):
                frame_idx = round(float(Path(p).stem.removeprefix("frame_").removesuffix("s")) * fps)
                indices.add(frame_idx)
            print(f"scene {idx:02d}: 0 action segments in range -> fallback uniform sample, "
                  f"{len(indices)} frame(s)")

        scene["selected_frame_indices"] = sorted(indices)

    expert_json_path.write_text(json.dumps(expert_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote selected_frame_indices for {len(scenes)} scene(s) into {expert_json_path} "
          f"(auto-selected from {report.n_segments} kinematic action segment(s))")
    return expert_data


def build_selection_manifest(expert_json_path: str | Path, frames_dir: str | Path,
                             out_path: str | Path, mask_path: str | Path | None = None) -> dict:
    """Extract each scene's curated reference frames.
    Args:
        expert_json_path: Path to the expert.json file.
        frames_dir: Directory containing the curated reference frames for each scene.
        out_path: Path to write the selection manifest (selected_frames.json).
        mask_path: Optional ROI mask path for expert video.
    Returns:
        The selection manifest as a dictionary.
    """
    expert_json_path = Path(expert_json_path)
    frames_dir = Path(frames_dir)
    out_path = Path(out_path)

    expert_data = json.loads(expert_json_path.read_text(encoding="utf-8"))
    task_name = expert_data["task_name"]
    expert_video = expert_data["expert_video"][0]
    video_path = Path(expert_video["file_path"])
    operations_catalog = {op["name"]: op for op in expert_data["operations"]}

    # Auto-detect mask for expert video if not passed explicitly
    detected_mask = find_mask_for_video(video_path, explicit_mask=mask_path)
    if detected_mask:
        print(f"[Phase 2 / expert] Using ROI mask for expert video: {detected_mask}")

    selected_frames = {}
    for scene in expert_video["scenes"]:
        idx = scene["scene_index"]
        indices = scene.get("selected_frame_indices")
        if not indices:
            raise ValueError(
                f"scene {idx} has no selected_frame_indices — annotate expert.json for this "
                "scene before running Phase 1 (see README)")

        scene_frames_dir = frames_dir / f"scene_{idx:02d}"
        frame_paths = extract_frames_by_index(video_path, indices, scene_frames_dir)

        op_names = scene["operations"]
        operations = [operations_catalog.get(name, {"name": name}) for name in op_names]
        selected_frames[idx] = {
            "operations": operations,
            "timestamp_start": scene["timestamp_start"],
            "timestamp_end": scene["timestamp_end"],
            "frames": [str(p) for p in frame_paths],
        }
        print(f"scene {idx:02d}: extracted {len(frame_paths)} frame(s) at indices {indices} "
              f"-> {scene_frames_dir}/, ops={op_names}")

    manifest = {
        "task_name": task_name,
        "video_path": str(video_path),
        "mask_path": str(detected_mask) if detected_mask else None,
        "scenes": selected_frames
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nTask: {task_name}")
    print(f"Selection manifest written to {out_path}")
    return manifest


def _build_learning_messages(task_name: str, scene_data: dict, mask_path: str | Path | None = None) -> list[dict]:
    operation_name = "+ ".join(op["name"] for op in scene_data["operations"])
    frame_paths = scene_data["frames"]

    frame_content = labeled_frames(
        (f"Frame {order}/{len(frame_paths)} ({Path(fp).name}):", encode_expert_frame(fp, mask_path=mask_path))
        for order, fp in enumerate(frame_paths, start=1))

    # NOTE: the template's $operation_name is the whole task and $task_name the
    # scene's operations — historical naming kept as-is in the prompt file.
    user_content = render_template_content(
        USER_LEARNING_PHASE, dict(operation_name=task_name, task_name=operation_name),
        {"task_video_content": frame_content})
    return [
        {"role": "system", "content": SYSTEM_LEARNING_PHASE},
        {"role": "user", "content": user_content},
    ]


def generate_scene_guidelines(manifest_path: str | Path, vlm_client: OpenRouterClient,
                              mask_path: str | Path | None = None) -> dict:
    """Use the VLM for generating a guideline (how the
    operation is performed + product state before/during/after) and save it
    back into the manifest.
    Args:
        manifest_path: Path to the selection manifest (selected_frames.json).
        vlm_client: An instance of OpenRouterClient for interacting with the VLM.
        mask_path: Optional ROI mask path for expert frames.
    Returns:
        The updated manifest with scene guidelines.
    """
    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_name = data["task_name"]
    mask = mask_path or data.get("mask_path")

    for sid, scene_data in data["scenes"].items():
        messages = _build_learning_messages(task_name, scene_data, mask_path=mask)
        try:
            reply = vlm_client.chat_json(messages)
        except Exception as e:
            print(f"scene {sid}: ERROR - {e}")
            continue
        scene_data["guideline"] = reply
        print(f"scene {sid}: {reply.get('operation_name')}")

    manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nUpdated guideline for {len(data['scenes'])} scenes in {manifest_path}")
    return data


def _format_scene_summary(sid: str, scene_data: dict) -> str:
    guideline = scene_data.get("guideline", {})
    lines = [f"### Scene {sid} — operations: {scene_op_name(scene_data)} "
             f"(t={scene_data.get('timestamp_start')}s-{scene_data.get('timestamp_end')}s)"]
    if guideline:
        lines.append(f"- Operation description: {guideline.get('operation_description', '')}")
        lines.append(format_product_state(guideline))
        for i, step in enumerate(guideline.get("how_to_steps", []), 1):
            lines.append(f"  {i}. {step}")
    else:
        lines.append("- (no guideline yet)")
    return "\n".join(lines)


def synthesize_process_knowledge(manifest_path: str | Path, vlm_client: OpenRouterClient,
                                 out_path: str | Path,
                                 max_ref_frames_per_scene: int = MAX_REF_FRAMES_PER_SCENE,
                                 mask_path: str | Path | None = None) -> dict:
    """Merge every scene's guideline into one process-wide reference: per
    operation (deduplicated across repeats), observable traits, start/end
    cues, and easily-confused neighbors.
    Args:
        manifest_path: Path to the selection manifest (selected_frames.json).
        vlm_client: An instance of OpenRouterClient for interacting with the VLM.
        out_path: Path to write the synthesized process knowledge (process_knowledge.json).
        max_ref_frames_per_scene: Maximum number of reference frames to include per scene.
        mask_path: Optional ROI mask path for expert frames.
    Returns:
        The synthesized process knowledge as a dictionary.
    """
    manifest_path = Path(manifest_path)
    out_path = Path(out_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_name = data["task_name"]
    mask = mask_path or data.get("mask_path")

    ordered_scenes = ordered_scene_items(data)
    scenes_summary = "\n\n".join(_format_scene_summary(sid, sd) for sid, sd in ordered_scenes)

    ref_frame_content: list[dict] = []
    for sid, scene_data in ordered_scenes:
        ref_paths = pick_evenly_spread(scene_data["frames"], max_ref_frames_per_scene)
        ref_frame_content += labeled_frames(
            (f"[Scene {sid} - {scene_op_name(scene_data)}] {Path(fp).name}:", encode_expert_frame(fp, mask_path=mask))
            for fp in ref_paths)

    user_content = render_template_content(
        USER_SYNTHESIS_PHASE, dict(task_name=task_name, scenes_summary=scenes_summary),
        {"reference_frames_content": ref_frame_content})
    synthesis_messages = [
        {"role": "system", "content": SYSTEM_SYNTHESIS_PHASE},
        {"role": "user", "content": user_content},
    ]

    print(f"Total reference frames sent: {len(ref_frame_content) // 2} "
          f"(over {len(ordered_scenes)} scenes, max {max_ref_frames_per_scene}/scene)")

    process_knowledge = vlm_client.chat_json(synthesis_messages)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(process_knowledge, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved process knowledge to {out_path}")
    return process_knowledge


def run(expert_json: str | Path = EXPERT_JSON,
       out_dir: str | Path = EXPERT_SCENES_DIR,
       vlm_model: str = VLM_MODEL,
       force_kinematic: bool = False,
       mask_path: str | Path | None = None) -> Path:
    """Run the full expert-analysis sub-step: auto-select reference frames from
    kinematic action segmentation -> build selection manifest -> generate scene
    guidelines -> synthesize process knowledge.
    Args:
        expert_json: Path to the expert.json file.
        out_dir: Directory to save the output files.
        vlm_model: The VLM model to use for generating scene guidelines and synthesizing process knowledge.
        force_kinematic: force re-running kinematic action segmentation on expert.mp4 even
            if a cached report already exists.
        mask_path: Optional ROI mask path for expert video.
    Returns:
        Path to the process_knowledge.json file.
    """
    out_dir = Path(out_dir)
    manifest_path = out_dir / "selected_frames.json"
    process_knowledge_path = out_dir / "process_knowledge.json"

    auto_select_frames_from_kinematic(
        expert_json_path=expert_json, frames_dir=out_dir / "frames", force_kinematic=force_kinematic)

    build_selection_manifest(expert_json, out_dir / "frames", manifest_path, mask_path=mask_path)

    vlm_client = OpenRouterClient(model=vlm_model)
    generate_scene_guidelines(manifest_path, vlm_client, mask_path=mask_path)
    synthesize_process_knowledge(manifest_path, vlm_client, process_knowledge_path, mask_path=mask_path)
    return process_knowledge_path

