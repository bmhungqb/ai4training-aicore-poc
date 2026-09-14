"""Phase 2 (classify step): classify the worker's action segments produced by
Phase 1 (kinematic pre-segmentation) against the expert's manifest of scenes/
operations (Phase 2's "expert" step output), using the VLM.

Takes Phase 1's `action_segments.json` (physical motion boundaries, no
labels) plus `selected_frames.json` (the expert's ordered scenes, each with
its operation name, how-to steps, product states) and labels each physical
segment with the operation name it matches, flagging off-standard technique
where the VLM sees a deviation.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

import cv2
import numpy as np

from src.config.phase2_classify import (
    CUTS_DIR, MAX_STEP_FPS, MIN_STEP_FPS, MODEL, OUT_DIR, TIMELINE_DEBUG_PATH, WORKER_FRAMES_DIR,
    WORKER_VIDEO)
from src.config.phase1_segmentation import ACTION_SEGMENTS_PATH
from src.config.phase2_expert import MANIFEST_PATH
from src.manifest import expected_duration as scene_expected_duration
from src.manifest import format_product_state, ordered_scene_items, scene_op_name
from src.prompts.kinematic_classify_prompts import SYSTEM_KINEMATIC_CLASSIFY, USER_KINEMATIC_CLASSIFY
from src.segmentation.kinematic import KinematicReport, KinematicSegment, parse_action_segments
from src.utils.frames import encode_expert_frame, encode_worker_frame, find_mask_for_video
from src.utils.message_content import labeled_frames, render_template_content, text_content
from src.utils.video import cut_clip, sample_window_frames_cached
from src.vlm_client import OpenRouterClient, BatchedVlmClient
from src.utils.motion_viz import (
    compute_mhi, load_frames_from_paths, create_motion_composite, encode_motion_composite,
)


# ---------------------------------------------------------------------------
# Macro-window clustering for two-pass classification
# ---------------------------------------------------------------------------

# Confidence below which a VLM UNKNOWN reply triggers a note in the segment
# rather than being silently absorbed.
VLM_UNCERTAIN_THRESHOLD = 0.7

@dataclass
class MacroWindow:
    """A macro window grouping consecutive micro-segments for batched VLM classification.

    Solves the granularity mismatch: Stage 1 produces 60-134 micro-segments (0.5-1.2s each),
    but the expert labels 24 macro-steps (2-6s each). Clustering micro-segments into
    macro-windows reduces VLM calls 5-10x while improving label consistency.
    """
    window_idx: int
    start_time: float
    end_time: float
    segments: list[KinematicSegment] = field(default_factory=list)
    expected_operations: list[str] = field(default_factory=list)
    # Motion stats aggregated from constituent segments
    avg_speed: float = 0.0
    avg_turbulence: float = 0.0
    dominant_direction: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    def segment_indices(self) -> list[int]:
        return [s.segment_idx for s in self.segments]


def _norm_op(s: str, min_len: int = 3) -> str:
    """Normalize operation name for fuzzy matching with length guard.

    Dropping strings shorter than min_len prevents spurious substring matches
    (e.g. raw="h" matching any op_name containing "how").
    """
    normalized = re.sub(r'\s+', ' ', re.sub(r'[\(\)\[\]]', ' ', s)).strip().lower()
    return normalized if len(normalized) >= min_len else ""


def _fuzzy_match(norm_raw: str, op_name: str) -> bool:
    """Check whether norm_raw matches op_name via substring fuzzy logic.

    Applies only when both strings are long enough to carry meaning
    (>= 3 chars each) to avoid short-noise matches.
    """
    if not norm_raw or not op_name:
        return False
    norm_op = _norm_op(op_name)
    return (
        norm_raw == norm_op
        or (len(norm_raw) >= 3 and norm_raw in norm_op)
        or (len(norm_op) >= 3 and norm_op in norm_raw)
    )


def _find_scene_at_time(t: float, ordered_scenes: list[tuple[str, dict]]) -> str | None:
    """Find the expert scene whose [timestamp_start, timestamp_end] contains time t."""
    for sid, sd in ordered_scenes:
        t0 = sd.get("timestamp_start", 0)
        t1 = sd.get("timestamp_end", float("inf"))
        if t0 <= t < t1:
            return sid
        # Handle last scene inclusive
        if t == t1:
            return sid
    return None


def cluster_into_macrowindows(
    segments: list[KinematicSegment],
    ordered_scenes: list[tuple[str, dict]],
    max_window_duration: float = 4.0,
    min_segments_per_window: int = 2,
    npz_path: Path | None = None,
) -> list[MacroWindow]:
    """Cluster consecutive KinematicSegments into MacroWindows matching expert scene durations.

    Uses scene boundaries as natural breakpoints (expert defines 24 macro-steps),
    and clusters within scenes when segments are too short.

    Args:
        segments: KinematicSegments from Stage 1.
        ordered_scenes: Expert scenes from manifest.
        max_window_duration: Max duration for a single window (seconds).
        min_segments_per_window: Min segments to form a window before forcing a break.
        npz_path: Path to decomposed_motion.npz for real motion statistics (speed, turbulence).

    Returns:
        List of MacroWindow objects covering all segments.
    """
    if not segments:
        return []

    # Pre-load motion stats once if npz is available
    motion_stats_cache: dict | None = None
    if npz_path:
        motion_stats_cache = _load_motion_stats_for_segments(segments, npz_path)

    windows: list[MacroWindow] = []
    current_group: list[KinematicSegment] = []
    current_start = 0.0
    current_scene = None

    for seg in segments:
        seg_scene = _find_scene_at_time(seg.start_time_s, ordered_scenes)

        # Force break: window exceeds max duration OR scene changes
        if current_group:
            window_duration = seg.end_time_s - current_start
            exceeds_max = window_duration > max_window_duration

            # Force break if: exceeds max AND we have enough segments
            if exceeds_max and len(current_group) >= min_segments_per_window:
                windows.append(_make_macrowindow(
                    windows, current_group, current_start,
                    seg.start_time_s, ordered_scenes,
                    motion_stats=motion_stats_cache,
                ))
                current_group = []
                current_scene = None

        if not current_group:
            current_start = seg.start_time_s
            current_scene = seg_scene

        current_group.append(seg)

    # Flush last window
    if current_group:
        windows.append(_make_macrowindow(
            windows, current_group, current_start,
            current_group[-1].end_time_s, ordered_scenes,
            motion_stats=motion_stats_cache,
        ))

    return windows


def _load_motion_stats_for_segments(
    segments: list[KinematicSegment],
    npz_path: Path,
) -> dict:
    """Load aggregated motion stats from decomposed_motion.npz for a list of segments.

    Returns speed_median and turbulence from actual kinematic analysis,
    not a placeholder like segment confidence.
    """
    if not npz_path.exists():
        return {}
    try:
        data = np.load(str(npz_path), allow_pickle=True)
        fps = float(data["fps"])
        n = data["left_smooth_speeds"].shape[0]
        times = np.arange(n) / fps

        speeds = data["left_smooth_speeds"]
        turb = data["left_smooth_turbulences"]
        likelihood = data["overall_likelihood"]

        # Collect all speed/turbulence values for these segments
        seg_speeds = []
        seg_turbs = []
        seg_liks = []
        for seg in segments:
            mask = (times >= seg.start_time_s) & (times <= seg.end_time_s)
            if mask.sum() > 0:
                seg_speeds.extend(speeds[mask].tolist())
                seg_turbs.extend(turb[mask].tolist())
                seg_liks.extend(likelihood[mask].tolist())

        if not seg_speeds:
            return {}
        return {
            "speed_median": float(np.median(seg_speeds)),
            "speed_rms": float(np.sqrt(np.mean(np.array(seg_speeds) ** 2))),
            "turbulence": float(np.median(seg_turbs)),
            "avg_likelihood": float(np.mean(seg_liks)),
        }
    except Exception:
        return {}


def _make_macrowindow(
    windows: list[MacroWindow],
    segments: list[KinematicSegment],
    start_time: float,
    end_time: float,
    ordered_scenes: list[tuple[str, dict]],
    motion_stats: dict | None = None,
) -> MacroWindow:
    """Create a MacroWindow from a group of segments with aggregated stats.

    motion_stats: real motion statistics from decomposed_motion.npz (speed_median,
    speed_rms, turbulence). When None, falls back to confidence as proxy.
    """
    # Collect expected operations from scene boundaries within this window
    expected_ops: list[str] = []
    seen_ops = set()
    for t in np.linspace(start_time, end_time, max(3, len(segments) * 2)):
        sid = _find_scene_at_time(t, ordered_scenes)
        if sid:
            sd = dict(ordered_scenes).get(sid)
            if sd:
                op_name = scene_op_name(sd)
                if op_name not in seen_ops:
                    seen_ops.add(op_name)
                    expected_ops.append(op_name)

    # Use real motion stats from npz if available; confidence as last-resort fallback
    if motion_stats:
        avg_speed = motion_stats.get("speed_median", 0.0)
        avg_turbulence = motion_stats.get("turbulence", 0.0)
    else:
        confidences = [s.confidence for s in segments]
        avg_speed = float(np.mean(confidences)) if confidences else 0.0
        avg_turbulence = 0.0

    return MacroWindow(
        window_idx=len(windows),
        start_time=start_time,
        end_time=end_time,
        segments=segments,
        expected_operations=expected_ops,
        avg_speed=avg_speed,
        avg_turbulence=avg_turbulence,
        dominant_direction=0.0,
    )


def format_process_overview(scenes_ordered: list[tuple[str, dict]]) -> str:
    """Format the process overview as a string."""
    return "\n".join(f"{i}. {scene_op_name(sd)} (~{scene_expected_duration(sd):.1f}s)"
                     for i, (sid, sd) in enumerate(scenes_ordered, 1))


def format_candidate_operations(scenes_ordered: list[tuple[str, dict]]) -> str:
    """Format candidate operations for VLM classification against pre-segmented intervals."""
    lines = []
    for i, (sid, sd) in enumerate(scenes_ordered, 1):
        name = scene_op_name(sd)
        dur = scene_expected_duration(sd)
        lines.append(f"{i}. \"{name}\" (~{dur:.1f}s)")
        how_to = format_how_to(sd.get("guideline", {}))
        if how_to and how_to != "(none)":
            lines.append(f"   Steps: {how_to}")
        p_state = format_product_state(sd.get("guideline", {}))
        lines.append(f"   {p_state}")
    lines.append(f"{len(scenes_ordered)+1}. \"UNKNOWN\" (Hành động không khớp bất kỳ thao tác nào ở trên, hoặc bị che khuất/không rõ ràng)")
    lines.append(f"{len(scenes_ordered)+2}. \"IDLE\" (Công nhân dừng tay, không thao tác gì, nghỉ hoặc chờ đợi)")
    return "\n".join(lines)


def format_how_to(guideline: dict) -> str:
    """Format the how-to steps from the guideline as a numbered list."""
    steps = guideline.get("how_to_steps", [])
    return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)) or "(none)"


def build_kinematic_classify_messages(task_name: str, scenes_ordered: list[tuple[str, dict]],
                                      expert_ref_paths: list[str], frames: list[tuple[float, str]],
                                      t0: float, t1: float,
                                      crop_save_dir: str | Path | None = None,
                                      expert_mask_path: str | Path | None = None,
                                      worker_mask_path: str | Path | None = None) -> list[dict]:
    """Build messages for classifying a kinematic physical slice against manifest operations."""
    system_prompt = Template(SYSTEM_KINEMATIC_CLASSIFY).substitute(
        task_name=task_name, process_overview=format_process_overview(scenes_ordered))

    ref_content: list[dict] = [text_content(
        "Khung hình tham chiếu từ video CHUYÊN GIA (chỉ để đối chiếu kỹ thuật chuẩn):")]
    ref_content += labeled_frames(
        (f"[Expert reference] {Path(fp).name}:", encode_expert_frame(fp, mask_path=expert_mask_path))
        for fp in expert_ref_paths)

    win_content = labeled_frames(
        (f"[Frame #{i}, t={ts:.1f}s]", encode_worker_frame(fp, mask_path=worker_mask_path, save_dir=crop_save_dir))
        for i, (ts, fp) in enumerate(frames, start=1))

    candidate_ops = format_candidate_operations(scenes_ordered)
    subs = dict(
        start_time_s=f"{t0:.2f}",
        end_time_s=f"{t1:.2f}",
        duration_s=f"{t1 - t0:.2f}",
        candidate_operations_text=candidate_ops,
    )
    user_content = render_template_content(
        USER_KINEMATIC_CLASSIFY, subs,
        {"expert_ref_frames_content": ref_content, "window_frames_content": win_content})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Batched Segment Classifier (Solves Latency/Cost + Granularity Mismatch)
# ---------------------------------------------------------------------------

class BatchedSegmentClassifier:
    """Classify worker segments using:
    1. Macro-window clustering (reduces granularity mismatch)
    2. Motion-enhanced composite frames (solves temporal blindness)
    3. Concurrent VLM calls via BatchedVlmClient (solves latency/cost explosion)

    This is the recommended classifier for POC — replaces sequential SegmentClassifier.run().
    """

    def __init__(self, manifest_path=MANIFEST_PATH,
                 video_path=WORKER_VIDEO,
                 out_dir=OUT_DIR,
                 frames_dir=WORKER_FRAMES_DIR,
                 model=MODEL,
                 save_crop_frames=False,
                 action_segments_path=ACTION_SEGMENTS_PATH,
                 mask_path=None,
                 expert_mask_path=None,
                 max_workers: int = 15,
                 max_window_duration: float = 4.0,
                 use_motion_composite: bool = True,
                 max_frames_per_call: int = 4):
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.manifest = manifest
        self.task_name = manifest["task_name"]
        self.ordered_scenes = ordered_scene_items(manifest)
        self.video_path = Path(video_path)
        self.video_duration = self._probe_duration(self.video_path)
        self.action_segments_path = Path(action_segments_path)

        # Masks
        self.worker_mask_path = find_mask_for_video(
            self.video_path, action_segments_path=self.action_segments_path, explicit_mask=mask_path)
        manifest_expert_mask = manifest.get("mask_path")
        self.expert_mask_path = (Path(expert_mask_path) if expert_mask_path
                                 else (Path(manifest_expert_mask) if manifest_expert_mask
                                       else find_mask_for_video(manifest.get("video_path"))))

        self.frames_dir = Path(frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.crop_frames_dir = self.frames_dir.parent / f"{self.frames_dir.name}_cropped" \
            if save_crop_frames else None

        # Batched VLM client for concurrent calls
        self.batch_client = BatchedVlmClient(model=model, max_workers=max_workers)
        self._call_n = 0
        self.run_cost = 0.0
        self.max_window_duration = max_window_duration
        self.use_motion_composite = use_motion_composite
        self.max_frames_per_call = max_frames_per_call

    @staticmethod
    def _probe_duration(video_path: Path) -> float:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n_frames / fps

    def _load_action_segments(self) -> KinematicReport:
        if not self.action_segments_path.exists():
            raise SystemExit(
                f"Missing {self.action_segments_path} — run `python pipeline.py segment` first.")
        return parse_action_segments(self.action_segments_path)

    def _build_expert_refs(self) -> list[str]:
        """Build deduplicated expert reference frame paths (max 8 for prompt length)."""
        all_paths = []
        for _, sd in self.ordered_scenes:
            all_paths.extend(sd.get("frames", []))
        seen, dedup = set(), []
        for p in all_paths:
            if p not in seen:
                seen.add(p)
                dedup.append(p)
        return dedup[:8]

    def _load_motion_stats(self, start_s: float, end_s: float) -> dict:
        """Load aggregated motion stats from decomposed_motion.npz if available.

        The npz file contains arrays indexed by frame number, with no explicit times
        array. We reconstruct a time index from the stored fps and array length:
            times[i] = i / fps

        Available arrays (all same length): left_smooth_speeds, left_smooth_turbulences,
        overall_likelihood (per-frame confidence). Falls back to overall_likelihood if
        no speed array is found.
        """
        npz_path = self.action_segments_path.parent / "decomposed_motion.npz"
        if not npz_path.exists():
            return {}
        try:
            data = np.load(str(npz_path), allow_pickle=True)
            fps = float(data["fps"])
            n = data["left_smooth_speeds"].shape[0]
            times = np.arange(n) / fps

            mask = (times >= start_s) & (times <= end_s)
            if mask.sum() == 0:
                return {}

            speeds = data["left_smooth_speeds"]
            turb = data["left_smooth_turbulences"]
            likelihood = data["overall_likelihood"]

            return {
                "speed_median": float(np.median(speeds[mask])),
                "speed_rms": float(np.sqrt(np.mean(speeds[mask] ** 2))),
                "turbulence": float(np.median(turb[mask])),
                "avg_likelihood": float(np.mean(likelihood[mask])),
            }
        except Exception:
            pass
        return {}

    def _classify_window(self, window: MacroWindow, expert_ref_paths: list[str]) -> dict:
        """Classify a single macro-window: builds messages with motion composite."""
        t0, t1 = window.start_time, window.end_time
        dur = t1 - t0
        if dur <= 0.05:
            return None

        # Sample frames within window
        sample_fps = max(MIN_STEP_FPS, min(MAX_STEP_FPS, 8.0 / max(dur, 0.5)))
        frames = sample_window_frames_cached(self.video_path, t0, t1, sample_fps, self.frames_dir)
        if not frames:
            return None

        # Get motion stats
        motion_stats = self._load_motion_stats(t0, t1)

        # Build messages
        if self.use_motion_composite and len(frames) >= 2:
            messages = self._build_motion_enhanced_messages(
                window, frames, expert_ref_paths, motion_stats)
        else:
            messages = build_kinematic_classify_messages(
                task_name=self.task_name,
                scenes_ordered=self.ordered_scenes,
                expert_ref_paths=expert_ref_paths,
                frames=frames,
                t0=t0, t1=t1,
                crop_save_dir=self.crop_frames_dir,
                expert_mask_path=self.expert_mask_path,
                worker_mask_path=self.worker_mask_path,
            )

        return {"messages": messages, "window": window, "frames": frames, "motion_stats": motion_stats}

    def _build_motion_enhanced_messages(
        self, window: MacroWindow, frames: list[tuple[float, str]],
        expert_ref_paths: list[str], motion_stats: dict,
    ) -> list[dict]:
        """Build messages with motion-enhanced composite for temporal understanding."""
        system_prompt = Template(SYSTEM_KINEMATIC_CLASSIFY).substitute(
            task_name=self.task_name,
            process_overview=format_process_overview(self.ordered_scenes))

        # Expert reference frames
        ref_content: list[dict] = [text_content(
            "Khung hình tham chiếu từ video CHUYÊN GIA (kỹ thuật chuẩn):")]
        ref_content += labeled_frames(
            (f"[Expert ref] {Path(fp).name}:",
             encode_expert_frame(fp, mask_path=self.expert_mask_path))
            for fp in expert_ref_paths)

        # Motion-enhanced worker content
        frame_paths = [fp for _, fp in frames]
        frame_ts = [ts for ts, _ in frames]

        # Limit frames to max_frames_per_call
        if len(frame_paths) > self.max_frames_per_call:
            step = len(frame_paths) / self.max_frames_per_call
            indices = sorted({round(i * step) for i in range(self.max_frames_per_call)})
            frame_paths = [frame_paths[i] for i in indices]
            frame_ts = [frame_ts[i] for i in indices]

        if len(frame_paths) >= 2:
            # Create motion composite (MHI + key frame)
            composite_b64 = encode_motion_composite(
                frame_paths, motion_stats=motion_stats, target_width=1024)
            motion_content = [
                {"type": "text", "text": (
                    f"[Motion Composite — window {window.window_idx}, {window.duration:.1f}s, "
                    f"{window.n_segments} micro-segments]\n"
                    f"Motion stats: speed={motion_stats.get('speed_median', 0):.3f}, "
                    f"turbulence={motion_stats.get('turbulence', 0):.3f}\n"
                    f"Expected operations: {', '.join(window.expected_operations) or '(not specified)'}"
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{composite_b64}"}},
            ]
        else:
            # Fall back to single frame
            motion_content = labeled_frames(
                (f"[Frame t={ts:.1f}s]", encode_worker_frame(
                    fp, mask_path=self.worker_mask_path, save_dir=self.crop_frames_dir))
                for ts, fp in zip(frame_ts, frame_paths))

        candidate_ops = format_candidate_operations(self.ordered_scenes)
        user_text = f"""Phân loại hành động của công nhân trong khoảng thời gian [{window.start_time:.1f}s - {window.end_time:.1f}s] ({window.duration:.1f}s, {window.n_segments} vi-thao tác).

Các khung hình worker bên dưới có MHI overlay (Motion History Image) cho thấy quỹ đạo chuyển động. Màu nóng (đỏ/vàng) = chuyển động gần đây, màu lạnh (xanh) = chuyển động cũ.

Candidate operations:
{candidate_ops}

Hãy phân loại hành động này dựa trên CHUYỂN ĐỘNG (motion trajectory trong ảnh MHI) và KHUNG HÌNH CHUYÊN GIA làm chuẩn.

ĐÁP ÁN (JSON only, không giải thích):
{{"operation_name": "...", "reasoning": "...", "off_standard": true/false, "off_standard_description": "..."}}"""
        user_content = [{"type": "text", "text": user_text}]

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [*ref_content, *motion_content, *user_content]},
        ]

    def _parse_window_result(self, window: MacroWindow, reply: dict | str,
                             frames: list, motion_stats: dict) -> list[dict]:
        """Parse VLM reply and expand window result to per-segment labels."""
        if isinstance(reply, str):
            # Error fallback
            return [{
                "start_time": round(window.start_time, 2),
                "end_time": round(window.end_time, 2),
                "operation_name": "UNKNOWN",
                "off_standard": True,
                "off_standard_desc": str(reply),
                "evidence": str(reply),
                "action_evidence": "",
                "product_state_evidence": "",
                "n_vlm_calls": 1,
                "cost_usd": 0.0,
                "worker_duration_s": round(window.duration, 2),
                "worker_frame_count": len(frames),
                "motion_stats": motion_stats,
                "kinematic_data": {"window_idx": window.window_idx, "n_segments": window.n_segments},
                "model_output": {"error": str(reply)},
            }]

        raw_op = str(reply.get("operation_name", "UNKNOWN")).strip().strip('"').strip("'")
        norm_raw = _norm_op(raw_op)
        matched_op = "UNKNOWN"
        for _, sd in self.ordered_scenes:
            op_name = scene_op_name(sd)
            if _fuzzy_match(norm_raw, op_name):
                matched_op = op_name
                break
        if raw_op.upper() in ["IDLE", "UNKNOWN", "NONE"]:
            matched_op = "UNKNOWN"

        off_standard = bool(reply.get("off_standard", False))
        off_desc = reply.get("off_standard_description", "")
        reasoning = reply.get("reasoning", "")
        action_ev = reply.get("action_evidence", "")
        product_ev = reply.get("product_state_evidence", "")

        # Flag uncertain results: VLM returned UNKNOWN on a low-confidence kinematic window.
        # This means neither the motion signal nor the VLM is confident — surface it
        # so downstream steps (micro_eval) can decide whether to investigate further.
        vlm_uncertain = (
            matched_op == "UNKNOWN"
            and window.avg_speed < VLM_UNCERTAIN_THRESHOLD
        )

        # Expand window result to per-segment results
        results = []
        for seg in window.segments:
            s_t0 = max(0.0, seg.start_time_s)
            s_t1 = min(self.video_duration, seg.end_time_s)
            s_dur = s_t1 - s_t0
            results.append({
                "start_time": round(s_t0, 2),
                "end_time": round(s_t1, 2),
                "operation_name": matched_op,
                "off_standard": off_standard,
                "off_standard_desc": off_desc,
                "evidence": reasoning or action_ev,
                "action_evidence": action_ev,
                "product_state_evidence": product_ev,
                "vlm_uncertain": vlm_uncertain,
                "n_vlm_calls": 1,
                "cost_usd": 0.0,  # Cost tracked at batch level
                "worker_duration_s": round(s_dur, 2),
                "worker_frame_count": 0,
                "motion_stats": motion_stats,
                "kinematic_data": {
                    "segment_idx": seg.segment_idx,
                    "boundary_type": seg.boundary_type,
                    "confidence": seg.confidence,
                    "window_idx": window.window_idx,
                },
                "model_output": reply,
            })
        return results

    def _progress_callback(self, completed: int, total: int):
        """Print progress during batch classification."""
        print(f"\r  Batch progress: {completed}/{total} windows ({100*completed/total:.0f}%)", end="", flush=True)

    def run(self) -> dict:
        """Run batched classification: cluster into macro-windows, classify concurrently."""
        report = self._load_action_segments()
        n_seg = len(report.segments)
        print(f"\n[BatchedSegmentClassifier] {n_seg} micro-segments")

        # Step 1: Cluster into macro-windows (pass npz_path for real motion stats)
        windows = cluster_into_macrowindows(
            report.segments, self.ordered_scenes,
            max_window_duration=self.max_window_duration,
            npz_path=self.action_segments_path.parent / "decomposed_motion.npz",
        )
        print(f"[BatchedSegmentClassifier] -> {len(windows)} macro-windows")

        expert_ref_paths = self._build_expert_refs()

        # Step 2: Build all VLM requests
        requests = []
        for window in windows:
            req = self._classify_window(window, expert_ref_paths)
            if req:
                requests.append(req)

        # Step 3: Execute batch concurrently
        print(f"[BatchedSegmentClassifier] Executing {len(requests)} VLM calls concurrently...")
        t0 = time.monotonic()
        batch_results = self.batch_client.classify_batch(requests, progress_callback=self._progress_callback)
        elapsed = time.monotonic() - t0
        print(f"\n[BatchedSegmentClassifier] Done in {elapsed:.1f}s")

        # Step 4: Parse results and expand to per-segment
        raw_segments: list[dict] = []
        for req, result in zip(requests, batch_results):
            window = req["window"]
            frames = req["frames"]
            motion_stats = req["motion_stats"]
            if result.success:
                parsed = self._parse_window_result(window, result.data, frames, motion_stats)
                raw_segments.extend(parsed)
            else:
                print(f"\n  [!] Window {window.window_idx} failed: {result.error}")
                parsed = self._parse_window_result(window, result.error or "unknown error",
                                                  frames, motion_stats)
                raw_segments.extend(parsed)

        # Step 5: Merge consecutive segments with same label (except UNKNOWN)
        merged_segments: list[dict] = []
        for s in raw_segments:
            if (merged_segments and
                merged_segments[-1]["operation_name"] == s["operation_name"] and
                s["operation_name"] != "UNKNOWN"):
                prev = merged_segments[-1]
                prev["end_time"] = s["end_time"]
                prev["worker_duration_s"] = round(prev["end_time"] - prev["start_time"], 2)
                prev["n_vlm_calls"] += s["n_vlm_calls"]
                prev["worker_frame_count"] += s["worker_frame_count"]
                if s.get("off_standard"):
                    prev["off_standard"] = True
                    if s.get("off_standard_desc"):
                        prev["off_standard_desc"] = (prev.get("off_standard_desc", "") +
                                                    "; " + s["off_standard_desc"]).strip("; ")
            else:
                merged_segments.append(s)

        batch_summary = self.batch_client.summary()
        worker_segments = {
            "task_name": self.task_name,
            "segments": merged_segments,
            "raw_action_segments_count": len(raw_segments),
            "total_cost_usd": round(batch_summary["total_cost_usd"], 4),
            "total_vlm_calls": batch_summary["total_requests"],
            "classification_mode": "batched_macrowindow",
            "n_macrowindows": len(windows),
            "batch_summary": batch_summary,
            "elapsed_s": round(elapsed, 2),
        }
        out_path = self.out_dir / "worker_segments.json"
        out_path.write_text(json.dumps(worker_segments, ensure_ascii=False, indent=2), encoding="utf-8")

        n_off = sum(1 for s in merged_segments if s["off_standard"])
        n_unknown = sum(1 for s in merged_segments if s["operation_name"] == "UNKNOWN")
        print(f"\n[BatchedSegmentClassifier] Saved {len(merged_segments)} merged segment(s) "
              f"({n_off} off-standard, {n_unknown} UNKNOWN)")
        print(f"[BatchedSegmentClassifier] {batch_summary['total_requests']} VLM calls "
              f"in {elapsed:.1f}s, ${batch_summary['total_cost_usd']:.4f}")
        return worker_segments


class SegmentClassifier:
    """Classify each physical action segment Phase 1 produced against the
    expert's manifest of operations, via the VLM."""

    def __init__(self, manifest_path=MANIFEST_PATH,
                 video_path=WORKER_VIDEO,
                 out_dir=OUT_DIR,
                 frames_dir=WORKER_FRAMES_DIR,
                 model=MODEL,
                 save_crop_frames=False,
                 action_segments_path=ACTION_SEGMENTS_PATH,
                 mask_path=None,
                 expert_mask_path=None):
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.manifest = manifest
        self.task_name = manifest["task_name"]
        self.ordered_scenes = ordered_scene_items(manifest)
        self.video_path = Path(video_path)
        self.video_duration = self._probe_duration(self.video_path)
        self.action_segments_path = Path(action_segments_path)

        # Worker mask: explicit mask > auto-detect next to video or in action segments dir
        self.worker_mask_path = find_mask_for_video(
            self.video_path, action_segments_path=self.action_segments_path, explicit_mask=mask_path)
        if self.worker_mask_path:
            print(f"[Phase 2 / classify] Using ROI mask for worker video: {self.worker_mask_path}")

        # Expert mask: explicit > manifest's mask_path > auto-detect
        manifest_expert_mask = manifest.get("mask_path")
        self.expert_mask_path = (Path(expert_mask_path) if expert_mask_path
                                 else (Path(manifest_expert_mask) if manifest_expert_mask
                                       else find_mask_for_video(manifest.get("video_path"))))
        if self.expert_mask_path:
            print(f"[Phase 2 / classify] Using ROI mask for expert reference: {self.expert_mask_path}")

        self.frames_dir = Path(frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.crop_frames_dir = self.frames_dir.parent / f"{self.frames_dir.name}_cropped" \
            if save_crop_frames else None

        self.client = OpenRouterClient(model=model)
        self._call_n = 0
        self.run_cost = 0.0

    @staticmethod
    def _probe_duration(video_path: Path) -> float:
        """Get the duration of the video in seconds."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n_frames / fps

    def _call(self, messages: list[dict]) -> tuple[dict, float]:
        """Call the VLM with the given messages and return the response and cost."""
        reply = self.client.chat_json(messages)
        cost = self.client.last_cost_usd or 0.0
        self.run_cost += cost
        self._call_n += 1
        return reply, cost

    def _classify_segment(self, k_seg: KinematicSegment,
                          all_expert_ref_paths: list[str]) -> dict | None:
        """Classify a single physical action segment against the expert process."""
        t0 = max(0.0, k_seg.start_time_s)
        t1 = min(self.video_duration, k_seg.end_time_s)
        dur = t1 - t0
        if dur <= 0.05:
            return None

        # Sample frames within [t0, t1]
        sample_fps = max(MIN_STEP_FPS, min(MAX_STEP_FPS, 6.0 / max(dur, 0.5)))
        frames = sample_window_frames_cached(self.video_path, t0, t1, sample_fps, self.frames_dir)
        if not frames:
            return None

        messages = build_kinematic_classify_messages(
            task_name=self.task_name,
            scenes_ordered=self.ordered_scenes,
            expert_ref_paths=all_expert_ref_paths,
            frames=frames,
            t0=t0,
            t1=t1,
            crop_save_dir=self.crop_frames_dir,
            expert_mask_path=self.expert_mask_path,
            worker_mask_path=self.worker_mask_path,
        )

        try:
            reply, cost = self._call(messages)
        except Exception as e:
            print(f"[{t0:.1f}s - {t1:.1f}s] Classify error: {e}")
            reply = {"operation_name": "UNKNOWN", "off_standard": True, "evidence": str(e)}
            cost = 0.0

        raw_op = str(reply.get("operation_name", "UNKNOWN")).strip().strip('"').strip("'")
        norm_raw = _norm_op(raw_op)
        matched_op = "UNKNOWN"
        for _, sd in self.ordered_scenes:
            op_name = scene_op_name(sd)
            if _fuzzy_match(norm_raw, op_name):
                matched_op = op_name
                break

        if raw_op.upper() in ["IDLE", "UNKNOWN", "NONE"]:
            matched_op = "UNKNOWN"

        off_standard = bool(reply.get("off_standard", False))
        off_desc = reply.get("off_standard_description", "")
        reasoning = reply.get("reasoning", "")
        action_ev = reply.get("action_evidence", "")
        product_ev = reply.get("product_state_evidence", "")

        flag = " [OFF-STANDARD]" if off_standard else ""
        print(f"[{t0:>6.1f}s - {t1:>6.1f}s] ({dur:.2f}s) -> {matched_op} "
              f"({k_seg.boundary_type}, conf={k_seg.confidence:.2f}, ${cost:.5f}){flag}")
        if reasoning:
            print(f"    -> [reasoning] {reasoning}")

        return {
            "start_time": round(t0, 2),
            "end_time": round(t1, 2),
            "operation_name": matched_op,
            "off_standard": off_standard,
            "off_standard_desc": off_desc,
            "evidence": reasoning or action_ev,
            "action_evidence": action_ev,
            "product_state_evidence": product_ev,
            "n_vlm_calls": 1,
            "cost_usd": round(cost, 5),
            "worker_duration_s": round(dur, 2),
            "worker_frame_count": len(frames),
            "kinematic_data": {
                "segment_idx": k_seg.segment_idx,
                "boundary_type": k_seg.boundary_type,
                "confidence": k_seg.confidence,
                "sources": k_seg.sources,
            },
            "model_output": reply,
        }

    def _load_action_segments(self) -> KinematicReport:
        """Load Phase 1's action_segments.json (worker action boundaries)."""
        if not self.action_segments_path.exists():
            raise SystemExit(
                f"Missing {self.action_segments_path} — run `python pipeline.py segment` "
                "(Phase 1) first.")
        return parse_action_segments(self.action_segments_path)

    def run(self) -> dict:
        """Classify every action segment from Phase 1 against the expert manifest."""
        report = self._load_action_segments()
        print(f"\n[Phase 2: classify] Classifying {len(report.segments)} action segment(s)...")

        all_expert_ref_paths: list[str] = []
        for _, sd in self.ordered_scenes:
            all_expert_ref_paths.extend(sd.get("frames", []))
        seen = set()
        dedup_expert_refs = []
        for p in all_expert_ref_paths:
            if p not in seen:
                seen.add(p)
                dedup_expert_refs.append(p)

        raw_segments: list[dict] = []
        for k_seg in report.segments:
            seg_dict = self._classify_segment(k_seg, dedup_expert_refs[:8])
            if seg_dict is not None:
                raw_segments.append(seg_dict)

        # Merge consecutive segments with identical operation names (except UNKNOWN)
        merged_segments: list[dict] = []
        for s in raw_segments:
            if (merged_segments and
                merged_segments[-1]["operation_name"] == s["operation_name"] and
                s["operation_name"] != "UNKNOWN"):
                prev = merged_segments[-1]
                prev["end_time"] = s["end_time"]
                prev["worker_duration_s"] = round(prev["end_time"] - prev["start_time"], 2)
                prev["n_vlm_calls"] += s["n_vlm_calls"]
                prev["cost_usd"] = round(prev["cost_usd"] + s["cost_usd"], 5)
                prev["worker_frame_count"] += s["worker_frame_count"]
                if s.get("off_standard"):
                    prev["off_standard"] = True
                    if s.get("off_standard_desc"):
                        prev["off_standard_desc"] = (prev.get("off_standard_desc", "") + "; " + s["off_standard_desc"]).strip("; ")
            else:
                merged_segments.append(s)

        worker_segments = {
            "task_name": self.task_name,
            "segments": merged_segments,
            "raw_action_segments_count": len(raw_segments),
            "total_cost_usd": round(self.run_cost, 4),
            "total_vlm_calls": self._call_n,
        }
        out_path = self.out_dir / "worker_segments.json"
        out_path.write_text(json.dumps(worker_segments, ensure_ascii=False, indent=2), encoding="utf-8")

        n_off = sum(1 for s in merged_segments if s["off_standard"])
        n_unknown = sum(1 for s in merged_segments if s["operation_name"] == "UNKNOWN")
        print(f"\nSaved {len(merged_segments)} merged segment(s) ({n_off} off-standard, {n_unknown} UNKNOWN) to {out_path}")
        print(f"Total {self._call_n} VLM call(s), ${self.run_cost:.4f}")
        return worker_segments


def dump_timeline_debug(worker_segments: dict, out_path: str | Path = TIMELINE_DEBUG_PATH) -> None:
    """Write a compact, human-scannable timeline (--visualize debug artifact):
    one line per segment with its time range, matched operation and flags —
    faster to eyeball for misclassification than the full worker_segments.json."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"Task: {worker_segments['task_name']}", ""]
    for i, s in enumerate(worker_segments["segments"]):
        flag = " [OFF-STANDARD]" if s.get("off_standard") else ""
        lines.append(f"{i:02d} [{s['start_time']:>6.1f}s - {s['end_time']:>6.1f}s] "
                     f"{s['operation_name']}{flag}")
        if s.get("off_standard_desc"):
            lines.append(f"     -> {s['off_standard_desc']}")
    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    print(f"\n[visualize] Timeline debug dump -> {out_path}")
    print(text)


# ---------------------------------------------------------------------------
# Cut worker.mp4 into one clip per segment
# ---------------------------------------------------------------------------
def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w]+", "_", name, flags=re.UNICODE).strip("_")
    return slug or "UNKNOWN"


def cut_worker_segments(segments_json_path, video_path=WORKER_VIDEO,
                        out_dir=CUTS_DIR) -> None:
    """Cut the worker video into one clip per segment."""
    segments_json_path = Path(segments_json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(segments_json_path.read_text(encoding="utf-8"))

    n_cut = 0
    for i, seg in enumerate(data["segments"]):
        start, end = seg["start_time"], seg["end_time"]
        duration = end - start
        if duration <= 0:
            print(f"segment {i:02d}: skipped (duration={duration})")
            continue
        flag = "_OFFSTANDARD" if seg.get("off_standard") else ""
        out_path = out_dir / f"{i:02d}_{_slugify(seg['operation_name'])}{flag}.mp4"

        cut_clip(video_path, start, end, out_path)
        n_cut += 1

        note = f" [{seg.get('off_standard_desc', '')}]" if seg.get("off_standard") and seg.get("off_standard_desc") else ""
        print(f"segment {i:02d}: [{start:>6.1f}s - {end:>6.1f}s] ({duration:.1f}s) -> "
              f"{out_path.name}{note}")

    n_lech = sum(1 for s in data["segments"] if s.get("off_standard"))
    n_unknown = sum(1 for s in data["segments"] if s["operation_name"] == "UNKNOWN")
    print(f"\nCut {n_cut} clip(s) into {out_dir}/ ({n_lech} off-standard, {n_unknown} UNKNOWN)")


def main():
    ap = argparse.ArgumentParser(description="Phase 2 (classify step): classify worker action "
                                             "segments against the expert manifest")
    ap.add_argument("--manifest", default=MANIFEST_PATH)
    ap.add_argument("--video", default=WORKER_VIDEO)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--frames-dir", default=WORKER_FRAMES_DIR)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--action-segments", default=ACTION_SEGMENTS_PATH,
                    help="path to Phase 1's action_segments.json")
    ap.add_argument("--cut", action="store_true",
                    help="also cut worker.mp4 into one clip per segment after running")
    ap.add_argument("--save-crop-frames", action="store_true",
                    help="save the cropped+resized frames actually sent to the VLM "
                         "(to <frames-dir>_cropped) for inspection")
    ap.add_argument("--mask", default=None,
                    help="path to worker ROI mask image (defaults to auto-detecting next to video)")
    ap.add_argument("--expert-mask", default=None,
                    help="path to expert ROI mask image (defaults to auto-detecting from manifest or next to video)")
    ap.add_argument("--visualize", action="store_true",
                    help="also write a human-scannable timeline debug dump")
    # Batched classification flags
    ap.add_argument("--batched", action="store_true",
                    help="use BatchedSegmentClassifier (macro-window + motion composite + concurrent VLM calls)")
    ap.add_argument("--max-workers", type=int, default=15,
                    help="max concurrent VLM workers for batched mode (default: 15)")
    ap.add_argument("--max-window-duration", type=float, default=4.0,
                    help="max duration in seconds for one macro-window (default: 4.0s)")
    ap.add_argument("--no-motion-composite", action="store_true",
                    help="disable motion-enhanced composite frames in batched mode")
    ap.add_argument("--max-frames-per-call", type=int, default=4,
                    help="max frames per VLM call in batched mode (default: 4)")
    args = ap.parse_args()

    if args.batched:
        print("[Phase 2: classify] Using BATCHED mode (macro-windows + motion composite + concurrent calls)")
        runner = BatchedSegmentClassifier(
            manifest_path=args.manifest, video_path=args.video,
            out_dir=args.out_dir, frames_dir=args.frames_dir, model=args.model,
            save_crop_frames=args.save_crop_frames,
            action_segments_path=args.action_segments,
            mask_path=args.mask,
            expert_mask_path=args.expert_mask,
            max_workers=args.max_workers,
            max_window_duration=args.max_window_duration,
            use_motion_composite=not args.no_motion_composite,
            max_frames_per_call=args.max_frames_per_call,
        )
    else:
        runner = SegmentClassifier(manifest_path=args.manifest, video_path=args.video,
                                  out_dir=args.out_dir, frames_dir=args.frames_dir, model=args.model,
                                  save_crop_frames=args.save_crop_frames,
                                  action_segments_path=args.action_segments,
                                  mask_path=args.mask,
                                  expert_mask_path=args.expert_mask)

    print(f"task: {runner.task_name} | {len(runner.ordered_scenes)} scene(s) | "
          f"worker duration: {runner.video_duration:.1f}s")
    # Check availability: BatchedSegmentClassifier has batch_client, SegmentClassifier has client
    client_attr = getattr(runner, "batch_client", None) or getattr(runner, "client", None)
    if client_attr and not client_attr.available:
        raise SystemExit("Missing OPENROUTER_API_KEY (env or .env)")
    worker_segments = runner.run()
    if args.visualize:
        dump_timeline_debug(worker_segments, Path(args.out_dir) / "timeline_debug.json")
    if args.cut:
        cut_worker_segments(Path(args.out_dir) / "worker_segments.json",
                            video_path=args.video, out_dir=Path(args.out_dir) / "cuts")


if __name__ == "__main__":
    main()
