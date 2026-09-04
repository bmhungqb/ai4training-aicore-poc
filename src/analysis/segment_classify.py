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
from pathlib import Path
from string import Template

import cv2

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
from src.vlm_client import OpenRouterClient


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

        # Match raw_op to manifest operations
        matched_op = "UNKNOWN"
        for _, sd in self.ordered_scenes:
            op_name = scene_op_name(sd)
            if raw_op.lower() == op_name.lower() or raw_op.lower() in op_name.lower() or op_name.lower() in raw_op.lower():
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
    args = ap.parse_args()

    runner = SegmentClassifier(manifest_path=args.manifest, video_path=args.video,
                               out_dir=args.out_dir, frames_dir=args.frames_dir, model=args.model,
                               save_crop_frames=args.save_crop_frames,
                               action_segments_path=args.action_segments,
                               mask_path=args.mask,
                               expert_mask_path=args.expert_mask)
    print(f"task: {runner.task_name} | {len(runner.ordered_scenes)} scene(s) | "
          f"worker duration: {runner.video_duration:.1f}s")
    if not runner.client.available:
        raise SystemExit("Missing OPENROUTER_API_KEY (env or .env)")
    worker_segments = runner.run()
    if args.visualize:
        dump_timeline_debug(worker_segments, Path(args.out_dir) / "timeline_debug.json")
    if args.cut:
        cut_worker_segments(Path(args.out_dir) / "worker_segments.json",
                            video_path=args.video, out_dir=Path(args.out_dir) / "cuts")


if __name__ == "__main__":
    main()
