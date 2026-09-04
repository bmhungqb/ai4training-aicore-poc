"""Phase 1: worker action segmentation (kinematic pre-segmentation).

Provides data structures and loaders for physical motion boundaries produced by
the multi-modal SAM3 + SEA-RAFT + Magnitude/Direction Kinematic Pipeline
(from src/kinematic_pipeline/exp_pipe1.py). No VLM, no expert knowledge — this
phase only finds WHERE the worker's actions start/stop, not WHAT they are.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

from src.config import phase1_segmentation as cfg0


@dataclass
class KinematicSegment:
    segment_idx: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    boundary_type: str = "UNKNOWN"
    confidence: float = 1.0
    sources: list[str] = field(default_factory=list)


@dataclass
class KinematicReport:
    video_path: str
    fps: float
    n_segments: int
    segments: list[KinematicSegment]
    boundaries_s: list[float]
    boundaries_f: list[int]
    report_file: Path | None = None

    def get_segment_for_time(self, t: float) -> KinematicSegment | None:
        """Find the segment covering timestamp t."""
        for seg in self.segments:
            if seg.start_time_s <= t <= seg.end_time_s:
                return seg
        return None


def parse_kinematic_report(report_path: str | Path,
                           boundaries_npy_path: str | Path | None = None) -> KinematicReport:
    """Parse pipe1_report.json (and optional direction_boundaries.npy) into a structured KinematicReport."""
    r_path = Path(report_path)
    if not r_path.exists():
        raise FileNotFoundError(f"Kinematic report not found at {r_path}")

    data = json.loads(r_path.read_text(encoding="utf-8"))
    fps = float(data.get("fps", 25.0))
    video_path = str(data.get("video", ""))

    # Try loading boundary details if available
    boundary_details_map: dict[int, dict] = {}
    if boundaries_npy_path and Path(boundaries_npy_path).exists():
        try:
            b_data = np.load(str(boundaries_npy_path), allow_pickle=True).item()
            for det in b_data.get("boundary_details", []):
                boundary_details_map[int(det["frame"])] = det
        except Exception:
            pass

    segments: list[KinematicSegment] = []
    boundaries_f: list[int] = []
    boundaries_s: list[float] = []

    for i, s in enumerate(data.get("segments", [])):
        s_f = int(s.get("start_frame", 0))
        e_f = int(s.get("end_frame", 0))
        t0 = float(s.get("start_time_s", round(s_f / fps, 3)))
        t1 = float(s.get("end_time_s", round(e_f / fps, 3)))
        dur = float(s.get("duration_s", round(t1 - t0, 3)))

        # Match boundary details at end_frame
        det = boundary_details_map.get(e_f, {})
        b_type = det.get("type", "JOINT_MAG_DIR")
        conf = float(det.get("confidence", 0.95))
        sources = det.get("sources", [])

        seg = KinematicSegment(
            segment_idx=i,
            start_frame=s_f,
            end_frame=e_f,
            start_time_s=t0,
            end_time_s=t1,
            duration_s=dur,
            boundary_type=b_type,
            confidence=conf,
            sources=sources,
        )
        segments.append(seg)
        boundaries_f.append(s_f)
        boundaries_s.append(t0)

    if segments:
        boundaries_f.append(segments[-1].end_frame)
        boundaries_s.append(segments[-1].end_time_s)

    return KinematicReport(
        video_path=video_path,
        fps=fps,
        n_segments=len(segments),
        segments=segments,
        boundaries_s=sorted(list(set(boundaries_s))),
        boundaries_f=sorted(list(set(boundaries_f))),
        report_file=r_path,
    )


class KinematicSegmenter:
    """Manager for obtaining and parsing kinematic segmentation results."""

    def __init__(self,
                 video_path: Path = cfg0.WORKER_VIDEO,
                 out_dir: Path = cfg0.KINEMATIC_OUT_DIR,
                 report_path: Path | None = None,
                 boundaries_path: Path | None = None,
                 mask_path: Path | None = None):
        self.video_path = Path(video_path)
        self.out_dir = Path(out_dir)
        self.report_path = Path(report_path) if report_path is not None else self.out_dir / "pipe1_report.json"
        self.boundaries_path = Path(boundaries_path) if boundaries_path is not None else self.out_dir / "action_boundaries_dynamic.npy"
        # ROI mask restricting SAM3/SEA-RAFT to one worker/expert's area when the
        # frame also shows someone else (see tools/mask_editor for drawing masks).
        # Explicit mask_path wins; otherwise auto-detect "<video>.mask.png" next to
        # the video; None if neither exists (full-frame processing).
        if mask_path is not None:
            self.mask_path = Path(mask_path)
        else:
            default_mask = self.video_path.with_suffix("").with_suffix(".mask.png")
            self.mask_path = default_mask if default_mask.exists() else None

    def locate_existing_report(self) -> KinematicReport | None:
        """Search in primary out_dir as well as the kinematic sub-pipeline's own fallback output dir."""
        candidates = [
            (self.report_path, self.boundaries_path),
            (cfg0.SEGMENT_FLOW_OUTPUT / "pipe1_report.json",
             cfg0.SEGMENT_FLOW_OUTPUT / "action_boundaries_dynamic.npy"),
        ]
        for rep, bnd in candidates:
            if rep.exists():
                try:
                    return parse_kinematic_report(rep, bnd if bnd.exists() else None)
                except Exception as e:
                    print(f"Warning: Failed reading kinematic report at {rep}: {e}")
        return None

    def run(self, force: bool = False,
            angle_tolerance: float = cfg0.ANGLE_TOLERANCE,
            min_segment_len: int = cfg0.MIN_SEGMENT_LEN,
            min_distance: float = cfg0.MIN_DISTANCE,
            prominence: float = cfg0.PROMINENCE,
            smooth_window: int = cfg0.SMOOTH_WINDOW,
            require_both: bool = cfg0.REQUIRE_BOTH,
            visualize: bool = False,
            action_segments_path: Path | None = None,
            resize_scale: float | None = None,
            frame_step: int | None = None,
            frame_by_frame: bool = False) -> KinematicReport:
        """Run or load kinematic segmentation.

        visualize: also let the kinematic sub-pipeline render its annotated debug
        video + boundary plots (skipped by default via --no-video, since it's an
        extra, slow rendering pass only useful when inspecting boundaries by
        eye). Output lands in `self.out_dir` (e.g. `<video_stem>_pipe1_viz.mp4`).

        action_segments_path: where to save the resulting action_segments.json.
        Defaults to the Phase-1/worker path (cfg0.ACTION_SEGMENTS_PATH) — pass an
        explicit path when segmenting a different video (e.g. expert.mp4).
        """
        action_segments_path = Path(action_segments_path) if action_segments_path else (self.out_dir / "action_segments.json")
        if not force:
            existing = self.locate_existing_report()
            if existing is not None:
                print(f"[Kinematic] Found existing report at {existing.report_file} "
                      f"({existing.n_segments} segments) -> Reusing.")
                save_action_segments(existing, action_segments_path)
                return existing

        self.out_dir.mkdir(parents=True, exist_ok=True)
        mask_note = f" (ROI mask: {self.mask_path})" if self.mask_path else " (full frame, no mask)"
        print(f"[Kinematic] Running kinematic segmentation on {self.video_path}...{mask_note}")

        # Invoke src/kinematic_pipeline/exp_pipe1.py
        exp_script = cfg0.KINEMATIC_PIPELINE_DIR / "exp_pipe1.py"
        if not exp_script.exists():
            raise FileNotFoundError(
                f"Cannot find exp_pipe1.py at {exp_script}. Please ensure src/kinematic_pipeline is available.")

        cmd = [
            sys.executable, str(exp_script),
            "--video", str(self.video_path.resolve()),
            "--output", str(self.out_dir.resolve()),
            "--recompute-segmentation",
            "--angle-tolerance", str(angle_tolerance),
            "--min-segment-len", str(min_segment_len),
            "--min-distance", str(min_distance),
            "--prominence", str(prominence),
            "--smooth-window", str(smooth_window),
        ]
        if self.mask_path is not None:
            cmd += ["--mask", str(self.mask_path.resolve())]
        if resize_scale is not None:
            cmd += ["--resize-scale", str(resize_scale)]
        if frame_step is not None:
            cmd += ["--frame-step", str(frame_step)]
        if frame_by_frame:
            cmd.append("--frame-by-frame")
        cmd += [
            "--no-classify",
        ]
        if not visualize:
            cmd.append("--no-video")
        if require_both:
            cmd.append("--require-both")
        else:
            cmd.append("--union-fusion")

        subprocess.run(cmd, check=True)

        report = parse_kinematic_report(self.out_dir / "pipe1_report.json",
                                       self.out_dir / "action_boundaries_dynamic.npy")
        print(f"[Kinematic] Successfully generated {report.n_segments} segments.")
        if visualize:
            viz_path = self.out_dir / f"{self.video_path.stem}_pipe1_viz.mp4"
            print(f"[visualize] Annotated debug video -> {viz_path}")
        save_action_segments(report, action_segments_path)
        return report


def save_action_segments(report: KinematicReport, out_path: Path | None = None) -> Path:
    """Save a KinematicReport as action_segments.json: the Phase 1 output
    consumed by Phase 2's classify step — plain boundaries, no labels."""
    out_path = Path(out_path) if out_path else cfg0.ACTION_SEGMENTS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_path": report.video_path or str(cfg0.WORKER_VIDEO),
        "fps": report.fps,
        "n_segments": report.n_segments,
        "segments": [
            {
                "segment_idx": s.segment_idx,
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "start_time_s": s.start_time_s,
                "end_time_s": s.end_time_s,
                "duration_s": s.duration_s,
                "boundary_type": s.boundary_type,
                "confidence": s.confidence,
                "sources": s.sources,
            }
            for s in report.segments
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Kinematic] Saved {report.n_segments} action segment(s) to {out_path}")
    return out_path


def parse_action_segments(path: str | Path) -> KinematicReport:
    """Load action_segments.json (Phase 1 output) back into a KinematicReport,
    for Phase 2's classify step to consume."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = [
        KinematicSegment(
            segment_idx=s["segment_idx"], start_frame=s["start_frame"], end_frame=s["end_frame"],
            start_time_s=s["start_time_s"], end_time_s=s["end_time_s"], duration_s=s["duration_s"],
            boundary_type=s.get("boundary_type", "UNKNOWN"), confidence=s.get("confidence", 1.0),
            sources=s.get("sources", []),
        )
        for s in data["segments"]
    ]
    boundaries_f = sorted({s.start_frame for s in segments} | {s.end_frame for s in segments})
    boundaries_s = sorted({s.start_time_s for s in segments} | {s.end_time_s for s in segments})
    return KinematicReport(
        video_path=data.get("video_path", ""), fps=data.get("fps", 25.0),
        n_segments=len(segments), segments=segments,
        boundaries_s=boundaries_s, boundaries_f=boundaries_f, report_file=path,
    )
