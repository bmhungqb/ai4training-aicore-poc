from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class PipelineConfig:
    video: Path
    output: Path
    force: bool = False
    only_step12: bool = False
    recompute_segmentation: bool = False
    mask: Optional[Path] = None  # ROI mask image (white=keep, black=ignore); None = full frame

    # SAM3
    sam_threshold: float = 0.5
    frame_step: int = 1
    max_frames: Optional[int] = None
    sam_reset_interval: int = 20
    frame_by_frame: bool = False

    # SEA-RAFT
    raft_model: str = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"
    raft_iters: int = 12
    raft_device: str = "cuda"
    resize_scale: float = 0.5

    # Segmentation & Fusion
    smooth_window: int = 5
    min_distance: float = 0.2
    prominence: float = 0.1
    margin: int = 2
    angle_tolerance: float = 40.0
    min_segment_len: int = 8
    min_speed: float = 0.5
    
    # Weighted Multi-Modal Scoring
    fusion_threshold: float = 0.40
    w_speed: float = 0.25
    w_dir_left: float = 0.18
    w_dir_right: float = 0.18
    w_both_hands: float = 0.10
    w_idle: float = 0.20
    w_accel: float = 0.09
    require_both: Optional[bool] = None

    # Classification
    classify_min_frames: int = 5
    classify_max_frames: int = 15
    no_classify: bool = False
    vlm_model: str = "qwen/qwen3.7-flash"

    # Visualization
    no_video: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PipelineConfig":
        return cls(
            video=Path(args.video),
            output=Path(args.output),
            force=args.force,
            only_step12=args.only_step12,
            recompute_segmentation=args.recompute_segmentation,
            mask=Path(args.mask) if getattr(args, "mask", None) else None,
            sam_threshold=args.sam_threshold,
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            sam_reset_interval=args.sam_reset_interval,
            frame_by_frame=args.frame_by_frame,
            raft_model=args.raft_model,
            raft_iters=args.raft_iters,
            raft_device=args.raft_device,
            resize_scale=args.resize_scale,
            smooth_window=args.smooth_window,
            min_distance=args.min_distance,
            prominence=args.prominence,
            margin=args.margin,
            angle_tolerance=args.angle_tolerance,
            min_segment_len=args.min_segment_len,
            min_speed=args.min_speed,
            fusion_threshold=args.fusion_threshold,
            w_speed=args.w_speed,
            w_dir_left=args.w_dir_left,
            w_dir_right=args.w_dir_right,
            w_both_hands=args.w_both_hands,
            w_idle=args.w_idle,
            w_accel=args.w_accel,
            require_both=args.require_both,
            classify_min_frames=args.classify_min_frames,
            classify_max_frames=args.classify_max_frames,
            no_classify=args.no_classify,
            vlm_model=args.vlm_model,
            no_video=args.no_video,
        )
