"""Phase 2 (segment classification) configuration.

Classifies the action segments Phase 1 produced (kinematic boundaries) against
the expert manifest, scene by scene, via the VLM.
"""
from __future__ import annotations

from src.config.common import DATA_DIR

# --- model -------------------------------------------------------------------
MODEL = "qwen/qwen3.7-plus"

# --- paths -------------------------------------------------------------------
WORKER_VIDEO = DATA_DIR / "worker.mp4"                    # input: the video being evaluated
WORKER_FRAMES_DIR = DATA_DIR / "worker_frames"            # frames sampled on demand (also read by micro eval)
OUT_DIR = DATA_DIR / "worker_segments"
SEGMENTS_PATH = OUT_DIR / "worker_segments.json"          # output, read by macro eval
CUTS_DIR = OUT_DIR / "cuts"                               # one clip per segment, only with --cut
TIMELINE_DEBUG_PATH = OUT_DIR / "timeline_debug.json"     # --visualize output

# --- frame sampling ----------------------------------------------------------
MIN_STEP_FPS = 2
MAX_STEP_FPS = 8.0
