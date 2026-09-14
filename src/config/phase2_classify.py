"""Phase 2 (segment classification) configuration.

Classifies the action segments Phase 1 produced (kinematic boundaries) against
the expert manifest, scene by scene, via the VLM.
"""
from __future__ import annotations

from pathlib import Path
from src.config.common import DATA_DIR

# --- model -------------------------------------------------------------------
MODEL = "qwen/qwen3.7-flash"

# --- paths -------------------------------------------------------------------
DEFAULT_CD_DIR = DATA_DIR / "1" if (DATA_DIR / "1").is_dir() else DATA_DIR
WORKER_VIDEO = DEFAULT_CD_DIR / "worker.mp4"                    # input: the video being evaluated
WORKER_FRAMES_DIR = DEFAULT_CD_DIR / "worker_frames"            # frames sampled on demand (also read by micro eval)
OUT_DIR = DEFAULT_CD_DIR / "worker_segments"
SEGMENTS_PATH = OUT_DIR / "worker_segments.json"          # output, read by macro eval
CUTS_DIR = OUT_DIR / "cuts"                               # one clip per segment, only with --cut
TIMELINE_DEBUG_PATH = OUT_DIR / "timeline_debug.json"     # --visualize output

def get_classify_paths(cd: str | int | Path = "1") -> dict[str, Path]:
    cd_dir = DATA_DIR / str(cd) if not isinstance(cd, Path) else cd
    out_dir = cd_dir / "worker_segments"
    return {
        "cd_dir": cd_dir,
        "worker_video": cd_dir / "worker.mp4",
        "worker_frames_dir": cd_dir / "worker_frames",
        "out_dir": out_dir,
        "segments_path": out_dir / "worker_segments.json",
        "cuts_dir": out_dir / "cuts",
        "timeline_debug_path": out_dir / "timeline_debug.json",
    }

# --- frame sampling ----------------------------------------------------------
MIN_STEP_FPS = 2
MAX_STEP_FPS = 8.0
