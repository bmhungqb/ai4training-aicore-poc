"""Phase 2 (expert analysis) configuration."""
from __future__ import annotations

from pathlib import Path
from src.config.common import DATA_DIR

# --- model -------------------------------------------------------------------
VLM_MODEL = "qwen/qwen3.7-flash"

# --- paths -------------------------------------------------------------------
DEFAULT_CD_DIR = DATA_DIR / "1" if (DATA_DIR / "1" / "expert.json").is_file() else DATA_DIR
EXPERT_VIDEO = DEFAULT_CD_DIR / "expert.mp4"
EXPERT_JSON = DEFAULT_CD_DIR / "expert.json"                        # input: scene boundaries (ground truth,
                                                              # still authored by hand) + operation names
EXPERT_SCENES_DIR = DEFAULT_CD_DIR / "expert_scenes"                # output root for this phase
FRAMES_DIR = EXPERT_SCENES_DIR / "frames"                     # scene_NN/ reference frames
MANIFEST_PATH = EXPERT_SCENES_DIR / "selected_frames.json"    # output, read by Phases 2/3/4
PROCESS_KNOWLEDGE_PATH = EXPERT_SCENES_DIR / "process_knowledge.json"

def get_expert_paths(cd: str | int | Path = "1") -> dict[str, Path]:
    cd_dir = DATA_DIR / str(cd) if not isinstance(cd, Path) else cd
    scenes_dir = cd_dir / "expert_scenes"
    return {
        "cd_dir": cd_dir,
        "expert_video": cd_dir / "expert.mp4",
        "expert_json": cd_dir / "expert.json",
        "expert_scenes_dir": scenes_dir,
        "frames_dir": scenes_dir / "frames",
        "manifest_path": scenes_dir / "selected_frames.json",
        "process_knowledge_path": scenes_dir / "process_knowledge.json",
    }

# --- reference-frame selection (auto, kinematic-driven) ----------------------
# Reference frames are now picked automatically: kinematic action segmentation runs on
# expert.mp4 (same engine as Phase 1 for worker.mp4), each action segment is assigned to
# the scene whose [timestamp_start, timestamp_end] contains its midpoint, and
# FRAMES_PER_ACTION_SEGMENT frames are sampled per action segment (uniformly over time,
# picking the sharpest frame in a small neighborhood around each sample point — see
# pick_sharpest_spread() in src/utils/frames.py). No manual frame curation.
EXPERT_KINEMATIC_OUT_DIR = DEFAULT_CD_DIR / "kinematic_expert"
EXPERT_ACTION_SEGMENTS_PATH = EXPERT_KINEMATIC_OUT_DIR / "action_segments.json"
FRAMES_PER_ACTION_SEGMENT = 2   # frames sampled per action segment (>=1)
SHARPNESS_POOL_FACTOR = 3       # width (in step-multiples) of the sharpest-frame search window
                                # around each uniform sample point; higher = more blur-dodging
                                # but less even spacing

# --- tunables ----------------------------------------------------------------
MAX_REF_FRAMES_PER_SCENE = 3  # frames sent to the synthesis step (guideline already distills the rest)
