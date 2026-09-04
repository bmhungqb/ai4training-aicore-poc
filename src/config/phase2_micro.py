"""Phase 2 (micro evaluation) configuration."""
from __future__ import annotations

from src.config.common import DATA_DIR
from src.config.phase2_classify import WORKER_FRAMES_DIR  # noqa: F401  (re-exported: micro re-reads
                                                          # the frames the classify step sampled)

# --- model -------------------------------------------------------------------
MODEL = "qwen/qwen3.7-plus"

# --- paths -------------------------------------------------------------------
MICRO_EVAL_PATH = DATA_DIR / "micro_eval.json"  # output

# --- tunables ----------------------------------------------------------------
MAX_EXPERT_FRAMES = 8    # expert reference frames sent per slow segment
MAX_WORKER_FRAMES = 8    # worker frames sent per slow segment
WINDOW_PAD_SEC = 0.2     # a worker frame this far outside [start_time, end_time] still counts as
                         # in-window (frames are sampled at discrete timestamps)
