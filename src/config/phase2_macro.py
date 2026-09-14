"""Phase 2 (macro evaluation) configuration."""
from __future__ import annotations

from src.config.common import DATA_DIR

# --- paths -------------------------------------------------------------------
DEFAULT_CD_DIR = DATA_DIR / "1" if (DATA_DIR / "1").is_dir() else DATA_DIR
MACRO_EVAL_PATH = DEFAULT_CD_DIR / "macro_eval.json"  # output (its slow_segments feed Phase 4)

# --- tunables ----------------------------------------------------------------
TIMING_SLOW_RATIO = 1.5   # worker/expected above this -> "slow"
TIMING_FAST_RATIO = 0.5   # below this -> "fast"
