"""Phase 2 (macro evaluation) configuration."""
from __future__ import annotations

from src.config.common import DATA_DIR

# --- paths -------------------------------------------------------------------
MACRO_EVAL_PATH = DATA_DIR / "macro_eval.json"  # output (its slow_segments feed Phase 4)

# --- tunables ----------------------------------------------------------------
TIMING_SLOW_RATIO = 1.5   # worker/expected above this -> "slow"
TIMING_FAST_RATIO = 0.5   # below this -> "fast"
