"""Settings shared by more than one phase.

The crop boxes/widths live here rather than in a single phase's config because
they describe the two source VIDEOS, not one phase: Phase 1 encodes expert
frames, Phase 2 worker frames, Phase 4 both.
"""
from __future__ import annotations

from pathlib import Path

# Root of every input/output artifact (expert.json, worker.mp4, *_eval.json, ...).
DATA_DIR = Path("data")

# Fixed crop of the work area: None by default.
# If a mask (*.mask.png) exists from Stage 1 (or is passed explicitly), Stage 2 automatically
# applies the mask and crops to its bounding box.
EXPERT_CROP_BOX = None
EXPERT_FRAME_WIDTH = 640

WORKER_CROP_BOX = None
WORKER_FRAME_WIDTH = 640

