#!/usr/bin/env python3
"""Run Step 1.1 (auto_select_frames_from_kinematic) and Step 1.2 (build_selection_manifest)
for Expert video without calling VLM.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.expert_analysis import auto_select_frames_from_kinematic, build_selection_manifest


def run(cd: str = "1", force_kinematic: bool = False) -> None:
    cd_dir = Path(f"data/{cd}")
    expert_json = cd_dir / "expert.json"
    frames_dir = cd_dir / "expert_scenes" / "frames"
    manifest_path = cd_dir / "expert_scenes" / "selected_frames.json"

    print("=" * 60)
    print(f"[STAGE 2 - STEP 1.1] Auto-selecting frames from kinematic for CĐ {cd}...")
    print("=" * 60)
    auto_select_frames_from_kinematic(
        expert_json_path=expert_json,
        frames_dir=frames_dir,
        force_kinematic=force_kinematic,
    )

    print("\n" + "=" * 60)
    print(f"[STAGE 2 - STEP 1.2] Building selection manifest (extracting frames)...")
    print("=" * 60)
    manifest = build_selection_manifest(
        expert_json_path=expert_json,
        frames_dir=frames_dir,
        out_path=manifest_path,
    )
    print(f"\n[OK] Manifest updated with {len(manifest.get('scenes', {}))} scenes -> {manifest_path}")


if __name__ == "__main__":
    cd = sys.argv[1] if len(sys.argv) > 1 else "1"
    run(cd)
