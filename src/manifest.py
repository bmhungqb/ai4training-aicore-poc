"""Helpers for the Phase 1 manifest (selected_frames.json) shared by every
later phase: scene ordering, the scene's display/matching name, and the small
guideline-formatting pieces that more than one phase prints."""
from __future__ import annotations


def scene_op_name(scene_data: dict) -> str:
    """Display/matching name of a scene: its operation names joined in order.
    Phases 2/3/4 all match segments back to scenes by this exact string."""
    return ", ".join(op["name"] for op in scene_data["operations"])


def ordered_scene_items(manifest: dict) -> list[tuple[str, dict]]:
    """`manifest["scenes"]` items sorted by numeric scene id (JSON keys are
    strings, so a plain sort would put "10" before "2")."""
    return sorted(manifest["scenes"].items(), key=lambda kv: int(kv[0]))


def expected_duration(scene_data: dict) -> float:
    """How long the expert took for this scene."""
    return scene_data["timestamp_end"] - scene_data["timestamp_start"]


def format_product_state(guideline: dict) -> str:
    """One-line before/during/after product-state summary from a scene guideline."""
    tp = guideline.get("product_state", {})
    return (f"- Product state: before=\"{tp.get('state_before', '')}\"; "
            f"during=\"{tp.get('state_during', '')}\"; after=\"{tp.get('state_after', '')}\"")
