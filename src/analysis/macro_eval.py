"""Phase 2 (macro step): macro evaluation — overall comparison of the worker run against
the expert standard.

Summarize the worker's output segments against the manifest, producing a report of
- missing operations (scenes in the manifest that were never reached)
- extra/undetermined segments (worker output segments that don't match any manifest scene)
- timing deviations (worker segments that are significantly faster or slower than the expert standard)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.config.phase2_macro import TIMING_FAST_RATIO, TIMING_SLOW_RATIO
from src.manifest import expected_duration, ordered_scene_items, scene_op_name


@dataclass
class MacroSummary:
    task_name: str
    n_scenes: int
    evaluated: list[dict] = field(default_factory=list)
    missing: list[dict] = field(default_factory=list)
    extra: list[dict] = field(default_factory=list)
    slow_segments: list[dict] = field(default_factory=list)
    fast_segments: list[dict] = field(default_factory=list)


def _timing_verdict(actual: float, expected: float) -> tuple[float | None, str]:
    """Compare the actual duration of a worker segment against the expected duration."""
    if expected <= 0:
        return None, "unknown"
    ratio = actual / expected
    if ratio > TIMING_SLOW_RATIO:
        verdict = "slow"
    elif ratio < TIMING_FAST_RATIO:
        verdict = "fast"
    else:
        verdict = "ok"
    return round(ratio, 2), verdict


def summarize(result: dict, manifest: dict) -> MacroSummary:
    """Summarize the worker's output segments against the manifest.
    """
    segments = result["segments"]
    ordered_scenes = ordered_scene_items(manifest)

    extra = [s for s in segments if s["operation_name"] == "UNKNOWN"]
    named_segments = [s for s in segments if s["operation_name"] != "UNKNOWN"]

    remaining_by_name: dict[str, list[dict]] = {}
    for sid, sd in ordered_scenes:
        remaining_by_name.setdefault(scene_op_name(sd), []).append(sd)

    evaluated = []
    for seg in named_segments:
        candidates = remaining_by_name.get(seg["operation_name"])
        scene = candidates.pop(0) if candidates else None
        dur = seg["end_time"] - seg["start_time"]
        if scene is not None:
            expected = expected_duration(scene)
            ratio, verdict = _timing_verdict(dur, expected)
            seg = {**seg, "expert_duration_s": round(expected, 2),
                  "duration_ratio": ratio, "timing_verdict": verdict}
        else:
            seg = {**seg, "expert_duration_s": None, "duration_ratio": None, "timing_verdict": "unknown"}
        evaluated.append(seg)

    # scenes never matched to any output segment -> missing
    missing = [{"operation_name": name, "note": "worker did not perform this / video ended before reaching it"}
              for name, remaining in remaining_by_name.items() for _ in remaining]

    slow_segments = [s for s in evaluated if s["timing_verdict"] == "slow"]
    fast_segments = [s for s in evaluated if s["timing_verdict"] == "fast"]

    return MacroSummary(
        task_name=result.get("task_name", manifest.get("task_name", "")),
        n_scenes=len(ordered_scenes),
        evaluated=evaluated, missing=missing, extra=extra,
        slow_segments=slow_segments, fast_segments=fast_segments,
    )


def print_report(summary: MacroSummary) -> None:
    print(f"Task: {summary.task_name}")
    print(f"{len(summary.evaluated)} evaluated | {len(summary.missing)} missing | "
          f"{len(summary.extra)} extra/undetermined segment(s)\n")

    for s in summary.evaluated:
        dur = s["end_time"] - s["start_time"]
        tag = ""
        if s["timing_verdict"] == "slow":
            tag = " ⚠ SLOWER THAN STANDARD"
        elif s["timing_verdict"] == "fast":
            tag = " ⚡ FASTER THAN STANDARD"
        print(f"[{s['start_time']:>6.1f}s - {s['end_time']:>6.1f}s] {s['operation_name']}")
        if s["expert_duration_s"] is not None:
            print(f"    worker {dur:.1f}s vs standard {s['expert_duration_s']}s "
                  f"(x{s['duration_ratio']}){tag}")
        else:
            print(f"    worker {dur:.1f}s (no matching standard scene found for time comparison)")
        if s["off_standard"]:
            print(f"    ⚠ off-standard technique: {s['off_standard_desc']}")
        print()

    print("=" * 70)
    print("OVERVIEW (MACRO)")
    print("=" * 70)

    print(f"\n1. MISSING operations ({len(summary.missing)}):")
    if not summary.missing:
        print("   (none — all standard scenes were evaluated)")
    for s in summary.missing:
        print(f"   - {s['operation_name']}: {s.get('note', '')}")

    print(f"\n2. EXTRA / undetermined segments ({len(summary.extra)}):")
    if not summary.extra:
        print("   (none)")
    for s in summary.extra:
        print(f"   - [{s['start_time']}s-{s['end_time']}s] {s.get('off_standard_desc', '')}")

    print(f"\n3. Operations SLOWER than standard ({len(summary.slow_segments)}):")
    for s in summary.slow_segments:
        dur = s["end_time"] - s["start_time"]
        print(f"   - {s['operation_name']}: worker {dur:.1f}s vs standard "
              f"{s['expert_duration_s']}s (x{s['duration_ratio']})")

    print(f"\n4. Operations FASTER than standard ({len(summary.fast_segments)}):")
    for s in summary.fast_segments:
        dur = s["end_time"] - s["start_time"]
        print(f"   - {s['operation_name']}: worker {dur:.1f}s vs standard "
              f"{s['expert_duration_s']}s (x{s['duration_ratio']}) — review whether small steps were skipped")

    print("\n" + "=" * 70)
    print("VLM USAGE NOTES")
    print("=" * 70)
    print(f"""
- Missing/extra/timing deviations: NO additional VLM needed — timing is compared in code at this phase
  (v3 only flags off_standard on technique when closing a segment, does not compare timing itself).
- For operations SLOWER than standard ({len(summary.slow_segments)} operations): VLM recommended (Phase 4,
  micro_eval) — frame-by-frame comparison of expert vs. worker gestures/movements is needed to pinpoint
  which small sub-step the worker is slow on; the current evidence field is only a 1-sentence summary,
  insufficient for targeted retraining.
""")


def save(summary: MacroSummary, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_name": summary.task_name,
        "n_scenes": summary.n_scenes,
        "n_evaluated": len(summary.evaluated),
        "n_missing": len(summary.missing),
        "n_extra": len(summary.extra),
        "n_slow": len(summary.slow_segments),
        "n_fast": len(summary.fast_segments),
        "evaluated": summary.evaluated,
        "missing": summary.missing,
        "extra": summary.extra,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved macro summary to {out_path}")
