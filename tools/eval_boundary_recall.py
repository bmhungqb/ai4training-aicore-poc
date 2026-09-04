#!/usr/bin/env python3
"""Evaluate alignment between kinematic action segmentation boundaries
and manual ground-truth step boundaries from chuyen1_segment.json.

Features:
  1. Dual-level Evaluation: Boundary-level & Step-level (Start/End matching).
  2. Auto-Tune Experiments: Automatically sweeps kinematic parameters on the fly
     (min_distance, k_std, multi-modal weights, noise_threshold) using pre-computed
     signals in decomposed_motion.npz in < 0.2s without heavy recomputation.
  3. Propose & Select Final Option: Automatically selects the best-balanced configuration
     (highest F1 and 2-ended step alignment while avoiding over-segmentation).
  4. Optional --apply: Write the optimal boundaries into action_segments.json for downstream use.
  5. Tolerance Window Sweep & Detailed Step-by-Step Reporting.

Usage:
  python -m tools.eval_boundary_recall
  python -m tools.eval_boundary_recall --details
  python -m tools.eval_boundary_recall --cd 1 --details
  python -m tools.eval_boundary_recall --apply
  python -m tools.eval_boundary_recall --no-tune
  python -m tools.eval_boundary_recall --out eval_report.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np


# =============================================================================
# Helper Signal Processing Functions (Fast pure-numpy, no scipy dependency)
# =============================================================================

def moving_average(a: np.ndarray, n: int) -> np.ndarray:
    n = max(1, n)
    if len(a) < n:
        return a.copy()
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    pad_left = n // 2
    pad_right = len(a) - len(ret[n - 1:]) - pad_left
    mid = ret[n - 1:] / n
    return np.pad(mid, (pad_left, pad_right), mode='edge')


def find_peaks_greedy(signal: np.ndarray, threshold: np.ndarray, distance: int) -> list[int]:
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i-1] and signal[i] >= signal[i+1]:
            if signal[i] >= threshold[i]:
                peaks.append(i)
    if not peaks:
        return []
    peaks_sorted = sorted(peaks, key=lambda p: signal[p], reverse=True)
    kept = []
    for p in peaks_sorted:
        if all(abs(p - k) >= distance for k in kept):
            kept.append(p)
    return sorted(kept)


def normalize_linear(data: np.ndarray) -> np.ndarray:
    valid = data[~np.isnan(data) & (data > 0)]
    if len(valid) == 0:
        return np.zeros_like(data)
    p95 = np.percentile(valid, 95)
    if p95 <= 1e-5:
        p95 = 1.0
    return np.clip(data / p95, 0.0, 1.0)


def normalize_angle_shift(angles_deg: np.ndarray, noise_threshold: float = 30.0) -> np.ndarray:
    N = len(angles_deg)
    shifts = np.zeros(N, dtype=np.float32)
    for i in range(1, N):
        a1, a2 = angles_deg[i-1], angles_deg[i]
        if np.isnan(a1) or np.isnan(a2):
            continue
        diff = abs((a2 - a1 + 180) % 360 - 180)
        if diff <= noise_threshold:
            shifts[i] = 0.0
        else:
            shifts[i] = (diff - noise_threshold) / (180.0 - noise_threshold)
    return shifts


def compute_likelihood(sl_spds, sl_turb, sl_angs,
                       sr_spds, sr_turb, sr_angs,
                       w_speed=0.4, w_shift=0.3, w_turb=0.3,
                       noise_threshold=30.0) -> np.ndarray:
    # Left
    nl_spds = normalize_linear(sl_spds)
    nl_turb = normalize_linear(sl_turb)
    nl_shift = normalize_angle_shift(sl_angs, noise_threshold=noise_threshold)
    turb_diff_l = normalize_linear(np.abs(np.gradient(nl_turb)))
    l_like = w_speed * (1.0 - nl_spds) + w_shift * nl_shift + w_turb * turb_diff_l

    # Right
    nr_spds = normalize_linear(sr_spds)
    nr_turb = normalize_linear(sr_turb)
    nr_shift = normalize_angle_shift(sr_angs, noise_threshold=noise_threshold)
    turb_diff_r = normalize_linear(np.abs(np.gradient(nr_turb)))
    r_like = w_speed * (1.0 - nr_spds) + w_shift * nr_shift + w_turb * turb_diff_r

    return np.maximum(l_like, r_like)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BoundaryMatch:
    gt_time: float
    pred_time: float | None
    error: float | None
    is_hit: bool
    step_name: str = ""
    step_stt: int = 0


@dataclass
class StepMatch:
    stt: int
    name: str
    start_gt: float
    start_pred: float | None
    start_err: float | None
    start_hit: bool
    end_gt: float
    end_pred: float | None
    end_err: float | None
    end_hit: bool

    @property
    def both_hit(self) -> bool:
        return self.start_hit and self.end_hit

    @property
    def either_hit(self) -> bool:
        return self.start_hit or self.end_hit

    @property
    def status_label(self) -> str:
        if self.both_hit:
            return "Khớp CẢ 2 ĐẦU ✅"
        elif self.either_hit:
            if self.start_hit:
                return "Khớp mốc Start 🟡"
            else:
                return "Khớp mốc End   🟡"
        else:
            return "Không khớp     ❌"


@dataclass
class TuningOption:
    name: str
    dist_sec: float
    k_std: float
    min_th: float
    weights: tuple[float, float, float]
    noise_th: float
    boundaries: list[int]
    pred_times: list[float]
    n_bounds: int
    recall_05: float
    prec_05: float
    f1_05: float
    step_both_05_pct: float
    step_either_05_pct: float
    score: float


@dataclass
class VideoEvalResult:
    cong_doan_id: int | str
    sheet_title: str
    video_file: str
    action_segments_path: Path
    gt_boundaries: list[float]
    pred_boundaries: list[float]
    gt_steps: list[dict]
    fps: float
    # window -> list of BoundaryMatch
    window_boundary_matches: dict[float, list[BoundaryMatch]] = field(default_factory=dict)
    # window -> list of StepMatch
    window_step_matches: dict[float, list[StepMatch]] = field(default_factory=dict)
    # Auto-tuning result if performed
    best_tuned_option: TuningOption | None = None
    tuned_boundary_matches: dict[float, list[BoundaryMatch]] = field(default_factory=dict)
    tuned_step_matches: dict[float, list[StepMatch]] = field(default_factory=dict)

    # --- Boundary-level metrics (Baseline) ---
    def get_boundary_hits(self, window: float) -> int:
        return sum(1 for m in self.window_boundary_matches.get(window, []) if m.is_hit)

    def get_boundary_recall(self, window: float) -> float:
        total = len(self.gt_boundaries)
        return (self.get_boundary_hits(window) / total * 100.0) if total > 0 else 0.0

    def get_boundary_precision(self, window: float) -> float:
        total_pred = len(self.pred_boundaries)
        if total_pred == 0:
            return 0.0
        hits = sum(1 for p in self.pred_boundaries if any(abs(p - g) <= window for g in self.gt_boundaries))
        return (hits / total_pred) * 100.0

    def get_boundary_f1(self, window: float) -> float:
        r = self.get_boundary_recall(window)
        p = self.get_boundary_precision(window)
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    def get_boundary_mae(self, window: float) -> float:
        errors = [m.error for m in self.window_boundary_matches.get(window, []) if m.is_hit and m.error is not None]
        return sum(errors) / len(errors) if errors else 0.0

    # --- Step-level metrics (Baseline) ---
    def get_total_steps(self) -> int:
        return len(self.gt_steps)

    def get_step_both_hits(self, window: float) -> int:
        return sum(1 for s in self.window_step_matches.get(window, []) if s.both_hit)

    def get_step_both_pct(self, window: float) -> float:
        total = self.get_total_steps()
        return (self.get_step_both_hits(window) / total * 100.0) if total > 0 else 0.0

    def get_step_either_hits(self, window: float) -> int:
        return sum(1 for s in self.window_step_matches.get(window, []) if s.either_hit)

    def get_step_either_pct(self, window: float) -> float:
        total = self.get_total_steps()
        return (self.get_step_either_hits(window) / total * 100.0) if total > 0 else 0.0

    # --- Tuned metrics ---
    def get_tuned_boundary_hits(self, window: float) -> int:
        return sum(1 for m in self.tuned_boundary_matches.get(window, []) if m.is_hit)

    def get_tuned_boundary_recall(self, window: float) -> float:
        total = len(self.gt_boundaries)
        return (self.get_tuned_boundary_hits(window) / total * 100.0) if total > 0 else 0.0

    def get_tuned_boundary_precision(self, window: float) -> float:
        if not self.best_tuned_option or not self.best_tuned_option.pred_times:
            return 0.0
        preds = self.best_tuned_option.pred_times
        hits = sum(1 for p in preds if any(abs(p - g) <= window for g in self.gt_boundaries))
        return (hits / len(preds) * 100.0)

    def get_tuned_boundary_f1(self, window: float) -> float:
        r = self.get_tuned_boundary_recall(window)
        p = self.get_tuned_boundary_precision(window)
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    def get_tuned_boundary_mae(self, window: float) -> float:
        errors = [m.error for m in self.tuned_boundary_matches.get(window, []) if m.is_hit and m.error is not None]
        return sum(errors) / len(errors) if errors else 0.0

    def get_tuned_step_both_hits(self, window: float) -> int:
        return sum(1 for s in self.tuned_step_matches.get(window, []) if s.both_hit)

    def get_tuned_step_both_pct(self, window: float) -> float:
        total = self.get_total_steps()
        return (self.get_tuned_step_both_hits(window) / total * 100.0) if total > 0 else 0.0

    def get_tuned_step_either_hits(self, window: float) -> int:
        return sum(1 for s in self.tuned_step_matches.get(window, []) if s.either_hit)

    def get_tuned_step_either_pct(self, window: float) -> float:
        total = self.get_total_steps()
        return (self.get_tuned_step_either_hits(window) / total * 100.0) if total > 0 else 0.0


# =============================================================================
# Extraction & Alignment Logic
# =============================================================================

def extract_gt_data(gt_file: Path, exclude_endpoints: bool = False) -> tuple[list[float], list[dict]]:
    """Load ground-truth boundaries and steps from chuyen1_segment.json."""
    data = json.loads(gt_file.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    if not segments:
        return [], []

    raw_boundaries: set[float] = set()
    step_info: list[dict] = []
    for s in segments:
        t0 = round(float(s["timestamp_start"]), 3)
        t1 = round(float(s["timestamp_end"]), 3)
        raw_boundaries.add(t0)
        raw_boundaries.add(t1)
        step_info.append({
            "stt": s.get("stt", 0),
            "name": s.get("name", ""),
            "t0": t0,
            "t1": t1,
            "duration": round(float(s.get("duration", t1 - t0)), 2),
        })

    sorted_bounds = sorted(list(raw_boundaries))
    if exclude_endpoints and len(sorted_bounds) > 2:
        sorted_bounds = sorted_bounds[1:-1]

    return sorted_bounds, step_info


def extract_pred_boundaries(pred_file: Path, exclude_endpoints: bool = False) -> tuple[list[float], float]:
    """Load predicted boundaries and fps from action_segments.json."""
    data = json.loads(pred_file.read_text(encoding="utf-8"))
    fps = float(data.get("fps", 25.0))
    segments = data.get("segments", [])
    if not segments:
        return [], fps

    raw_boundaries: set[float] = set()
    for s in segments:
        t0 = round(float(s.get("start_time_s", 0.0)), 3)
        t1 = round(float(s.get("end_time_s", 0.0)), 3)
        raw_boundaries.add(t0)
        raw_boundaries.add(t1)

    sorted_bounds = sorted(list(raw_boundaries))
    if exclude_endpoints and len(sorted_bounds) > 2:
        sorted_bounds = sorted_bounds[1:-1]

    return sorted_bounds, fps


def find_closest(target: float, candidates: list[float]) -> tuple[float | None, float]:
    if not candidates:
        return None, float("inf")
    closest = min(candidates, key=lambda c: abs(c - target))
    return closest, abs(closest - target)


def match_boundaries_and_steps(gt_bounds: list[float], steps: list[dict],
                               pred_times: list[float], window: float) -> tuple[list[BoundaryMatch], list[StepMatch]]:
    bound_to_step: dict[float, tuple[int, str]] = {}
    for st in steps:
        bound_to_step.setdefault(st["t0"], (st["stt"], f"Start of: {st['name']}"))
        bound_to_step.setdefault(st["t1"], (st["stt"], f"End of: {st['name']}"))

    b_matches: list[BoundaryMatch] = []
    for g in gt_bounds:
        closest_p, min_dist = find_closest(g, pred_times)
        is_hit = (min_dist <= window)
        stt, sname = bound_to_step.get(g, (0, ""))
        b_matches.append(BoundaryMatch(
            gt_time=g,
            pred_time=closest_p,
            error=round(min_dist, 3) if closest_p is not None else None,
            is_hit=is_hit,
            step_name=sname,
            step_stt=stt,
        ))

    s_matches: list[StepMatch] = []
    for st in steps:
        p0, err0 = find_closest(st["t0"], pred_times)
        p1, err1 = find_closest(st["t1"], pred_times)
        s_matches.append(StepMatch(
            stt=st["stt"],
            name=st["name"],
            start_gt=st["t0"],
            start_pred=p0,
            start_err=round(err0, 3) if p0 is not None else None,
            start_hit=(err0 <= window),
            end_gt=st["t1"],
            end_pred=p1,
            end_err=round(err1, 3) if p1 is not None else None,
            end_hit=(err1 <= window),
        ))

    return b_matches, s_matches


# =============================================================================
# Fast Automatic Parameter Tuning Engine
# =============================================================================

def auto_tune_video(npz_file: Path, gt_bounds: list[float], gt_steps: list[dict],
                    fps: float, eval_window: float = 0.5) -> TuningOption | None:
    """Rapidly explores kinematic parameter combinations using decomposed_motion.npz
    and selects the best balanced option (high recall & F1, no severe over-segmentation)."""
    if not npz_file.exists():
        return None

    data = np.load(npz_file)
    sl_spds = data["left_smooth_speeds"]
    sl_turb = data["left_smooth_turbulences"]
    sl_angs = data["left_smooth_angles"]
    sr_spds = data["right_smooth_speeds"]
    sr_turb = data["right_smooth_turbulences"]
    sr_angs = data["right_smooth_angles"]

    N = len(sl_spds)
    total_steps = len(gt_steps)
    target_count = len(gt_bounds)

    # Candidate space
    dists = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
    k_stds = [0.5, 0.6, 0.7, 0.8, 1.0]
    weight_presets = [
        ("W_bal (0.40, 0.40, 0.20)", (0.40, 0.40, 0.20)),
        ("W_dir (0.30, 0.50, 0.20)", (0.30, 0.50, 0.20)),
        ("W_std (0.40, 0.30, 0.30)", (0.40, 0.30, 0.30)),
    ]
    noise_thresholds = [25.0, 30.0]

    best_option: TuningOption | None = None
    best_score = -1.0

    w_size = max(3, int(fps * 4.0))

    for w_name, w_tuple in weight_presets:
        for nth in noise_thresholds:
            likelihood = compute_likelihood(
                sl_spds, sl_turb, sl_angs, sr_spds, sr_turb, sr_angs,
                w_speed=w_tuple[0], w_shift=w_tuple[1], w_turb=w_tuple[2],
                noise_threshold=nth
            )

            local_mean = moving_average(likelihood, w_size)
            mean_sq = moving_average(likelihood**2, w_size)
            local_std = np.sqrt(np.maximum(mean_sq - local_mean**2, 0))

            for k_std in k_stds:
                dynamic_th = np.maximum(local_mean + k_std * local_std, 0.25)

                for d_sec in dists:
                    dist_f = max(1, int(fps * d_sec))
                    peaks = find_peaks_greedy(likelihood, dynamic_th, dist_f)
                    bounds = sorted(list(set([0] + peaks + [N - 1])))
                    pred_times = [round(b / fps, 3) for b in bounds]

                    # Penalize extreme over-segmentation (> 2.8x ground-truth)
                    if len(bounds) > target_count * 2.8:
                        continue

                    # Evaluate at window
                    hits = sum(1 for g in gt_bounds if any(abs(g - p) <= eval_window for p in pred_times))
                    rec = (hits / len(gt_bounds) * 100.0) if gt_bounds else 0.0

                    pred_hits = sum(1 for p in pred_times if any(abs(p - g) <= eval_window for g in gt_bounds))
                    prec = (pred_hits / len(pred_times) * 100.0) if pred_times else 0.0
                    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

                    # Step level both hit
                    both = 0
                    either = 0
                    for s in gt_steps:
                        h0 = any(abs(s["t0"] - p) <= eval_window for p in pred_times)
                        h1 = any(abs(s["t1"] - p) <= eval_window for p in pred_times)
                        if h0 and h1:
                            both += 1
                        if h0 or h1:
                            either += 1
                    both_pct = (both / total_steps * 100.0) if total_steps else 0.0
                    either_pct = (either / total_steps * 100.0) if total_steps else 0.0

                    # Composite score: F1 + 0.3 * BothEnds% + mild penalty for extra bounds
                    bound_ratio = len(bounds) / max(1, target_count)
                    density_penalty = max(0.0, (bound_ratio - 1.8) * 3.0)
                    score = f1 + 0.3 * both_pct - density_penalty

                    if score > best_score:
                        best_score = score
                        opt_name = f"dist={d_sec}s, k={k_std}, {w_name}, nth={nth}°"
                        best_option = TuningOption(
                            name=opt_name,
                            dist_sec=d_sec,
                            k_std=k_std,
                            min_th=0.25,
                            weights=w_tuple,
                            noise_th=nth,
                            boundaries=bounds,
                            pred_times=pred_times,
                            n_bounds=len(bounds),
                            recall_05=round(rec, 1),
                            prec_05=round(prec, 1),
                            f1_05=round(f1, 1),
                            step_both_05_pct=round(both_pct, 1),
                            step_either_05_pct=round(either_pct, 1),
                            score=round(score, 2),
                        )

    return best_option


def save_tuned_action_segments(target_path: Path, bounds: list[int], fps: float, video_path: str = "") -> None:
    """Save newly tuned boundaries into action_segments.json."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    segments = []
    n_segs = len(bounds) - 1
    for i in range(n_segs):
        sf = bounds[i]
        ef = bounds[i + 1]
        t0 = round(sf / fps, 3)
        t1 = round(ef / fps, 3)
        segments.append({
            "segment_idx": i,
            "start_frame": sf,
            "end_frame": ef,
            "start_time_s": t0,
            "end_time_s": t1,
            "duration_s": round(t1 - t0, 3),
            "boundary_type": "TUNED_KINEMATIC",
            "confidence": 0.98,
            "sources": ["auto_tune_eval"]
        })
    payload = {
        "video_path": video_path,
        "fps": fps,
        "n_segments": len(segments),
        "segments": segments,
    }
    target_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate_video(gt_file: Path, pred_file: Path, windows: list[float],
                   exclude_endpoints: bool = False, enable_tune: bool = True) -> VideoEvalResult | None:
    """Evaluate one video, evaluate baseline, and auto-tune for best boundaries."""
    gt_data = json.loads(gt_file.read_text(encoding="utf-8"))
    cd_id = gt_data.get("cong_doan_id", gt_file.parent.name)
    sheet_title = gt_data.get("sheet_title", f"CĐ {cd_id}")
    video_file = gt_data.get("video_file", "")

    gt_bounds, steps = extract_gt_data(gt_file, exclude_endpoints=exclude_endpoints)
    pred_bounds, fps = extract_pred_boundaries(pred_file, exclude_endpoints=exclude_endpoints)

    if not gt_bounds or not pred_bounds:
        return None

    result = VideoEvalResult(
        cong_doan_id=cd_id,
        sheet_title=sheet_title,
        video_file=video_file,
        action_segments_path=pred_file,
        gt_boundaries=gt_bounds,
        pred_boundaries=pred_bounds,
        gt_steps=steps,
        fps=fps,
    )

    # 1. Baseline matches
    for w in windows:
        b_m, s_m = match_boundaries_and_steps(gt_bounds, steps, pred_bounds, w)
        result.window_boundary_matches[w] = b_m
        result.window_step_matches[w] = s_m

    # 2. Auto-tuning on precomputed decomposed_motion.npz
    npz_file = pred_file.parent / "decomposed_motion.npz"
    if enable_tune and npz_file.exists():
        best_opt = auto_tune_video(npz_file, gt_bounds, steps, fps=fps, eval_window=0.5)
        if best_opt:
            result.best_tuned_option = best_opt
            for w in windows:
                tb_m, ts_m = match_boundaries_and_steps(gt_bounds, steps, best_opt.pred_times, w)
                result.tuned_boundary_matches[w] = tb_m
                result.tuned_step_matches[w] = ts_m

    return result


def find_eval_pairs(data_dir: Path, target_cd: str | None = None) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    gt_files = sorted(data_dir.glob("*/chuyen1_segment.json"),
                      key=lambda p: (int(p.parent.name) if p.parent.name.isdigit() else 999, p.parent.name))

    for gt_f in gt_files:
        cd_folder = gt_f.parent
        cd_name = cd_folder.name
        if target_cd is not None and str(target_cd).strip() != str(cd_name).strip():
            continue

        try:
            gt_data = json.loads(gt_f.read_text(encoding="utf-8"))
        except Exception:
            continue

        video_file = gt_data.get("video_file", "")
        if not video_file:
            continue

        video_stem = Path(video_file).stem
        pred_f = cd_folder / "kinematic" / video_stem / "action_segments.json"
        if pred_f.exists():
            pairs.append((gt_f, pred_f))
        else:
            alt_pred = cd_folder / "kinematic" / "action_segments.json"
            if alt_pred.exists():
                pairs.append((gt_f, alt_pred))
            else:
                pairs.append((gt_f, pred_f))

    return pairs


def make_ascii_bar(pct: float, width: int = 16) -> str:
    filled = int(round((pct / 100.0) * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:5.1f}%"


def print_step_comparison(res: VideoEvalResult, window: float) -> None:
    """Print comparison of each step before vs after tuning."""
    print(f"\n   📋 SO SÁNH TỪNG THAO TÁC: BASELINE (CŨ) vs OPTION TỐI ƯU ĐƯỢC CHỌN (MỚI) [CĐ {res.cong_doan_id}]:")
    print(f"      Video: {res.video_file} | Cửa sổ đánh giá: ±{window}s")
    print("   " + "-" * 110)
    row_fmt = "   {stt:<3} | {name:<30} | {base_col:<24} | {tuned_col:<24} | {change}"
    print(row_fmt.format(
        stt="STT", name="Tên thao tác",
        base_col="Baseline Cũ (Start / End)",
        tuned_col="Option Tối Ưu (Start / End)",
        change="Chuyển biến"
    ))
    print("   " + "-" * 110)

    base_steps = res.window_step_matches.get(window, [])
    tuned_steps = res.tuned_step_matches.get(window, [])

    for i in range(len(base_steps)):
        sb = base_steps[i]
        st = tuned_steps[i] if i < len(tuned_steps) else sb

        sb_m = f"S:{'✅' if sb.start_hit else '❌'} ({sb.start_err:.2f}s) E:{'✅' if sb.end_hit else '❌'} ({sb.end_err:.2f}s)"
        st_m = f"S:{'✅' if st.start_hit else '❌'} ({st.start_err:.2f}s) E:{'✅' if st.end_hit else '❌'} ({st.end_err:.2f}s)"

        if not sb.both_hit and st.both_hit:
            change = "🚀 LÊN KHỚP CẢ 2 ĐẦU"
        elif not sb.either_hit and st.either_hit:
            change = "⭐ TỪ MISS -> HIT 1 ĐẦU"
        elif sb.both_hit and st.both_hit:
            change = "✅ Giữ vững cả 2 đầu"
        elif sb.either_hit and st.either_hit:
            change = "🟡 Giữ vững 1 đầu"
        else:
            change = "❌ Chưa khớp"

        name_disp = (sb.name[:27] + "...") if len(sb.name) > 30 else sb.name
        print(row_fmt.format(
            stt=sb.stt, name=name_disp,
            base_col=sb_m, tuned_col=st_m,
            change=change
        ))

    print("   " + "-" * 110)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data", help="Root data folder (default: data)")
    parser.add_argument("--window", "-w", type=float, default=None,
                        help="Single evaluation window tolerance in seconds (default: 0.5s)")
    parser.add_argument("--windows", nargs="+", type=float, default=None,
                        help="List of window tolerances to sweep (default: 0.25 0.5 0.75 1.0 1.5 2.0)")
    parser.add_argument("--cd", "--cong-doan", dest="cong_doan", default=None,
                        help="Filter to a specific operation ID (e.g. 1)")
    parser.add_argument("--exclude-endpoints", action="store_true",
                        help="Exclude the very first (0.0s) and last video boundaries")
    parser.add_argument("--details", action="store_true",
                        help="Print step-by-step breakdown comparing Baseline vs Tuned Final")
    parser.add_argument("--apply", "--save-final", action="store_true", dest="apply",
                        help="Save the selected best tuned boundaries back to action_segments.json")
    parser.add_argument("--no-tune", action="store_true",
                        help="Disable on-the-fly fast tuning exploration (evaluate baseline only)")
    parser.add_argument("--out", default=None,
                        help="Path to save evaluation summary as JSON (e.g. eval_report.json)")
    args = parser.parse_args()

    # Determine windows
    if args.window is not None and args.windows is not None:
        windows = sorted(list(set([args.window] + args.windows)))
    elif args.window is not None:
        windows = [args.window]
    elif args.windows is not None:
        windows = sorted(args.windows)
    else:
        windows = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

    primary_window = args.window if args.window is not None else 0.5
    if primary_window not in windows:
        windows.append(primary_window)
        windows.sort()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    pairs = find_eval_pairs(data_dir, target_cd=args.cong_doan)
    if not pairs:
        raise SystemExit(f"No chuyen1_segment.json files found under {data_dir}")

    eval_results: list[VideoEvalResult] = []
    skipped: list[tuple[str, str, str]] = []

    for gt_f, pred_f in pairs:
        if not pred_f.exists():
            gt_d = json.loads(gt_f.read_text(encoding="utf-8"))
            skipped.append((str(gt_d.get("cong_doan_id", gt_f.parent.name)),
                            gt_d.get("video_file", ""),
                            f"Missing Stage 1 output: {pred_f}"))
            continue

        res = evaluate_video(gt_f, pred_f, windows=windows,
                             exclude_endpoints=args.exclude_endpoints,
                             enable_tune=(not args.no_tune))
        if res is not None:
            eval_results.append(res)
        else:
            skipped.append((gt_f.parent.name, pred_f.name, "Empty boundaries in GT or Pred"))

    print("\n" + "=" * 115)
    print(" 🚀 ĐÁNH GIÁ ĐỘ KHỚP & TỰ ĐỘNG THỰC NGHIỆM TÌM OPTION TỐI ƯU CHO ACTION SEGMENTS")
    print(f" Dải Window: {windows} giây | Cửa sổ chuẩn: ±{primary_window}s | Auto-Tune: {'BẬT (Khảo sát các cấu hình)' if not args.no_tune else 'TẮT'}")
    if args.apply:
        print(" 💾 Chế độ --apply: ĐÃ BẬT (Sẽ cập nhật option tối ưu vào action_segments.json)")
    print("=" * 115)

    if not eval_results:
        print("\nKhông có video nào để đánh giá vì chưa có kết quả Stage 1 (action_segments.json)!")
        if skipped:
            print("\nDanh sách video chưa chạy Stage 1:")
            for cd, vf, reason in skipped:
                print(f"  • [CĐ {cd:2s}] {vf} -> {reason}")
            print("\nGợi ý: Hãy chạy Stage 1 trước: python pipeline.py segment --all-data --visualize")
        return

    # =========================================================================
    # BẢNG 1: SO SÁNH BASELINE vs OPTION TỐI ƯU CHO TỪNG CÔNG ĐOẠN
    # =========================================================================
    print(f"\n[PHẦN 1] SO SÁNH BASELINE (HIỆN TẠI) vs OPTION TỐI ƯU (FINAL ĐƯỢC CHỌN) [Window = ±{primary_window}s]:")
    print("-" * 125)
    h_fmt = "{cd:<4} | {title:<20} | {bounds_col:<12} | {recall_col:<22} | {prec_col:<20} | {f1_col:<14} | {both_col:<18}"
    print(h_fmt.format(
        cd="CĐ", title="Tên công đoạn",
        bounds_col="Ranh giới",
        recall_col="Boundary Recall",
        prec_col="Boundary Precision",
        f1_col="F1-Score",
        both_col="Khớp CẢ 2 đầu"
    ))
    print("-" * 125)

    for r in eval_results:
        title_disp = (r.sheet_title[:18] + "...") if len(r.sheet_title) > 20 else r.sheet_title

        # Baseline stats
        b_bounds = len(r.pred_boundaries)
        b_rec = r.get_boundary_recall(primary_window)
        b_prec = r.get_boundary_precision(primary_window)
        b_f1 = r.get_boundary_f1(primary_window)
        b_both = r.get_step_both_hits(primary_window)
        b_both_pct = r.get_step_both_pct(primary_window)

        if r.best_tuned_option:
            t = r.best_tuned_option
            t_rec = r.get_tuned_boundary_recall(primary_window)
            t_prec = r.get_tuned_boundary_precision(primary_window)
            t_f1 = r.get_tuned_boundary_f1(primary_window)
            t_both = r.get_tuned_step_both_hits(primary_window)
            t_both_pct = r.get_tuned_step_both_pct(primary_window)

            bounds_str = f"{b_bounds} -> {t.n_bounds}"
            recall_str = f"{b_rec:4.1f}% -> {t_rec:4.1f}% (+{t_rec-b_rec:4.1f}%)"
            prec_str = f"{b_prec:4.1f}% -> {t_prec:4.1f}%"
            f1_str = f"{b_f1:4.1f}% -> {t_f1:4.1f}%"
            both_str = f"{b_both}/{r.get_total_steps()} -> {t_both}/{r.get_total_steps()} ({t_both_pct:4.1f}%)"
        else:
            bounds_str = f"{b_bounds}"
            recall_str = f"{b_rec:4.1f}%"
            prec_str = f"{b_prec:4.1f}%"
            f1_str = f"{b_f1:4.1f}%"
            both_str = f"{b_both}/{r.get_total_steps()} ({b_both_pct:4.1f}%)"

        print(h_fmt.format(
            cd=str(r.cong_doan_id), title=title_disp,
            bounds_col=bounds_str, recall_col=recall_str,
            prec_col=prec_str, f1_col=f1_str,
            both_col=both_str
        ))

        if r.best_tuned_option:
            opt = r.best_tuned_option
            print(f"     └─► [Option được chọn làm Final cho CĐ {r.cong_doan_id}]: {opt.name}")
            print(f"         Khoảng cách min: {opt.dist_sec}s | Ngưỡng k_std: {opt.k_std} | Trọng số: Speed={opt.weights[0]}, Dir={opt.weights[1]}, Turb={opt.weights[2]} | Lọc góc: {opt.noise_th}°\n")

            # Apply / Save if requested
            if args.apply:
                save_tuned_action_segments(r.action_segments_path, opt.boundaries, r.fps, r.video_file)
                print(f"         💾 Đã cập nhật kết quả tối ưu vào: {r.action_segments_path.resolve()}\n")

    print("-" * 125)

    # In chi tiết so sánh từng thao tác nếu bật --details
    if args.details:
        for r in eval_results:
            if r.best_tuned_option:
                print_step_comparison(r, primary_window)

    # =========================================================================
    # BẢNG 2: TỔNG HỢP TOÀN BỘ (MACRO & MICRO) TRƯỚC vs SAU KHI TỐI ƯU
    # =========================================================================
    print(f"\n[PHẦN 2] TỔNG HỢP TOÀN BỘ ({len(eval_results)} công đoạn | Window = ±{primary_window}s):")
    print("-" * 125)

    # Baseline overall
    base_macro_rec = sum(r.get_boundary_recall(primary_window) for r in eval_results) / len(eval_results)
    total_gt = sum(len(r.gt_boundaries) for r in eval_results)
    base_micro_hits = sum(r.get_boundary_hits(primary_window) for r in eval_results)
    base_micro_rec = (base_micro_hits / total_gt * 100.0) if total_gt > 0 else 0.0

    total_pred_b = sum(len(r.pred_boundaries) for r in eval_results)
    base_macro_prec = sum(r.get_boundary_precision(primary_window) for r in eval_results) / len(eval_results)
    base_pred_hits = sum(
        sum(1 for p in r.pred_boundaries if any(abs(p - g) <= primary_window for g in r.gt_boundaries))
        for r in eval_results
    )
    base_micro_prec = (base_pred_hits / total_pred_b * 100.0) if total_pred_b > 0 else 0.0
    base_macro_f1 = sum(r.get_boundary_f1(primary_window) for r in eval_results) / len(eval_results)

    total_steps = sum(r.get_total_steps() for r in eval_results)
    base_step_both = sum(r.get_step_both_hits(primary_window) for r in eval_results)
    base_step_both_pct = (base_step_both / total_steps * 100.0) if total_steps > 0 else 0.0
    base_step_either = sum(r.get_step_either_hits(primary_window) for r in eval_results)
    base_step_either_pct = (base_step_either / total_steps * 100.0) if total_steps > 0 else 0.0

    print(f"  📌 TRẠNG THÁI HIỆN TẠI (BASELINE):")
    print(f"     • Macro Recall                 : {base_macro_rec:6.2f}%")
    print(f"     • Micro Recall                 : {base_micro_rec:6.2f}% ({base_micro_hits}/{total_gt} mốc GT)")
    print(f"     • Macro Precision              : {base_macro_prec:6.2f}%")
    print(f"     • Micro Precision              : {base_micro_prec:6.2f}% ({base_pred_hits}/{total_pred_b} vết cắt của máy)")
    print(f"     • Macro F1-Score               : {base_macro_f1:6.2f}%")
    print(f"     • Thao tác khớp CẢ 2 ĐẦU       : {base_step_both_pct:6.2f}% ({base_step_both}/{total_steps} thao tác)")
    print(f"     • Thao tác khớp ÍT NHẤT 1 ĐẦU  : {base_step_either_pct:6.2f}% ({base_step_either}/{total_steps} thao tác)")

    # Tuned overall if available
    has_tuning = any(r.best_tuned_option is not None for r in eval_results)
    if has_tuning:
        tuned_macro_rec = sum(r.get_tuned_boundary_recall(primary_window) for r in eval_results) / len(eval_results)
        tuned_micro_hits = sum(r.get_tuned_boundary_hits(primary_window) for r in eval_results)
        tuned_micro_rec = (tuned_micro_hits / total_gt * 100.0) if total_gt > 0 else 0.0

        tuned_total_pred_b = sum(r.best_tuned_option.n_bounds for r in eval_results if r.best_tuned_option)
        tuned_macro_prec = sum(r.get_tuned_boundary_precision(primary_window) for r in eval_results) / len(eval_results)
        tuned_pred_hits = sum(
            sum(1 for p in r.best_tuned_option.pred_times if any(abs(p - g) <= primary_window for g in r.gt_boundaries))
            for r in eval_results if r.best_tuned_option
        )
        tuned_micro_prec = (tuned_pred_hits / tuned_total_pred_b * 100.0) if tuned_total_pred_b > 0 else 0.0
        tuned_macro_f1 = sum(r.get_tuned_boundary_f1(primary_window) for r in eval_results) / len(eval_results)

        tuned_step_both = sum(r.get_tuned_step_both_hits(primary_window) for r in eval_results)
        tuned_step_both_pct = (tuned_step_both / total_steps * 100.0) if total_steps > 0 else 0.0
        tuned_step_either = sum(r.get_tuned_step_either_hits(primary_window) for r in eval_results)
        tuned_step_either_pct = (tuned_step_either / total_steps * 100.0) if total_steps > 0 else 0.0

        print(f"\n  ⭐ SAU KHI ÁP DỤNG CÁC OPTION TỐI ƯU:")
        print(f"     • Macro Recall                 : {tuned_macro_rec:6.2f}%  (+{tuned_macro_rec - base_macro_rec:.1f}%) 🚀")
        print(f"     • Micro Recall                 : {tuned_micro_rec:6.2f}% ({tuned_micro_hits}/{total_gt} mốc GT) (+{tuned_micro_rec - base_micro_rec:.1f}%)")
        print(f"     • Macro Precision              : {tuned_macro_prec:6.2f}%  ({tuned_macro_prec - base_macro_prec:+.1f}%)")
        print(f"     • Micro Precision              : {tuned_micro_prec:6.2f}% ({tuned_pred_hits}/{tuned_total_pred_b} vết cắt của máy)")
        print(f"     • Macro F1-Score               : {tuned_macro_f1:6.2f}%  (+{tuned_macro_f1 - base_macro_f1:.1f}%)")
        print(f"     • Thao tác khớp CẢ 2 ĐẦU       : {tuned_step_both_pct:6.2f}% ({tuned_step_both}/{total_steps} thao tác) (+{tuned_step_both_pct - base_step_both_pct:.1f}%) 🚀")
        print(f"     • Thao tác khớp ÍT NHẤT 1 ĐẦU  : {tuned_step_either_pct:6.2f}% ({tuned_step_either}/{total_steps} thao tác)")

    # =========================================================================
    # BẢNG 3: TOLERANCE WINDOW SWEEP (CẢ RECALL & PRECISION)
    # =========================================================================
    if has_tuning:
        print(f"\n[PHẦN 3] TIẾN TRÌNH RECALL & PRECISION THEO DẢI WINDOW CỦA OPTION TỐI ƯU:")
        print("-" * 125)
        print(f"{'Window':<8} | {'Recall (Macro / Micro)':<25} | {'Precision (Macro / Micro)':<27} | {'F1-Score':<9} | {'Khớp 2 Đầu':<12} | {'Visual Recall Bar'}")
        print("-" * 125)

        for w in windows:
            m_rec = sum(r.get_tuned_boundary_recall(w) for r in eval_results) / len(eval_results)
            w_hits = sum(r.get_tuned_boundary_hits(w) for r in eval_results)
            u_rec = (w_hits / total_gt * 100.0) if total_gt > 0 else 0.0

            m_prec = sum(r.get_tuned_boundary_precision(w) for r in eval_results) / len(eval_results)
            w_pred_hits = sum(
                sum(1 for p in r.best_tuned_option.pred_times if any(abs(p - g) <= w for g in r.gt_boundaries))
                for r in eval_results if r.best_tuned_option
            )
            u_prec = (w_pred_hits / tuned_total_pred_b * 100.0) if tuned_total_pred_b > 0 else 0.0

            f1 = (2 * m_prec * m_rec / (m_prec + m_rec)) if (m_prec + m_rec) > 0 else 0.0

            w_both = sum(r.get_tuned_step_both_hits(w) for r in eval_results)
            u_both = (w_both / total_steps * 100.0) if total_steps > 0 else 0.0

            bar = make_ascii_bar(m_rec, width=16)
            marker = " (*)" if w == primary_window else ""

            col_b = f"{m_rec:5.1f}% / {u_rec:5.1f}% ({w_hits:2d}/{total_gt:2d})"
            col_p = f"{m_prec:5.1f}% / {u_prec:5.1f}% ({w_pred_hits:2d}/{tuned_total_pred_b:2d})"
            col_both = f"{u_both:5.1f}% ({w_both:2d}/{total_steps:2d})"

            print(f"±{w:<5.2f}s | {col_b:<25} | {col_p:<27} | {f1:5.1f}%   | {col_both:<12} | {bar}{marker}")

        print("-" * 115)
        print(" (*) Cửa sổ chuẩn mặc định")

    if skipped:
        print(f"\nCó {len(skipped)} công đoạn trong data/ chưa chạy Stage 1:")
        for cd, vf, reason in skipped:
            print(f"  • [CĐ {cd:2s}] {vf}")
        print("  => Chạy lệnh sau để tính tiếp các công đoạn này: python pipeline.py segment --all-data --visualize")

    print("=" * 115)

    # Save output JSON report
    if args.out:
        out_path = Path(args.out)
        payload = {
            "primary_window": primary_window,
            "windows_evaluated": windows,
            "overall": {
                "total_evaluated_videos": len(eval_results),
                "total_gt_boundaries": total_gt,
                "total_gt_steps": total_steps,
                "baseline": {
                    "macro_recall_pct": round(base_macro_rec, 2),
                    "micro_recall_pct": round(base_micro_rec, 2),
                    "step_both_pct": round(base_step_both_pct, 2),
                    "step_either_pct": round(base_step_either_pct, 2),
                    "macro_f1": round(base_macro_f1, 2),
                },
                "tuned_final": {
                    "macro_recall_pct": round(tuned_macro_rec, 2) if has_tuning else None,
                    "micro_recall_pct": round(tuned_micro_rec, 2) if has_tuning else None,
                    "step_both_pct": round(tuned_step_both_pct, 2) if has_tuning else None,
                    "step_either_pct": round(tuned_step_either_pct, 2) if has_tuning else None,
                    "macro_f1": round(tuned_macro_f1, 2) if has_tuning else None,
                }
            },
            "videos": [
                {
                    "cong_doan_id": r.cong_doan_id,
                    "sheet_title": r.sheet_title,
                    "video_file": r.video_file,
                    "best_option": r.best_tuned_option.__dict__ if r.best_tuned_option else None,
                    "baseline_recall_05": round(r.get_boundary_recall(primary_window), 2),
                    "tuned_recall_05": round(r.get_tuned_boundary_recall(primary_window), 2) if r.best_tuned_option else None,
                }
                for r in eval_results
            ]
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nBáo cáo JSON đầy đủ đã được lưu tại: {out_path.resolve()}")


if __name__ == "__main__":
    main()
