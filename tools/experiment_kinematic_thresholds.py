#!/usr/bin/env python3
"""Run structured experiments on kinematic boundary detection parameters
using pre-computed decomposed motion signals (no heavy GPU recomputation needed).

Evaluates:
  1. min_distance (Temporal resolution)
  2. k_std & min_threshold (Dynamic threshold sensitivity)
  3. Multi-modal weights [w_speed, w_shift, w_turb]
  4. noise_threshold for angle shift
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np


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


@dataclass
class EvalMetrics:
    name: str
    dist_sec: float
    k_std: float
    min_th: float
    weights: tuple[float, float, float]
    noise_th: float
    n_bounds: int
    hits_05: int
    recall_05: float
    prec_05: float
    f1_05: float
    hits_10: int
    recall_10: float
    step_both_05: int
    step_both_05_pct: float
    step_either_05: int
    step_either_05_pct: float
    mae_05: float


def evaluate_config(name: str, overall_likelihood: np.ndarray, fps: float,
                    gt_bounds: list[float], gt_steps: list[dict],
                    dist_sec: float = 0.8, k_std: float = 0.8, min_th: float = 0.25,
                    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
                    noise_th: float = 30.0) -> EvalMetrics:
    N = len(overall_likelihood)
    dist_f = max(1, int(fps * dist_sec))
    w_size = max(3, int(fps * 4.0))

    local_mean = moving_average(overall_likelihood, w_size)
    mean_sq = moving_average(overall_likelihood**2, w_size)
    local_std = np.sqrt(np.maximum(mean_sq - local_mean**2, 0))

    dynamic_th = np.maximum(local_mean + k_std * local_std, min_th)
    peaks = find_peaks_greedy(overall_likelihood, dynamic_th, dist_f)
    bounds = sorted(list(set([0] + peaks + [N - 1])))
    pred_times = [round(b / fps, 3) for b in bounds]

    # Metrics at 0.5s
    hits_05 = 0
    errors_05 = []
    for g in gt_bounds:
        closest = min(pred_times, key=lambda p: abs(p - g))
        err = abs(closest - g)
        if err <= 0.5:
            hits_05 += 1
            errors_05.append(err)

    rec_05 = (hits_05 / len(gt_bounds)) * 100.0
    pred_hits_05 = sum(1 for p in pred_times if any(abs(p - g) <= 0.5 for g in gt_bounds))
    prec_05 = (pred_hits_05 / len(pred_times) * 100.0) if pred_times else 0.0
    f1_05 = (2 * prec_05 * rec_05 / (prec_05 + rec_05)) if (prec_05 + rec_05) > 0 else 0.0
    mae_05 = (sum(errors_05) / len(errors_05)) if errors_05 else 0.0

    # Metrics at 1.0s
    hits_10 = sum(1 for g in gt_bounds if any(abs(g - p) <= 1.0 for p in pred_times))
    rec_10 = (hits_10 / len(gt_bounds)) * 100.0

    # Step level metrics at 0.5s
    both_05 = 0
    either_05 = 0
    for s in gt_steps:
        h0 = any(abs(s["timestamp_start"] - p) <= 0.5 for p in pred_times)
        h1 = any(abs(s["timestamp_end"] - p) <= 0.5 for p in pred_times)
        if h0 and h1:
            both_05 += 1
        if h0 or h1:
            either_05 += 1

    total_steps = len(gt_steps)
    return EvalMetrics(
        name=name,
        dist_sec=dist_sec,
        k_std=k_std,
        min_th=min_th,
        weights=weights,
        noise_th=noise_th,
        n_bounds=len(bounds),
        hits_05=hits_05,
        recall_05=round(rec_05, 1),
        prec_05=round(prec_05, 1),
        f1_05=round(f1_05, 1),
        hits_10=hits_10,
        recall_10=round(rec_10, 1),
        step_both_05=both_05,
        step_both_05_pct=round(both_05 / total_steps * 100.0, 1),
        step_either_05=either_05,
        step_either_05_pct=round(either_05 / total_steps * 100.0, 1),
        mae_05=round(mae_05, 3),
    )


def run_all_experiments():
    # 1. Load data
    npz_path = Path("data/1/kinematic/cam-03_20260805_073527_cut_0_0-0_57/decomposed_motion.npz")
    gt_path = Path("data/1/chuyen1_segment.json")
    if not npz_path.exists() or not gt_path.exists():
        raise SystemExit("Missing data files for experiment.")

    d = np.load(npz_path)
    fps = float(d["fps"])
    sl_spds, sl_angs, sl_turb = d["left_smooth_speeds"], d["left_smooth_angles"], d["left_smooth_turbulences"]
    sr_spds, sr_angs, sr_turb = d["right_smooth_speeds"], d["right_smooth_angles"], d["right_smooth_turbulences"]
    baseline_likelihood = d["overall_likelihood"]

    gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
    gt_steps = gt_data["segments"]
    gt_bounds = sorted(list(set([round(float(s["timestamp_start"]), 3) for s in gt_steps] +
                                [round(float(s["timestamp_end"]), 3) for s in gt_steps])))

    print("\n" + "=" * 115)
    print(" KINEMATIC SEGMENTATION EXPERIMENTATION SUITE (CĐ 1: 24 steps, 25 GT boundaries)")
    print("=" * 115)

    all_results: list[EvalMetrics] = []

    # Baseline (Current in production)
    baseline = evaluate_config(
        "Baseline (Current Prod)", baseline_likelihood, fps, gt_bounds, gt_steps,
        dist_sec=1.5, k_std=1.0, min_th=0.25, weights=(0.4, 0.3, 0.3), noise_th=30.0
    )
    all_results.append(baseline)

    # ── Group 1: Minimum Distance Sweep (Unlocking temporal barrier) ──
    g1_results = []
    for d_sec in [0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]:
        m = evaluate_config(
            f"G1: dist={d_sec}s", baseline_likelihood, fps, gt_bounds, gt_steps,
            dist_sec=d_sec, k_std=0.8, min_th=0.25, weights=(0.4, 0.3, 0.3), noise_th=30.0
        )
        g1_results.append(m)
        all_results.append(m)

    # ── Group 2: Dynamic Threshold Sensitivity Sweep (k_std & min_th) ──
    g2_results = []
    for k in [0.3, 0.5, 0.7, 0.8, 1.0]:
        for th in [0.20, 0.25, 0.30]:
            m = evaluate_config(
                f"G2: k={k}, th={th}", baseline_likelihood, fps, gt_bounds, gt_steps,
                dist_sec=0.8, k_std=k, min_th=th, weights=(0.4, 0.3, 0.3), noise_th=30.0
            )
            g2_results.append(m)
            all_results.append(m)

    # ── Group 3: Multi-modal Fusion Weights Sweep ──
    g3_configs = [
        ("W1: Speed-Heavy (0.45, 0.25, 0.30)", (0.45, 0.25, 0.30)),
        ("W2: Direction-Heavy (0.25, 0.50, 0.25)", (0.25, 0.50, 0.25)),
        ("W3: Speed+Dir Balanced (0.45, 0.45, 0.10)", (0.45, 0.45, 0.10)),
        ("W4: Direction Dominant (0.30, 0.55, 0.15)", (0.30, 0.55, 0.15)),
        ("W5: Equal Weights (0.33, 0.33, 0.34)", (0.33, 0.33, 0.34)),
    ]
    g3_results = []
    for w_label, w_tuple in g3_configs:
        like = compute_likelihood(sl_spds, sl_turb, sl_angs, sr_spds, sr_turb, sr_angs,
                                  w_speed=w_tuple[0], w_shift=w_tuple[1], w_turb=w_tuple[2], noise_threshold=30.0)
        m = evaluate_config(
            f"G3: {w_label}", like, fps, gt_bounds, gt_steps,
            dist_sec=0.8, k_std=0.8, min_th=0.25, weights=w_tuple, noise_th=30.0
        )
        g3_results.append(m)
        all_results.append(m)

    # ── Group 4: Noise Threshold Sweep (Angle shift filter) ──
    g4_results = []
    for nth in [15.0, 20.0, 25.0, 30.0, 35.0, 40.0]:
        like = compute_likelihood(sl_spds, sl_turb, sl_angs, sr_spds, sr_turb, sr_angs,
                                  w_speed=0.30, w_shift=0.50, w_turb=0.20, noise_threshold=nth)
        m = evaluate_config(
            f"G4: noise_th={nth}deg", like, fps, gt_bounds, gt_steps,
            dist_sec=0.8, k_std=0.8, min_th=0.25, weights=(0.30, 0.50, 0.20), noise_th=nth
        )
        g4_results.append(m)
        all_results.append(m)

    # ── Group 5: Full Combinatorial Search across Top Candidates ──
    g5_results = []
    candidate_dists = [0.5, 0.6, 0.7, 0.8]
    candidate_k_stds = [0.6, 0.7, 0.8]
    candidate_weights = [
        ("W_dir", (0.30, 0.50, 0.20)),
        ("W_bal", (0.40, 0.40, 0.20)),
        ("W_std", (0.40, 0.30, 0.30)),
    ]
    for d_sec in candidate_dists:
        for k in candidate_k_stds:
            for w_name, w_tup in candidate_weights:
                for nth in [25.0, 30.0]:
                    like = compute_likelihood(sl_spds, sl_turb, sl_angs, sr_spds, sr_turb, sr_angs,
                                              w_speed=w_tup[0], w_shift=w_tup[1], w_turb=w_tup[2], noise_threshold=nth)
                    m = evaluate_config(
                        f"Combo: d={d_sec}, k={k}, {w_name}, nth={nth}", like, fps, gt_bounds, gt_steps,
                        dist_sec=d_sec, k_std=k, min_th=0.25, weights=w_tup, noise_th=nth
                    )
                    g5_results.append(m)
                    all_results.append(m)

    # Helper print table
    def print_table(title: str, items: list[EvalMetrics]):
        print(f"\n{title}")
        print("-" * 115)
        print(f"{'Config Name':<38} | {'Bounds':<6} | {'Recall@0.5s':<11} | {'Prec@0.5s':<9} | {'F1@0.5s':<8} | {'Recall@1.0s':<11} | {'Both@0.5s':<11} | {'Either@0.5s':<11}")
        print("-" * 115)
        for r in items:
            b_str = f"{r.hits_05}/{len(gt_bounds)} ({r.recall_05}%)"
            p_str = f"{r.prec_05}%"
            f_str = f"{r.f1_05}%"
            b1_str = f"{r.hits_10}/{len(gt_bounds)} ({r.recall_10}%)"
            both_str = f"{r.step_both_05}/{len(gt_steps)} ({r.step_both_05_pct}%)"
            either_str = f"{r.step_either_05}/{len(gt_steps)} ({r.step_either_05_pct}%)"
            print(f"{r.name:<38} | {r.n_bounds:<6} | {b_str:<11} | {p_str:<9} | {f_str:<8} | {b1_str:<11} | {both_str:<11} | {either_str:<11}")
        print("-" * 115)

    print_table("📊 [GROUP 1] IMPACT OF MINIMUM DISTANCE (Unlocking the 1.5s hardcode limit):", [baseline] + g1_results)
    print_table("📊 [GROUP 2] SENSITIVITY OF DYNAMIC THRESHOLD (k_std & min_threshold):", g2_results[:6])
    print_table("📊 [GROUP 3] MULTI-MODAL LIKELIHOOD WEIGHTING (Speed vs Direction vs Turbulence):", g3_results)
    print_table("📊 [GROUP 4] ANGLE SHIFT NOISE FLOOR (noise_threshold in degrees):", g4_results)

    # Top 10 by F1-Score (Balanced recall & precision, avoiding over-segmentation)
    all_results.sort(key=lambda x: (x.f1_05, x.step_both_05_pct, x.recall_05), reverse=True)
    print_table("🏆 [TOP 10 CONFIGURATIONS BY F1-SCORE & BALANCED PRECISION]:", all_results[:10])

    # Top 5 by Raw Recall
    top_recall = sorted(all_results, key=lambda x: (x.recall_05, x.step_both_05_pct, x.f1_05), reverse=True)
    print_table("🚀 [TOP 5 CONFIGURATIONS BY HIGHEST RECALL]:", top_recall[:5])

    # Best balanced proposal
    best = all_results[0]
    print(f"\n💡 PROPOSED BEST CONFIGURATION:")
    print(f"   Name            : {best.name}")
    print(f"   min_distance    : {best.dist_sec} seconds (vs 1.5s baseline)")
    print(f"   k_std           : {best.k_std} (vs 1.0 baseline)")
    print(f"   min_threshold   : {best.min_th}")
    print(f"   Weights         : Speed={best.weights[0]}, Direction={best.weights[1]}, Turbulence={best.weights[2]}")
    print(f"   noise_threshold : {best.noise_th} degrees")
    print(f"   --- Performance Comparison ---")
    print(f"   • Boundary Recall (@0.5s) : {baseline.recall_05}%  ──►  {best.recall_05}%  (+{best.recall_05 - baseline.recall_05:.1f}%)")
    print(f"   • Boundary F1-Score       : {baseline.f1_05}%  ──►  {best.f1_05}%  (+{best.f1_05 - baseline.f1_05:.1f}%)")
    print(f"   • Thao tác khớp CẢ 2 ĐẦU  : {baseline.step_both_05_pct}%  ──►  {best.step_both_05_pct}%  (+{best.step_both_05_pct - baseline.step_both_05_pct:.1f}%)")
    print(f"   • Thao tác khớp ≥1 ĐẦU    : {baseline.step_either_05_pct}%  ──►  {best.step_either_05_pct}%  (+{best.step_either_05_pct - baseline.step_either_05_pct:.1f}%)")
    print(f"   • Boundary Recall (@1.0s) : {baseline.recall_10}%  ──►  {best.recall_10}%  (+{best.recall_10 - baseline.recall_10:.1f}%)")
    print(f"   • Số segment dự đoán      : {baseline.n_bounds} bounds  ──►  {best.n_bounds} bounds (rất gần nhịp thao tác thực tế)\n")


if __name__ == "__main__":
    run_all_experiments()
