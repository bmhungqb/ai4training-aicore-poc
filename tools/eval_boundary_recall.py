#!/usr/bin/env python3
"""Evaluate alignment between kinematic action segmentation boundaries
and manual ground-truth step boundaries from chuyen1_segment.json.

Evaluates at two complementary levels:
  1. Boundary Level: Each unique transition timestamp (start/end).
  2. Step Level: Each manual operation ("thao tác") with both start and end timestamps.

Aggregations:
  - Per-Công Đoạn (Per-video): Detailed metrics for each operation.
  - Macro (Average per công đoạn): Unweighted mean across videos.
  - Micro (All steps pooled): Aggregated across all operations/steps combined.
  - Tolerance Window Sweep: Sensitivity curve across multiple windows (0.25s -> 2.0s).

Usage:
  python -m tools.eval_boundary_recall
  python -m tools.eval_boundary_recall -w 0.5
  python -m tools.eval_boundary_recall --cd 1 --details
  python -m tools.eval_boundary_recall --details
  python -m tools.eval_boundary_recall --out eval_report.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
class VideoEvalResult:
    cong_doan_id: int | str
    sheet_title: str
    video_file: str
    action_segments_path: Path
    gt_boundaries: list[float]
    pred_boundaries: list[float]
    gt_steps: list[dict]
    # window -> list of BoundaryMatch
    window_boundary_matches: dict[float, list[BoundaryMatch]] = field(default_factory=dict)
    # window -> list of StepMatch
    window_step_matches: dict[float, list[StepMatch]] = field(default_factory=dict)

    # --- Boundary-level metrics ---
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

    # --- Step-level metrics ---
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

    def get_step_start_hits(self, window: float) -> int:
        return sum(1 for s in self.window_step_matches.get(window, []) if s.start_hit)

    def get_step_start_pct(self, window: float) -> float:
        total = self.get_total_steps()
        return (self.get_step_start_hits(window) / total * 100.0) if total > 0 else 0.0

    def get_step_end_hits(self, window: float) -> int:
        return sum(1 for s in self.window_step_matches.get(window, []) if s.end_hit)

    def get_step_end_pct(self, window: float) -> float:
        total = self.get_total_steps()
        return (self.get_step_end_hits(window) / total * 100.0) if total > 0 else 0.0


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


def extract_pred_boundaries(pred_file: Path, exclude_endpoints: bool = False) -> list[float]:
    """Load predicted boundaries from action_segments.json."""
    data = json.loads(pred_file.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    if not segments:
        return []

    raw_boundaries: set[float] = set()
    for s in segments:
        t0 = round(float(s.get("start_time_s", 0.0)), 3)
        t1 = round(float(s.get("end_time_s", 0.0)), 3)
        raw_boundaries.add(t0)
        raw_boundaries.add(t1)

    sorted_bounds = sorted(list(raw_boundaries))
    if exclude_endpoints and len(sorted_bounds) > 2:
        sorted_bounds = sorted_bounds[1:-1]

    return sorted_bounds


def find_closest(target: float, candidates: list[float]) -> tuple[float | None, float]:
    """Find closest candidate to target timestamp. Returns (closest_val, abs_error)."""
    if not candidates:
        return None, float("inf")
    closest = min(candidates, key=lambda c: abs(c - target))
    return closest, abs(closest - target)


def evaluate_video(gt_file: Path, pred_file: Path, windows: list[float],
                   exclude_endpoints: bool = False) -> VideoEvalResult | None:
    """Evaluate one video at both boundary-level and step-level across all windows."""
    gt_data = json.loads(gt_file.read_text(encoding="utf-8"))
    cd_id = gt_data.get("cong_doan_id", gt_file.parent.name)
    sheet_title = gt_data.get("sheet_title", f"CĐ {cd_id}")
    video_file = gt_data.get("video_file", "")

    gt_bounds, steps = extract_gt_data(gt_file, exclude_endpoints=exclude_endpoints)
    pred_bounds = extract_pred_boundaries(pred_file, exclude_endpoints=exclude_endpoints)

    if not gt_bounds or not pred_bounds:
        return None

    # Map bound timestamp to step description
    bound_to_step: dict[float, tuple[int, str]] = {}
    for st in steps:
        bound_to_step.setdefault(st["t0"], (st["stt"], f"Start of: {st['name']}"))
        bound_to_step.setdefault(st["t1"], (st["stt"], f"End of: {st['name']}"))

    result = VideoEvalResult(
        cong_doan_id=cd_id,
        sheet_title=sheet_title,
        video_file=video_file,
        action_segments_path=pred_file,
        gt_boundaries=gt_bounds,
        pred_boundaries=pred_bounds,
        gt_steps=steps,
    )

    for w in windows:
        # 1. Boundary-level matching
        b_matches: list[BoundaryMatch] = []
        for g in gt_bounds:
            closest_p, min_dist = find_closest(g, pred_bounds)
            is_hit = (min_dist <= w)
            stt, sname = bound_to_step.get(g, (0, ""))
            b_matches.append(BoundaryMatch(
                gt_time=g,
                pred_time=closest_p,
                error=round(min_dist, 3) if closest_p is not None else None,
                is_hit=is_hit,
                step_name=sname,
                step_stt=stt,
            ))
        result.window_boundary_matches[w] = b_matches

        # 2. Step-level matching
        s_matches: list[StepMatch] = []
        for st in steps:
            p0, err0 = find_closest(st["t0"], pred_bounds)
            p1, err1 = find_closest(st["t1"], pred_bounds)
            s_matches.append(StepMatch(
                stt=st["stt"],
                name=st["name"],
                start_gt=st["t0"],
                start_pred=p0,
                start_err=round(err0, 3) if p0 is not None else None,
                start_hit=(err0 <= w),
                end_gt=st["t1"],
                end_pred=p1,
                end_err=round(err1, 3) if p1 is not None else None,
                end_hit=(err1 <= w),
            ))
        result.window_step_matches[w] = s_matches

    return result


def find_eval_pairs(data_dir: Path, target_cd: str | None = None) -> list[tuple[Path, Path]]:
    """Scan data_dir for chuyen1_segment.json and matching action_segments.json."""
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


def print_step_details(res: VideoEvalResult, window: float) -> None:
    """Print human-readable table of all steps in one công đoạn."""
    print(f"\n   📋 CHI TIẾT TỪNG THAO TÁC (CĐ {res.cong_doan_id} | {res.sheet_title}):")
    print(f"      Video: {res.video_file} | Cửa sổ sai số: ±{window}s")
    print("   " + "-" * 105)
    row_fmt = "   {stt:<3} | {name:<35} | {start_col:<20} | {end_col:<20} | {status}"
    print(row_fmt.format(
        stt="STT", name="Tên thao tác",
        start_col=f"Start (GT -> Pred / Lệch)",
        end_col=f"End (GT -> Pred / Lệch)",
        status="Đánh giá"
    ))
    print("   " + "-" * 105)

    for s in res.window_step_matches.get(window, []):
        st_mark = "✅" if s.start_hit else "❌"
        st_err = f"{s.start_err:4.2f}s" if s.start_err is not None else "N/A"
        st_pred = f"{s.start_pred:5.2f}s" if s.start_pred is not None else " None"
        start_col = f"{s.start_gt:5.2f}s -> {st_pred} ({st_err}) {st_mark}"

        en_mark = "✅" if s.end_hit else "❌"
        en_err = f"{s.end_err:4.2f}s" if s.end_err is not None else "N/A"
        en_pred = f"{s.end_pred:5.2f}s" if s.end_pred is not None else " None"
        end_col = f"{s.end_gt:5.2f}s -> {en_pred} ({en_err}) {en_mark}"

        name_disp = (s.name[:32] + "...") if len(s.name) > 35 else s.name
        print(row_fmt.format(
            stt=s.stt, name=name_disp,
            start_col=start_col, end_col=end_col,
            status=s.status_label
        ))

    print("   " + "-" * 105)
    both = res.get_step_both_hits(window)
    either = res.get_step_either_hits(window)
    total = res.get_total_steps()
    print(f"   => Tổng kết CĐ {res.cong_doan_id}: Khớp cả 2 đầu: {both}/{total} ({res.get_step_both_pct(window):.1f}%) | Khớp ít nhất 1 đầu: {either}/{total} ({res.get_step_either_pct(window):.1f}%)\n")


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
                        help="Print step-by-step breakdown for each evaluated video")
    parser.add_argument("--out", default=None,
                        help="Path to save evaluation summary as JSON (e.g. eval_report.json)")
    args = parser.parse_args()

    # Determine windows list
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

        res = evaluate_video(gt_f, pred_f, windows=windows, exclude_endpoints=args.exclude_endpoints)
        if res is not None:
            eval_results.append(res)
        else:
            skipped.append((gt_f.parent.name, pred_f.name, "Empty boundaries in GT or Pred"))

    print("\n" + "=" * 115)
    print(" 📊 BÁO CÁO ĐÁNH GIÁ ĐỘ KHỚP RANH GIỚI KINEMATIC ACTION SEGMENT vs MANUAL STEP (CHUYEN 1)")
    print(f" Dải Window đánh giá: {windows} giây | Cửa sổ chuẩn: ±{primary_window}s")
    if args.exclude_endpoints:
        print(" Lưu ý: Đã loại bỏ điểm đầu mút video (0s và mốc kết thúc video).")
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
    # BẢNG 1: PHÂN TÍCH CHI TIẾT CHO TỪNG CÔNG ĐOẠN (AT PRIMARY WINDOW)
    # =========================================================================
    print(f"\n[PHẦN 1] THỐNG KÊ CHI TIẾT TỪNG CÔNG ĐOẠN (Tại Window = ±{primary_window}s):")
    print("-" * 115)
    header = (f"{'CĐ':<4} | {'Tên công đoạn / Video':<36} | {'Mốc GT':<7} | {'Khớp Mốc':<9} | {'% Recall':<9} "
              f"| {'Tổng Thao Tác':<14} | {'Khớp 2 Đầu':<12} | {'Khớp ≥1 Đầu':<12} | {'MAE (s)':<7}")
    print(header)
    print("-" * 115)

    for r in eval_results:
        b_hits = r.get_boundary_hits(primary_window)
        b_total = len(r.gt_boundaries)
        b_rec = r.get_boundary_recall(primary_window)
        mae = r.get_boundary_mae(primary_window)

        s_total = r.get_total_steps()
        s_both = r.get_step_both_hits(primary_window)
        s_both_pct = r.get_step_both_pct(primary_window)
        s_either = r.get_step_either_hits(primary_window)
        s_either_pct = r.get_step_either_pct(primary_window)

        title_disp = f"{r.sheet_title}"
        if len(title_disp) > 34:
            title_disp = title_disp[:31] + "..."

        row_str = (f"{str(r.cong_doan_id):<4} | {title_disp:<36} | {b_total:<7} | {b_hits:<2}/{b_total:<6} | {b_rec:6.2f}%   "
                   f"| {s_total:<14} | {s_both:<2}/{s_total:<2} ({s_both_pct:4.1f}%) | {s_either:<2}/{s_total:<2} ({s_either_pct:4.1f}%) | {mae:5.3f}s")
        print(row_str)

    print("-" * 115)

    # In chi tiết từng bước thao tác nếu người dùng yêu cầu (--details hoặc chỉ định cụ thể --cd)
    if args.details or args.cong_doan is not None:
        for r in eval_results:
            print_step_details(r, primary_window)

    # =========================================================================
    # BẢNG 2: TỔNG HỢP CHO TOÀN BỘ (OVERALL POOLED: MACRO & MICRO)
    # =========================================================================
    print(f"\n[PHẦN 2] TỔNG HỢP CHO TOÀN BỘ ({len(eval_results)} công đoạn đã đánh giá):")
    print("-" * 115)

    # Boundary metrics
    macro_b_rec = sum(r.get_boundary_recall(primary_window) for r in eval_results) / len(eval_results)
    total_b_hits = sum(r.get_boundary_hits(primary_window) for r in eval_results)
    total_b_gt = sum(len(r.gt_boundaries) for r in eval_results)
    micro_b_rec = (total_b_hits / total_b_gt * 100.0) if total_b_gt > 0 else 0.0

    macro_b_prec = sum(r.get_boundary_precision(primary_window) for r in eval_results) / len(eval_results)
    total_pred_b = sum(len(r.pred_boundaries) for r in eval_results)
    total_pred_hits = sum(
        sum(1 for p in r.pred_boundaries if any(abs(p - g) <= primary_window for g in r.gt_boundaries))
        for r in eval_results
    )
    micro_b_prec = (total_pred_hits / total_pred_b * 100.0) if total_pred_b > 0 else 0.0

    # Step metrics
    macro_s_both = sum(r.get_step_both_pct(primary_window) for r in eval_results) / len(eval_results)
    macro_s_either = sum(r.get_step_either_pct(primary_window) for r in eval_results) / len(eval_results)

    total_s_cnt = sum(r.get_total_steps() for r in eval_results)
    total_s_both = sum(r.get_step_both_hits(primary_window) for r in eval_results)
    micro_s_both = (total_s_both / total_s_cnt * 100.0) if total_s_cnt > 0 else 0.0

    total_s_either = sum(r.get_step_either_hits(primary_window) for r in eval_results)
    micro_s_either = (total_s_either / total_s_cnt * 100.0) if total_s_cnt > 0 else 0.0

    total_s_start = sum(r.get_step_start_hits(primary_window) for r in eval_results)
    micro_s_start = (total_s_start / total_s_cnt * 100.0) if total_s_cnt > 0 else 0.0

    total_s_end = sum(r.get_step_end_hits(primary_window) for r in eval_results)
    micro_s_end = (total_s_end / total_s_cnt * 100.0) if total_s_cnt > 0 else 0.0

    print(f"  • MỨC RANH GIỚI THỜI GIAN (Boundary-level | Window = ±{primary_window}s):")
    print(f"    - Macro Recall (trung bình các CĐ)       : {macro_b_rec:6.2f}%")
    print(f"    - Micro Recall (toàn bộ các mốc gộp lại) : {micro_b_rec:6.2f}% ({total_b_hits}/{total_b_gt} mốc khớp)")
    print(f"    - Macro Precision                        : {macro_b_prec:6.2f}%")
    print(f"    - Micro Precision                        : {micro_b_prec:6.2f}% ({total_pred_hits}/{total_pred_b} mốc pred khớp)")

    print(f"\n  • MỨC THAO TÁC CÔNG VIỆC (Step-level | Tổng số thao tác = {total_s_cnt}):")
    print(f"    - Macro Khớp CẢ 2 ĐẦU (Start & End)      : {macro_s_both:6.2f}%")
    print(f"    - Micro Khớp CẢ 2 ĐẦU (Start & End)      : {micro_s_both:6.2f}% ({total_s_both}/{total_s_cnt} thao tác)")
    print(f"    - Macro Khớp ÍT NHẤT 1 ĐẦU (Start hoặc End): {macro_s_either:6.2f}%")
    print(f"    - Micro Khớp ÍT NHẤT 1 ĐẦU (Start hoặc End): {micro_s_either:6.2f}% ({total_s_either}/{total_s_cnt} thao tác)")
    print(f"    - Micro Khớp mốc bắt đầu (Start Hit)     : {micro_s_start:6.2f}% ({total_s_start}/{total_s_cnt} thao tác)")
    print(f"    - Micro Khớp mốc kết thúc (End Hit)      : {micro_s_end:6.2f}% ({total_s_end}/{total_s_cnt} thao tác)")

    # =========================================================================
    # BẢNG 3: QUÉT ĐỘ NHẠY TOLERANCE WINDOW (WINDOW SWEEP)
    # =========================================================================
    print(f"\n[PHẦN 3] BIỂU ĐỒ & BẢNG ĐỘ NHẠY KHI THAY ĐỔI TOLERANCE WINDOW (Window Sweep):")
    print("-" * 115)
    sw_header = (f"{'Window':<8} | {'Boundary Recall (Macro / Micro)':<33} | {'Step Khớp 2 Đầu (Micro)':<24} "
                 f"| {'Step Khớp ≥1 Đầu (Micro)':<24} | {'Visual Macro Recall'}")
    print(sw_header)
    print("-" * 115)

    sweep_records: list[dict[str, Any]] = []

    for w in windows:
        # Macro & Micro boundary recall
        m_b_rec = sum(r.get_boundary_recall(w) for r in eval_results) / len(eval_results)
        w_b_hits = sum(r.get_boundary_hits(w) for r in eval_results)
        u_b_rec = (w_b_hits / total_b_gt * 100.0) if total_b_gt > 0 else 0.0

        # Step micro
        w_s_both = sum(r.get_step_both_hits(w) for r in eval_results)
        u_s_both = (w_s_both / total_s_cnt * 100.0) if total_s_cnt > 0 else 0.0

        w_s_either = sum(r.get_step_either_hits(w) for r in eval_results)
        u_s_either = (w_s_either / total_s_cnt * 100.0) if total_s_cnt > 0 else 0.0

        # Macro precision
        m_b_prec = sum(r.get_boundary_precision(w) for r in eval_results) / len(eval_results)

        bar = make_ascii_bar(m_b_rec, width=16)
        marker = " (*)" if w == primary_window else ""

        col_b_rec = f"{m_b_rec:5.1f}% / {u_b_rec:5.1f}% ({w_b_hits:2d}/{total_b_gt:2d})"
        col_s_both = f"{u_s_both:5.1f}% ({w_s_both:2d}/{total_s_cnt:2d})"
        col_s_either = f"{u_s_either:5.1f}% ({w_s_either:2d}/{total_s_cnt:2d})"

        print(f"±{w:<5.2f}s | {col_b_rec:<33} | {col_s_both:<24} | {col_s_either:<24} | {bar}{marker}")

        sweep_records.append({
            "window_s": w,
            "macro_boundary_recall_pct": round(m_b_rec, 2),
            "micro_boundary_recall_pct": round(u_b_rec, 2),
            "total_boundary_hits": w_b_hits,
            "total_gt_boundaries": total_b_gt,
            "micro_step_both_pct": round(u_s_both, 2),
            "micro_step_either_pct": round(u_s_either, 2),
            "macro_precision_pct": round(m_b_prec, 2),
        })

    print("-" * 115)
    print(" (*) Cửa sổ chuẩn mặc định")

    if skipped:
        print(f"\nCó {len(skipped)} công đoạn trong data/ chưa chạy Stage 1:")
        for cd, vf, reason in skipped:
            print(f"  • [CĐ {cd:2s}] {vf}")
        print("  => Chạy lệnh sau để tính tiếp các công đoạn này: python pipeline.py segment --all-data --visualize")

    print("=" * 115)

    # Save output JSON
    if args.out:
        out_path = Path(args.out)
        payload = {
            "primary_window": primary_window,
            "windows_evaluated": windows,
            "summary_overall": {
                "total_evaluated_videos": len(eval_results),
                "total_gt_boundaries": total_b_gt,
                "total_gt_steps": total_s_cnt,
                "primary_window_metrics": {
                    "macro_boundary_recall_pct": round(macro_b_rec, 2),
                    "micro_boundary_recall_pct": round(micro_b_rec, 2),
                    "macro_boundary_precision_pct": round(macro_b_prec, 2),
                    "micro_boundary_precision_pct": round(micro_b_prec, 2),
                    "macro_step_both_pct": round(macro_s_both, 2),
                    "micro_step_both_pct": round(micro_s_both, 2),
                    "macro_step_either_pct": round(macro_s_either, 2),
                    "micro_step_either_pct": round(micro_s_either, 2),
                },
                "sweep_table": sweep_records,
            },
            "per_cong_doan": [
                {
                    "cong_doan_id": r.cong_doan_id,
                    "sheet_title": r.sheet_title,
                    "video_file": r.video_file,
                    "total_boundaries": len(r.gt_boundaries),
                    "total_steps": r.get_total_steps(),
                    "metrics_at_primary_window": {
                        "boundary_hits": r.get_boundary_hits(primary_window),
                        "boundary_recall_pct": round(r.get_boundary_recall(primary_window), 2),
                        "boundary_precision_pct": round(r.get_boundary_precision(primary_window), 2),
                        "mae_seconds": round(r.get_boundary_mae(primary_window), 3),
                        "step_both_hits": r.get_step_both_hits(primary_window),
                        "step_both_pct": round(r.get_step_both_pct(primary_window), 2),
                        "step_either_hits": r.get_step_either_hits(primary_window),
                        "step_either_pct": round(r.get_step_either_pct(primary_window), 2),
                    },
                    "steps": [
                        {
                            "stt": s.stt,
                            "name": s.name,
                            "start_gt": s.start_gt,
                            "start_pred": s.start_pred,
                            "start_err": s.start_err,
                            "start_hit": s.start_hit,
                            "end_gt": s.end_gt,
                            "end_pred": s.end_pred,
                            "end_err": s.end_err,
                            "end_hit": s.end_hit,
                            "status": s.status_label,
                        }
                        for s in r.window_step_matches.get(primary_window, [])
                    ]
                }
                for r in eval_results
            ]
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nBáo cáo JSON đã được lưu tại: {out_path.resolve()}")


if __name__ == "__main__":
    main()
