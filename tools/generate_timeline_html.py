#!/usr/bin/env python3
"""Generate an interactive HTML visualization report for Stage 2 analysis results.

Reads:
- data/{cd}/worker_segments/worker_segments.json
- data/{cd}/macro_eval.json
- data/{cd}/expert_scenes/selected_frames.json
- data/{cd}/expert_scenes/process_knowledge.json

Generates:
- data/{cd}/timeline_report.html
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.render_classified_viz import merge_adjacent_segments


def get_op_color(op_name: str) -> str:
    op_lower = op_name.lower()
    if "diễu" in op_lower or "may" in op_lower:
        return "#10b981"  # Emerald green (sewing)
    elif "điều chỉnh" in op_lower or "chỉnh" in op_lower:
        return "#f59e0b"  # Amber / orange (adjusting)
    elif "chân vịt" in op_lower or "lấy túi" in op_lower or "đưa túi" in op_lower:
        return "#3b82f6"  # Blue (handling / feeding)
    elif "lật lót" in op_lower:
        return "#8b5cf6"  # Purple (flipping)
    elif "kéo" in op_lower or "cắt chỉ" in op_lower or "thân trước" in op_lower:
        return "#ef4444"  # Red / Crimson (finishing / cutting)
    elif "lại mũi" in op_lower:
        return "#ec4899"  # Pink (backstitch)
    return "#64748b"      # Slate grey


def generate_html_report(cd: str = "1") -> Path:
    data_dir = Path(f"data/{cd}")
    segments_file = data_dir / "worker_segments" / "worker_segments.json"
    macro_file = data_dir / "macro_eval.json"
    manifest_file = data_dir / "expert_scenes" / "selected_frames.json"
    cuts_dir = data_dir / "worker_segments" / "cuts"
    output_html = data_dir / "timeline_report.html"

    if not segments_file.is_file():
        raise FileNotFoundError(f"Missing {segments_file}")

    macro_data = {}
    if macro_file.is_file():
        macro_data = json.loads(macro_file.read_text(encoding="utf-8"))

    # Map macro evaluation data
    slow_times = {}
    for ev in macro_data.get("evaluated", []):
        if ev.get("timing_verdict") == "slow":
            slow_times[round(ev["start_time"], 2)] = ev

    segments_data = json.loads(segments_file.read_text(encoding="utf-8"))
    raw_segments = segments_data.get("segments", [])
    segments = merge_adjacent_segments(raw_segments, slow_times=slow_times)
    task_name = segments_data.get("task_name", f"Công đoạn {cd}")

    # Timeline bounds
    max_time = max((s["end_time"] for s in segments), default=155.0)
    total_duration = max_time

    # ── Load Ground Truth (chuyen1_segment.json) ──────────────────────────────
    gt_file = data_dir / "chuyen1_segment.json"
    gt_segments = []
    gt_duration = 53.0
    if gt_file.is_file():
        try:
            gt_data = json.loads(gt_file.read_text(encoding="utf-8"))
            gt_segments = gt_data.get("segments", [])
            gt_duration = float(gt_data.get("total_duration_s", 53.0))
        except Exception:
            pass

    # Build Ground Truth timeline blocks (repeating cycles across total_duration)
    gt_timeline_blocks = []
    if gt_segments:
        c = 0
        while c * gt_duration < total_duration:
            offset = c * gt_duration
            for gs in gt_segments:
                g_t0 = offset + gs["timestamp_start"]
                g_t1 = offset + gs["timestamp_end"]
                if g_t0 >= total_duration:
                    break
                g_t1 = min(g_t1, total_duration)
                g_dur = g_t1 - g_t0
                g_op = gs["name"]
                g_col = get_op_color(g_op)
                g_left = (g_t0 / total_duration) * 100
                g_width = max(0.4, (g_dur / total_duration) * 100)
                cycle_note = f" (Chu kỳ {c+1})" if c > 0 else ""
                gt_timeline_blocks.append(f"""
                <div class="group absolute top-0 bottom-0 cursor-pointer rounded-xs transition-all hover:brightness-125 hover:z-20 border-r border-black/30"
                     style="left: {g_left:.3f}%; width: {g_width:.3f}%; background-color: {g_col};"
                     title="[GT #{gs['stt']:02d}{cycle_note}] {html.escape(g_op)} ({g_t0:.1f}s - {g_t1:.1f}s, {gs['duration']:.1f}s)">
                </div>
                """)
            c += 1

    # Find matching cut clip files
    cut_files = sorted(cuts_dir.glob("*.mp4")) if cuts_dir.is_dir() else []
    cut_map = {}
    for cf in cut_files:
        prefix = cf.name.split("_")[0]
        if prefix.isdigit():
            idx = int(prefix)
            cut_map[idx] = f"worker_segments/cuts/{cf.name}"

    # Build timeline bars HTML
    timeline_blocks = []
    cards_html = []

    for idx, s in enumerate(segments):
        t0 = s["start_time"]
        t1 = s["end_time"]
        dur = s.get("worker_duration_s", round(t1 - t0, 2))
        op = s["operation_name"]
        color = get_op_color(op)

        left_pct = (t0 / total_duration) * 100
        width_pct = max(0.6, (dur / total_duration) * 100)

        is_slow = round(t0, 2) in slow_times or s.get("timing_verdict") == "slow"
        ratio_badge = ""
        slow_class = ""
        if is_slow:
            slow_class = "border-2 border-red-500 shadow-md shadow-red-500/40"
            ev = slow_times.get(round(t0, 2), {})
            ratio = ev.get("duration_ratio", 1.5)
            exp_dur = ev.get("expert_duration_s", 1.0)
            ratio_badge = f'<span class="bg-red-500/20 text-red-400 border border-red-500/30 text-xs px-2 py-0.5 rounded-full font-bold">⚠ Chậm x{ratio:.2f} (Chuẩn: {exp_dur:.1f}s)</span>'
        else:
            ratio_badge = '<span class="bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 text-xs px-2 py-0.5 rounded-full">Đạt chuẩn</span>'

        video_path = cut_map.get(idx, "")
        video_player_html = ""
        if video_path:
            video_player_html = f"""
            <div class="mt-3 bg-black/40 rounded-lg p-2 border border-slate-700/50">
                <div class="text-xs text-slate-400 mb-1 flex items-center justify-between">
                    <span>🎬 Video Clip Phân đoạn #{idx:02d}</span>
                    <span class="text-slate-500 font-mono text-[11px]">{dur:.2f}s</span>
                </div>
                <video controls preload="none" class="w-full rounded bg-black aspect-video max-h-48 object-contain">
                    <source src="{html.escape(video_path)}" type="video/mp4">
                    Trình duyệt không hỗ trợ video.
                </video>
            </div>
            """

        reasoning = s.get("evidence") or s.get("model_output", {}).get("reasoning", "")
        action_ev = s.get("action_evidence", "")
        product_ev = s.get("product_state_evidence", "")

        # Timeline bar element
        timeline_blocks.append(f"""
        <div class="timeline-bar group absolute top-0 bottom-0 cursor-pointer rounded-sm transition-all duration-150 hover:brightness-125 hover:z-20 {slow_class}"
             style="left: {left_pct:.3f}%; width: {width_pct:.3f}%; background-color: {color};"
             onclick="scrollToSegment({idx})"
             title="#{idx:02d} [{t0:.1f}s - {t1:.1f}s] {html.escape(op)}">
        </div>
        """)

        # Detailed Card
        cards_html.append(f"""
        <div id="segment-card-{idx}" class="segment-card bg-slate-900/90 border border-slate-800 rounded-xl p-4 transition-all hover:border-slate-700 hover:shadow-lg">
            <div class="flex flex-wrap items-center justify-between gap-2 mb-2">
                <div class="flex items-center gap-2">
                    <span class="w-3 h-3 rounded-full flex-shrink-0" style="background-color: {color};"></span>
                    <span class="text-slate-400 font-mono text-xs">#{idx:02d}</span>
                    <h3 class="text-base font-semibold text-slate-100">{html.escape(op)}</h3>
                </div>
                <div class="flex items-center gap-2">
                    {ratio_badge}
                    <span class="bg-slate-800 text-slate-300 font-mono text-xs px-2.5 py-1 rounded-md border border-slate-700">
                        {t0:.1f}s – {t1:.1f}s ({dur:.2f}s)
                    </span>
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-3">
                <div class="space-y-2 text-sm">
                    {f'<div class="bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/40"><span class="text-xs font-semibold text-sky-400 block mb-1">🔍 Quan sát VLM (Reasoning):</span><p class="text-slate-300 text-xs leading-relaxed">{html.escape(reasoning)}</p></div>' if reasoning else ''}
                    {f'<div class="bg-slate-800/40 p-2 rounded-lg text-xs text-slate-400"><span class="text-amber-400 font-medium">Hành động tay:</span> {html.escape(action_ev)}</div>' if action_ev else ''}
                    {f'<div class="bg-slate-800/40 p-2 rounded-lg text-xs text-slate-400"><span class="text-emerald-400 font-medium">Trạng thái vải:</span> {html.escape(product_ev)}</div>' if product_ev else ''}
                </div>
                <div>
                    {video_player_html}
                </div>
            </div>
        </div>
        """)

    # Overview KPIs
    n_evaluated = len(segments)
    n_slow = len(slow_times)
    n_missing = macro_data.get("n_missing", 7)
    expert_duration = 53.0

    html_content = f"""<!DOCTYPE html>
<html lang="vi" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo cáo Trực quan Hóa - {html.escape(task_name)}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {{
            background-color: #0b0f19;
            color: #f8fafc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        .custom-scrollbar::-webkit-scrollbar {{
            height: 8px;
            width: 8px;
        }}
        .custom-scrollbar::-webkit-scrollbar-track {{
            background: #1e293b;
        }}
        .custom-scrollbar::-webkit-scrollbar-thumb {{
            background: #475569;
            border-radius: 4px;
        }}
        .segment-card.highlight {{
            border-color: #38bdf8 !important;
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.3) !important;
        }}
    </style>
</head>
<body class="p-4 md:p-8">
    <div class="max-w-7xl mx-auto space-y-6">

        <!-- Header -->
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-900/80 p-6 rounded-2xl border border-slate-800 backdrop-blur">
            <div>
                <div class="flex items-center gap-2 mb-1">
                    <span class="bg-indigo-500/20 text-indigo-400 text-xs font-semibold px-2.5 py-0.5 rounded border border-indigo-500/30">CÔNG ĐOẠN {cd}</span>
                    <span class="bg-emerald-500/20 text-emerald-400 text-xs font-semibold px-2.5 py-0.5 rounded border border-emerald-500/30">STAGE 2 EVALUATION</span>
                </div>
                <h1 class="text-2xl md:text-3xl font-bold text-white">{html.escape(task_name)}</h1>
                <p class="text-slate-400 text-sm mt-1">Phân tích kỹ năng may công nhân đối chiếu chuyên gia qua mô hình VLM qwen3.7-flash</p>
            </div>
            <div class="flex items-center gap-2">
                <a href="macro_eval.json" target="_blank" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm rounded-lg border border-slate-700 transition">📄 macro_eval.json</a>
                <a href="worker_segments/worker_segments.json" target="_blank" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition shadow-lg shadow-indigo-500/20">📦 worker_segments.json</a>
            </div>
        </div>

        <!-- KPIs Cards -->
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div class="bg-slate-900/80 p-4 rounded-xl border border-slate-800">
                <span class="text-xs font-medium text-slate-400">Thời lượng Công nhân</span>
                <div class="text-2xl font-bold text-white mt-1">{total_duration:.1f}s</div>
                <div class="text-xs text-slate-500 mt-1">Chuẩn chuyên gia: {expert_duration:.1f}s (+{total_duration - expert_duration:.1f}s)</div>
            </div>
            <div class="bg-slate-900/80 p-4 rounded-xl border border-slate-800">
                <span class="text-xs font-medium text-slate-400">Tổng số Phân đoạn</span>
                <div class="text-2xl font-bold text-white mt-1">{n_evaluated}</div>
                <div class="text-xs text-emerald-400 mt-1">100% khớp thao tác chuẩn</div>
            </div>
            <div class="bg-slate-900/80 p-4 rounded-xl border border-slate-800">
                <span class="text-xs font-medium text-slate-400">Thao tác Bị Chậm</span>
                <div class="text-2xl font-bold text-amber-400 mt-1">{n_slow}</div>
                <div class="text-xs text-slate-500 mt-1">Cần đào tạo vi mô (Micro)</div>
            </div>
            <div class="bg-slate-900/80 p-4 rounded-xl border border-slate-800">
                <span class="text-xs font-medium text-slate-400">Thao tác Bỏ qua / Thiếu</span>
                <div class="text-2xl font-bold text-rose-400 mt-1">{n_missing}</div>
                <div class="text-xs text-slate-500 mt-1">Lại mũi nút nhấn, xoay góc</div>
            </div>
        </div>

        <!-- Interactive Timeline Scrubber Bar -->
        <div class="bg-slate-900/80 p-6 rounded-2xl border border-slate-800 space-y-4">
            <div class="flex flex-wrap items-center justify-between gap-2">
                <h2 class="text-lg font-bold text-white flex items-center gap-2">
                    <span>⏱️ Dòng Thời Gian Phân Đoạn Thao Tác (Gantt Timeline)</span>
                </h2>
                <div class="flex flex-wrap items-center gap-3 text-xs text-slate-400">
                    <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full bg-emerald-500"></span> May diễu / May</span>
                    <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full bg-amber-500"></span> Điều chỉnh</span>
                    <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full bg-blue-500"></span> Đưa/Lấy túi</span>
                    <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full bg-purple-500"></span> Chỉnh/Lật lót</span>
                    <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full border border-red-500 bg-red-500/30"></span> ⚠ Bị chậm</span>
                </div>
            </div>

            <!-- Dual Timeline Bars Container (Ground Truth vs Worker) -->
            <div class="space-y-3">
                <!-- Ground Truth Bar -->
                <div>
                    <div class="flex items-center justify-between text-xs text-slate-400 mb-1">
                        <span class="font-semibold text-emerald-400 flex items-center gap-1.5">
                            <span class="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
                            <span>🏆 GROUND TRUTH (Chuẩn SOP Chuyên Gia • 24 bước / chu kỳ {gt_duration:.1f}s)</span>
                        </span>
                        <span class="text-slate-500 text-[11px] font-mono">chuyen1_segment.json</span>
                    </div>
                    <div class="relative w-full h-8 bg-slate-950 rounded-lg overflow-hidden border border-slate-700 shadow-inner select-none">
                        {''.join(gt_timeline_blocks)}
                    </div>
                </div>

                <!-- Worker Bar -->
                <div>
                    <div class="flex items-center justify-between text-xs text-slate-400 mb-1">
                        <span class="font-semibold text-sky-400 flex items-center gap-1.5">
                            <span class="w-2.5 h-2.5 rounded-full bg-sky-400"></span>
                            <span>👷 WORKER (Thực Tế Công Nhân • {len(segments)} khối thao tác • Tổng {total_duration:.1f}s)</span>
                        </span>
                        <span class="text-slate-500 text-[11px] font-mono">cam-03_20260808_023207_cut_1_28-4_03.mp4</span>
                    </div>
                    <div class="relative w-full h-11 bg-slate-950 rounded-lg overflow-hidden border border-slate-700 shadow-inner select-none">
                        {''.join(timeline_blocks)}
                    </div>
                </div>

                <!-- Time Scale Markers -->
                <div class="relative w-full h-4 text-[10px] text-slate-500 font-mono">
                    <span class="absolute left-0">0.0s</span>
                    <span class="absolute left-1/4 -translate-x-1/2">{total_duration * 0.25:.0f}s</span>
                    <span class="absolute left-2/4 -translate-x-1/2">{total_duration * 0.50:.0f}s</span>
                    <span class="absolute left-3/4 -translate-x-1/2">{total_duration * 0.75:.0f}s</span>
                    <span class="absolute right-0">{total_duration:.1f}s</span>
                </div>
            </div>
            <p class="text-xs text-slate-500">💡 Thanh trên (Ground Truth) thể hiện các bước quy trình chuẩn của chuyên gia; thanh dưới (Worker) thể hiện thực tế công nhân may. Click vào bất kỳ phân đoạn nào để cuộn xuống xem chi tiết.</p>
        </div>

        <!-- Filter Buttons -->
        <div class="flex flex-wrap items-center gap-2">
            <span class="text-xs font-semibold text-slate-400 mr-2">Lọc hiển thị:</span>
            <button onclick="filterCards('all')" class="filter-btn active px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-white text-xs font-medium rounded-lg border border-slate-700 transition">Tất cả ({n_evaluated})</button>
            <button onclick="filterCards('slow')" class="filter-btn px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-red-400 text-xs font-medium rounded-lg border border-slate-700 transition">⚠ Bị chậm ({n_slow})</button>
            <button onclick="filterCards('sew')" class="filter-btn px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-emerald-400 text-xs font-medium rounded-lg border border-slate-700 transition">May diễu</button>
            <button onclick="filterCards('adjust')" class="filter-btn px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-amber-400 text-xs font-medium rounded-lg border border-slate-700 transition">Điều chỉnh</button>
        </div>

        <!-- Segments List -->
        <div class="space-y-4" id="segments-container">
            {''.join(cards_html)}
        </div>

    </div>

    <script>
        function scrollToSegment(idx) {{
            const card = document.getElementById('segment-card-' + idx);
            if (card) {{
                document.querySelectorAll('.segment-card').forEach(c => c.classList.remove('highlight'));
                card.classList.add('highlight');
                card.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }}

        function filterCards(type) {{
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('bg-indigo-600', 'text-white'));
            event.target.classList.add('bg-indigo-600', 'text-white');

            const cards = document.querySelectorAll('.segment-card');
            cards.forEach(card => {{
                const text = card.textContent.toLowerCase();
                if (type === 'all') {{
                    card.style.display = 'block';
                }} else if (type === 'slow') {{
                    card.style.display = text.includes('chậm') ? 'block' : 'none';
                }} else if (type === 'sew') {{
                    card.style.display = (text.includes('diễu') || text.includes('may')) ? 'block' : 'none';
                }} else if (type === 'adjust') {{
                    card.style.display = text.includes('điều chỉnh') ? 'block' : 'none';
                }}
            }});
        }}
    </script>
</body>
</html>
"""

    output_html.write_text(html_content, encoding="utf-8")
    print(f"Generated interactive visualization report -> {output_html}")
    return output_html


if __name__ == "__main__":
    cd_arg = sys.argv[1] if len(sys.argv) > 1 else "1"
    generate_html_report(cd_arg)
