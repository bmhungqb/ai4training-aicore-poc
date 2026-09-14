#!/usr/bin/env python3
"""Visualize selected frames of each expert operation (scene) with ROI mask applied.

Reads:
- data/{cd}/expert_scenes/selected_frames.json
- data/{cd}/expert.json
- ROI mask image (e.g. data/{cd}/cam-03_...mask.png or expert.mask.png)

Outputs:
- data/{cd}/expert_scenes/frames_masked/scene_{idx:02d}/... (Cropped & masked frame images)
- data/{cd}/expert_scenes/grids/scene_{idx:02d}_grid.jpg (Grid images with mask applied)
- data/{cd}/expert_scenes_grid.html (Interactive dashboard with Masked vs Raw toggle)

Usage:
    python tools/visualize_expert_grid.py [cong_doan] [--raw] [--no-images]
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidate_fonts = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for fp in candidate_fonts:
        if os.path.isfile(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


def calculate_sharpness(img_gray: np.ndarray) -> float:
    """Calculate variance of Laplacian as a proxy for frame sharpness."""
    if img_gray is None or img_gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def apply_roi_mask(
    img_bgr: np.ndarray,
    mask_gray: np.ndarray | None,
    crop_bbox: bool = True,
) -> np.ndarray:
    """Apply ROI mask to image: zero out background outside mask, crop to mask bounding box."""
    if mask_gray is None:
        return img_bgr

    h, w = img_bgr.shape[:2]
    if mask_gray.shape[:2] != (h, w):
        mask_gray = cv2.resize(mask_gray, (w, h), interpolation=cv2.INTER_NEAREST)

    bin_mask = (mask_gray > 127).astype(np.uint8)
    masked = cv2.bitwise_and(img_bgr, img_bgr, mask=bin_mask)

    if crop_bbox:
        ys, xs = np.where(bin_mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())
            masked = masked[y1:y2 + 1, x1:x2 + 1]

    return masked


def render_scene_grid_image(
    scene_idx: int,
    op_name: str,
    t0: float,
    t1: float,
    frame_paths: list[Path],
    out_img_path: Path,
    mask_gray: np.ndarray | None = None,
    target_width: int = 480,
    target_height: int = 270,
) -> Path:
    """Render a clean grid image showing all selected frames of a single scene."""
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = len(frame_paths)
    if n_frames == 0:
        return out_img_path

    # Determine layout: 1 row if <= 4 frames, else 2 rows
    if n_frames <= 4:
        cols = n_frames
        rows = 1
    elif n_frames <= 8:
        cols = math.ceil(n_frames / 2)
        rows = 2
    else:
        cols = 4
        rows = math.ceil(n_frames / cols)

    pad = 12
    header_height = 74
    card_info_height = 42

    cell_w = target_width
    cell_h = target_height + card_info_height

    canvas_w = pad + cols * (cell_w + pad)
    canvas_h = header_height + pad + rows * (cell_h + pad)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(15, 23, 42))  # Slate-900
    draw = ImageDraw.Draw(canvas)

    font_title = get_font(21)
    font_sub = get_font(14)
    font_card = get_font(13)
    font_badge = get_font(12)

    # Header
    draw.rectangle([(0, 0), (canvas_w, header_height)], fill=(30, 41, 59))  # Slate-800
    mask_tag = " [ROI MASK APPLIED]" if mask_gray is not None else ""
    header_title = f"Thao tác #{scene_idx:02d}: {op_name}{mask_tag}"
    header_sub = f"Khoảng thời gian: {t0:.2f}s – {t1:.2f}s (Thời lượng: {t1 - t0:.2f}s) | Số frame chọn: {n_frames}"
    draw.text((pad + 6, 12), header_title, fill=(248, 250, 252), font=font_title)
    draw.text((pad + 6, 44), header_sub, fill=(148, 163, 184), font=font_sub)

    # Draw frames in grid
    for idx, fp in enumerate(frame_paths):
        r = idx // cols
        c = idx % cols

        x0 = pad + c * (cell_w + pad)
        y0 = header_height + pad + r * (cell_h + pad)

        # Load frame
        bgr = cv2.imread(str(fp))
        if bgr is not None:
            if mask_gray is not None:
                bgr = apply_roi_mask(bgr, mask_gray, crop_bbox=True)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            sharpness = calculate_sharpness(gray)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(rgb).resize((cell_w, target_height), Image.Resampling.BILINEAR)
            canvas.paste(frame_pil, (x0, y0))
        else:
            sharpness = 0.0
            draw.rectangle([(x0, y0), (x0 + cell_w, y0 + target_height)], fill=(51, 65, 85))
            draw.text((x0 + 10, y0 + 10), f"Missing: {fp.name}", fill=(239, 68, 68), font=font_card)

        # Draw frame border
        draw.rectangle([(x0, y0), (x0 + cell_w, y0 + target_height)], outline=(71, 85, 105), width=2)

        # Card info footer
        info_y0 = y0 + target_height
        draw.rectangle([(x0, info_y0), (x0 + cell_w, info_y0 + card_info_height)], fill=(30, 41, 59))
        draw.rectangle([(x0, info_y0), (x0 + cell_w, info_y0 + card_info_height)], outline=(71, 85, 105), width=1)

        # Name tag & sharpness
        name = fp.stem
        tag = f"Frame {idx + 1}/{n_frames} ({name})"
        sharp_tag = f"Nét: {sharpness:.1f}"

        draw.text((x0 + 8, info_y0 + 10), tag, fill=(226, 232, 240), font=font_card)
        badge_col = (16, 185, 129) if sharpness >= 400 else (245, 158, 11) if sharpness >= 150 else (239, 68, 68)
        draw.text((x0 + cell_w - 95, info_y0 + 10), sharp_tag, fill=badge_col, font=font_badge)

    canvas.save(out_img_path, quality=92)
    return out_img_path


def generate_html_viewer(
    task_name: str,
    scenes_data: list[dict],
    out_html_path: Path,
    has_mask: bool = True,
) -> Path:
    """Generate an interactive HTML visualizer with toggle between Masked and Raw frames."""
    out_html_path.parent.mkdir(parents=True, exist_ok=True)

    total_frames = sum(len(s["frames"]) for s in scenes_data)
    total_duration = max((s["timestamp_end"] for s in scenes_data), default=0.0)

    scenes_cards_html = []
    for s in scenes_data:
        idx = s["scene_index"]
        op_name = s["operation_name"]
        t0 = s["timestamp_start"]
        t1 = s["timestamp_end"]
        dur = t1 - t0
        frames = s["frames"]
        guideline = s.get("guideline", {})
        grid_img_rel = s.get("grid_img_rel", "")

        # Frames HTML
        frame_items_html = []
        for f_order, f in enumerate(frames, 1):
            raw_rel = f["raw_rel_path"]
            masked_rel = f["masked_rel_path"]
            initial_src = masked_rel if has_mask else raw_rel
            sharp = f["sharpness"]
            sharp_class = "text-emerald-400 bg-emerald-500/10 border-emerald-500/30" if sharp >= 400 else (
                "text-amber-400 bg-amber-500/10 border-amber-500/30" if sharp >= 150 else "text-rose-400 bg-rose-500/10 border-rose-500/30"
            )

            frame_items_html.append(f"""
            <div class="group relative bg-slate-900 border border-slate-800 rounded-lg overflow-hidden hover:border-sky-500/60 transition-all hover:shadow-lg">
                <div class="aspect-video bg-black/60 overflow-hidden cursor-pointer" onclick="openLightbox(this.querySelector('img').src, '#{idx:02d} {html.escape(op_name)} - Frame {f_order}')">
                    <img src="{initial_src}" data-masked="{masked_rel}" data-raw="{raw_rel}" alt="{f['name']}" loading="lazy" class="frame-img w-full h-full object-contain transition-transform duration-200 group-hover:scale-105">
                </div>
                <div class="p-2.5 flex items-center justify-between text-xs bg-slate-900/90 border-t border-slate-800">
                    <div>
                        <span class="font-mono text-slate-300 font-semibold">{f['name']}</span>
                        <span class="text-slate-500 block text-[11px]">Thứ tự: {f_order}/{len(frames)}</span>
                    </div>
                    <span class="px-2 py-0.5 rounded border text-[11px] font-mono {sharp_class}" title="Laplacian Sharpness Variance">
                        ⚡ {sharp:.1f}
                    </span>
                </div>
            </div>
            """)

        # Guideline snippet
        gl_html = ""
        if guideline:
            desc = guideline.get("operation_description", "")
            steps = guideline.get("how_to_steps", [])
            steps_html = "".join(f"<li class='text-xs text-slate-300'>{html.escape(st)}</li>" for st in steps)
            gl_html = f"""
            <div class="mt-3 p-3 bg-slate-950/60 rounded-lg border border-slate-800/80 text-xs text-slate-400 space-y-1.5">
                {f'<p><strong class="text-sky-400">Mô tả:</strong> {html.escape(desc)}</p>' if desc else ''}
                {f'<div><strong class="text-slate-300 block mb-1">Các bước SOP:</strong><ol class="list-decimal list-inside space-y-0.5">{steps_html}</ol></div>' if steps else ''}
            </div>
            """

        scenes_cards_html.append(f"""
        <div class="scene-card bg-slate-900/80 border border-slate-800 rounded-xl p-5 hover:border-slate-700 transition-all" data-op="{html.escape(op_name.lower())}">
            <div class="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800">
                <div class="flex items-center gap-3">
                    <span class="bg-sky-500/20 text-sky-400 font-mono text-xs px-2.5 py-1 rounded-md border border-sky-500/30 font-bold">
                        SCENE #{idx:02d}
                    </span>
                    <h2 class="text-lg font-bold text-white">{html.escape(op_name)}</h2>
                </div>
                <div class="flex items-center gap-2 text-xs">
                    <span class="bg-slate-800 text-slate-300 font-mono px-2.5 py-1 rounded-md border border-slate-700">
                        ⏱ {t0:.2f}s – {t1:.2f}s ({dur:.2f}s)
                    </span>
                    <span class="bg-indigo-500/20 text-indigo-300 font-mono px-2.5 py-1 rounded-md border border-indigo-500/30">
                        📷 {len(frames)} frames
                    </span>
                    {f'<a href="{grid_img_rel}" target="_blank" class="bg-slate-800 hover:bg-slate-700 text-sky-400 px-2.5 py-1 rounded-md border border-slate-700 transition">🖼 Mở ảnh Grid</a>' if grid_img_rel else ''}
                </div>
            </div>

            {gl_html}

            <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 mt-4">
                {"".join(frame_items_html)}
            </div>
        </div>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="vi" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualizer Selected Frames (Masked) - {html.escape(task_name)}</title>
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
    </style>
</head>
<body class="p-4 md:p-8">
    <div class="max-w-7xl mx-auto space-y-6">

        <!-- Header -->
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-900/90 p-6 rounded-2xl border border-slate-800 backdrop-blur shadow-xl">
            <div>
                <div class="flex items-center gap-2 mb-1">
                    <span class="bg-indigo-500/20 text-indigo-400 text-xs font-semibold px-2.5 py-0.5 rounded border border-indigo-500/30">STAGE 2 (EXPERT ANALYSIS)</span>
                    <span class="bg-purple-500/20 text-purple-400 text-xs font-semibold px-2.5 py-0.5 rounded border border-purple-500/30">ROI MASK APPLIED</span>
                </div>
                <h1 class="text-2xl md:text-3xl font-bold text-white">{html.escape(task_name)}</h1>
                <p class="text-slate-400 text-xs md:text-sm mt-1">Trực quan hóa Grid các khung hình tham chiếu đã áp dụng <strong>Mặt nạ ROI (Worker Mask)</strong> loại bỏ nền và công nhân bên cạnh.</p>
            </div>
            <div class="flex flex-wrap items-center gap-3">
                <div class="bg-slate-800/80 px-4 py-2 rounded-xl border border-slate-700 text-center">
                    <span class="text-xs text-slate-400 block">Tổng số Cảnh</span>
                    <span class="text-lg font-bold font-mono text-white">{len(scenes_data)}</span>
                </div>
                <div class="bg-slate-800/80 px-4 py-2 rounded-xl border border-slate-700 text-center">
                    <span class="text-xs text-slate-400 block">Tổng Frames</span>
                    <span class="text-lg font-bold font-mono text-emerald-400">{total_frames}</span>
                </div>
                <div class="bg-slate-800/80 px-4 py-2 rounded-xl border border-slate-700 text-center">
                    <span class="text-xs text-slate-400 block">Thời lượng Chuẩn</span>
                    <span class="text-lg font-bold font-mono text-sky-400">{total_duration:.1f}s</span>
                </div>
            </div>
        </div>

        <!-- Controls Bar: Filter + Mask Toggle -->
        <div class="bg-slate-900/70 p-4 rounded-xl border border-slate-800 flex flex-wrap items-center justify-between gap-4">
            <div class="flex items-center gap-3 flex-1 min-w-[260px]">
                <span class="text-slate-400 text-sm">🔍 Tìm thao tác:</span>
                <input type="text" id="searchInput" placeholder="Nhập tên thao tác (vd: 'diễu', 'lót', 'kéo')..." 
                       class="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-sky-500">
                <span id="matchCount" class="text-xs text-slate-400 font-mono whitespace-nowrap">Hiển thị {len(scenes_data)}/{len(scenes_data)} cảnh</span>
            </div>

            <!-- View Toggle Button -->
            <div class="flex items-center gap-2 bg-slate-950 p-1 rounded-lg border border-slate-800">
                <button id="btnMasked" onclick="setViewMode('masked')" class="px-3 py-1.5 rounded-md text-xs font-semibold transition bg-indigo-600 text-white shadow">
                    🎭 Đã Che Mask (VLM View)
                </button>
                <button id="btnRaw" onclick="setViewMode('raw')" class="px-3 py-1.5 rounded-md text-xs font-semibold text-slate-400 hover:text-white transition">
                    📷 Ảnh Gốc (Raw)
                </button>
            </div>
        </div>

        <!-- Scenes Cards List -->
        <div id="scenesContainer" class="space-y-5">
            {"".join(scenes_cards_html)}
        </div>

    </div>

    <!-- Lightbox Modal -->
    <div id="lightbox" class="fixed inset-0 bg-black/90 z-50 hidden flex flex-col items-center justify-center p-4" onclick="closeLightbox()">
        <div class="max-w-5xl w-full flex flex-col items-center gap-3" onclick="event.stopPropagation()">
            <div class="flex items-center justify-between w-full text-white text-sm">
                <span id="lightboxTitle" class="font-semibold text-slate-200"></span>
                <button onclick="closeLightbox()" class="text-slate-400 hover:text-white text-lg font-bold px-3 py-1">✕ Đóng</button>
            </div>
            <img id="lightboxImg" src="" class="max-h-[85vh] max-w-full rounded-lg border border-slate-700 object-contain shadow-2xl">
        </div>
    </div>

    <script>
        let currentMode = 'masked';

        function setViewMode(mode) {{
            currentMode = mode;
            const btnMasked = document.getElementById('btnMasked');
            const btnRaw = document.getElementById('btnRaw');
            const images = document.querySelectorAll('.frame-img');

            if (mode === 'masked') {{
                btnMasked.className = "px-3 py-1.5 rounded-md text-xs font-semibold transition bg-indigo-600 text-white shadow";
                btnRaw.className = "px-3 py-1.5 rounded-md text-xs font-semibold text-slate-400 hover:text-white transition";
                images.forEach(img => {{
                    const masked = img.getAttribute('data-masked');
                    if (masked) img.src = masked;
                }});
            }} else {{
                btnRaw.className = "px-3 py-1.5 rounded-md text-xs font-semibold transition bg-indigo-600 text-white shadow";
                btnMasked.className = "px-3 py-1.5 rounded-md text-xs font-semibold text-slate-400 hover:text-white transition";
                images.forEach(img => {{
                    const raw = img.getAttribute('data-raw');
                    if (raw) img.src = raw;
                }});
            }}
        }}

        function openLightbox(src, title) {{
            document.getElementById('lightboxImg').src = src;
            document.getElementById('lightboxTitle').innerText = title;
            document.getElementById('lightbox').classList.remove('hidden');
        }}

        function closeLightbox() {{
            document.getElementById('lightbox').classList.add('hidden');
        }}

        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') closeLightbox();
        }});

        // Live Search Filter
        const searchInput = document.getElementById('searchInput');
        const cards = document.querySelectorAll('.scene-card');
        const matchCount = document.getElementById('matchCount');

        searchInput.addEventListener('input', (e) => {{
            const query = e.target.value.toLowerCase().trim();
            let visible = 0;
            cards.forEach(card => {{
                const op = card.getAttribute('data-op');
                if (!query || op.includes(query)) {{
                    card.style.display = '';
                    visible++;
                }} else {{
                    card.style.display = 'none';
                }}
            }});
            matchCount.innerText = `Hiển thị ${{visible}}/${{cards.length}} cảnh`;
        }});
    </script>
</body>
</html>
"""
    out_html_path.write_text(html_content, encoding="utf-8")
    return out_html_path


def main():
    parser = argparse.ArgumentParser(description="Visualize grid selected frames for expert scenes with mask.")
    parser.add_argument("cong_doan", nargs="?", default="1", help="Cong doan ID (default: 1)")
    parser.add_argument("--no-mask", action="store_true", help="Do not apply mask, render raw frames")
    parser.add_argument("--no-images", action="store_true", help="Skip rendering jpg grid images, only output HTML")
    args = parser.parse_args()

    cd = str(args.cong_doan)
    cd_dir = Path(f"data/{cd}")
    manifest_path = cd_dir / "expert_scenes" / "selected_frames.json"
    grids_dir = cd_dir / "expert_scenes" / "grids"
    masked_frames_dir = cd_dir / "expert_scenes" / "frames_masked"
    out_html_path = cd_dir / "expert_scenes_grid.html"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_name = manifest.get("task_name", f"Công đoạn {cd}")
    scenes = manifest.get("scenes", {})

    # Detect mask
    mask_gray = None
    if not args.no_mask:
        mask_candidates = [
            Path(manifest.get("mask_path", "")) if manifest.get("mask_path") else None,
            cd_dir / "cam-03_20260805_073527_cut_0_0-0_57.mask.png",
            cd_dir / "expert.mask.png",
        ]
        for mc in mask_candidates:
            if mc and mc.is_file():
                mask_gray = cv2.imread(str(mc), cv2.IMREAD_GRAYSCALE)
                print(f"[visualize] Found and applying ROI mask: {mc.name}")
                break

    print(f"=== Visualizing Selected Frames for {task_name} ===")
    print(f"Total scenes: {len(scenes)} | Mask applied: {mask_gray is not None}")

    scenes_data = []
    scene_keys = sorted(scenes.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k))

    for k in scene_keys:
        s = scenes[k]
        idx = int(k) if str(k).isdigit() else 0
        ops = s.get("operations", [])
        op_name = " + ".join(op.get("name", "") for op in ops) if ops else f"Scene {idx}"
        t0 = float(s.get("timestamp_start", 0.0))
        t1 = float(s.get("timestamp_end", 0.0))

        raw_frames = s.get("frames", [])
        frame_items = []
        raw_frame_paths = []

        scene_masked_dir = masked_frames_dir / f"scene_{idx:02d}"
        scene_masked_dir.mkdir(parents=True, exist_ok=True)

        for f_str in raw_frames:
            fp = Path(f_str)
            if not fp.is_absolute():
                fp = Path.cwd() / fp
            raw_frame_paths.append(fp)

            # Generate and save masked frame
            masked_fp = scene_masked_dir / fp.name
            bgr = cv2.imread(str(fp))
            if bgr is not None and mask_gray is not None:
                masked_bgr = apply_roi_mask(bgr, mask_gray, crop_bbox=True)
                cv2.imwrite(str(masked_fp), masked_bgr)
                sharpness = calculate_sharpness(cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY))
            elif bgr is not None:
                sharpness = calculate_sharpness(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
                masked_fp = fp
            else:
                sharpness = 0.0
                masked_fp = fp

            try:
                raw_rel = os.path.relpath(fp, cd_dir)
            except Exception:
                raw_rel = str(fp)

            try:
                masked_rel = os.path.relpath(masked_fp, cd_dir)
            except Exception:
                masked_rel = str(masked_fp)

            frame_items.append({
                "path": str(fp),
                "raw_rel_path": raw_rel,
                "masked_rel_path": masked_rel,
                "name": fp.name,
                "sharpness": sharpness,
            })

        grid_img_rel = ""
        if not args.no_images and raw_frame_paths:
            out_img = grids_dir / f"scene_{idx:02d}_grid.jpg"
            render_scene_grid_image(
                scene_idx=idx,
                op_name=op_name,
                t0=t0,
                t1=t1,
                frame_paths=raw_frame_paths,
                out_img_path=out_img,
                mask_gray=mask_gray,
            )
            try:
                grid_img_rel = os.path.relpath(out_img, cd_dir)
            except Exception:
                grid_img_rel = str(out_img)
            print(f"  [Scene {idx:02d}] {len(raw_frame_paths)} frames -> {out_img.name}")

        scenes_data.append({
            "scene_index": idx,
            "operation_name": op_name,
            "timestamp_start": t0,
            "timestamp_end": t1,
            "frames": frame_items,
            "guideline": s.get("guideline", {}),
            "grid_img_rel": grid_img_rel,
        })

    # Generate HTML
    generate_html_viewer(
        task_name=task_name,
        scenes_data=scenes_data,
        out_html_path=out_html_path,
        has_mask=mask_gray is not None,
    )
    print(f"\n[OK] Rendered HTML Visualizer with Mask -> {out_html_path}")
    print(f"[OK] Rendered {len(scenes_data)} masked scene grid images in {grids_dir}/")
    print(f"[OK] Saved cropped & masked frames to {masked_frames_dir}/")


if __name__ == "__main__":
    main()
