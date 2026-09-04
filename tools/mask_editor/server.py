#!/usr/bin/env python3
"""Mask editor server: browse videos under a folder, draw a ROI mask for each
one in the browser, save it as "<video_stem>.mask.png" next to the video.

That mask file is picked up automatically by src/segmentation/kinematic.py
(KinematicSegmenter) for both worker.mp4 and expert.mp4 — Step 1 (SAM3) and
Step 2 (SEA-RAFT) in src/kinematic_pipeline/ then restrict processing to the
masked region only, so another person visible in the same frame isn't picked
up as motion noise.

Usage (run on the server, then open the printed URL from your browser):

    python -m tools.mask_editor.server --dir data --port 8765

No third-party dependencies beyond what the rest of the repo already uses
(opencv-python, stdlib http.server).
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

INDEX_HTML_PATH = Path(__file__).resolve().parent / "index.html"


def _find_videos(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def _mask_path_for(video_path: Path) -> Path:
    return video_path.with_suffix("").with_suffix(".mask.png")


def _read_first_frame(video_path: Path) -> tuple[np.ndarray | None, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0, 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, width, height
    return frame, width, height


class MaskEditorHandler(BaseHTTPRequestHandler):
    root_dir: Path  # set by make_handler()

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _resolve_video(self, rel_path: str) -> Path | None:
        """Resolve a client-supplied relative path to a video under root_dir,
        rejecting anything that escapes it."""
        try:
            candidate = (self.root_dir / rel_path).resolve()
            candidate.relative_to(self.root_dir.resolve())
        except (ValueError, RuntimeError):
            return None
        if candidate.suffix.lower() not in VIDEO_EXTS or not candidate.exists():
            return None
        return candidate

    # -- GET --------------------------------------------------------------
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            html = INDEX_HTML_PATH.read_bytes()
            self._send_bytes(html, "text/html; charset=utf-8")
            return

        if parsed.path == "/api/videos":
            videos = []
            for p in _find_videos(self.root_dir):
                rel = str(p.relative_to(self.root_dir))
                _, w, h = _read_first_frame(p)
                videos.append({
                    "path": rel,
                    "width": w,
                    "height": h,
                    "has_mask": _mask_path_for(p).exists(),
                })
            self._send_json({"root": str(self.root_dir), "videos": videos})
            return

        if parsed.path == "/api/frame":
            qs = parse_qs(parsed.query)
            rel_path = unquote(qs.get("path", [""])[0])
            video_path = self._resolve_video(rel_path)
            if video_path is None:
                self._send_json({"error": "video not found"}, status=404)
                return
            frame, _, _ = _read_first_frame(video_path)
            if frame is None:
                self._send_json({"error": "could not read frame"}, status=500)
                return
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                self._send_json({"error": "could not encode frame"}, status=500)
                return
            self._send_bytes(buf.tobytes(), "image/jpeg")
            return

        if parsed.path == "/api/mask":
            qs = parse_qs(parsed.query)
            rel_path = unquote(qs.get("path", [""])[0])
            video_path = self._resolve_video(rel_path)
            if video_path is None:
                self._send_json({"error": "video not found"}, status=404)
                return
            mask_path = _mask_path_for(video_path)
            if not mask_path.exists():
                self._send_json({"error": "no mask saved yet"}, status=404)
                return
            self._send_bytes(mask_path.read_bytes(), "image/png")
            return

        self._send_json({"error": "not found"}, status=404)

    # -- POST -------------------------------------------------------------
    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/mask":
            self._send_json({"error": "not found"}, status=404)
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON body"}, status=400)
            return

        rel_path = payload.get("path", "")
        data_url = payload.get("image_data", "")
        video_path = self._resolve_video(rel_path)
        if video_path is None:
            self._send_json({"error": "video not found"}, status=404)
            return
        if not data_url.startswith("data:image/png;base64,"):
            self._send_json({"error": "image_data must be a PNG data URL"}, status=400)
            return

        png_bytes = base64.b64decode(data_url.split(",", 1)[1])
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        mask_img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            self._send_json({"error": "could not decode mask image"}, status=400)
            return

        # Debug: log mask stats
        white_px = int((mask_img > 127).sum())
        total_px = mask_img.size
        print(f"[DEBUG POST /api/mask] {video_path.name}: {white_px}/{total_px} white pixels ({100*white_px/max(total_px,1):.1f}%)")

        _, _, native_w = 0, 0, 0
        _, native_w, native_h = _read_first_frame(video_path)
        if native_w and native_h and mask_img.shape[:2] != (native_h, native_w):
            mask_img = cv2.resize(mask_img, (native_w, native_h), interpolation=cv2.INTER_NEAREST)

        # binarize: anything painted (>127) is kept
        _, mask_bin = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

        mask_path = _mask_path_for(video_path)
        cv2.imwrite(str(mask_path), mask_bin)
        self._send_json({"ok": True, "saved_to": str(mask_path)})

    def log_message(self, fmt: str, *args) -> None:  # quieter default logging
        print(f"[mask_editor] {self.address_string()} - {fmt % args}")


def make_handler(root_dir: Path):
    class _Handler(MaskEditorHandler):
        pass
    _Handler.root_dir = root_dir
    return _Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data", help="root folder to scan for videos (recursively)")
    ap.add_argument("--host", default="0.0.0.0", help="bind address (0.0.0.0 to expose on the server's network)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    root_dir = Path(args.dir).resolve()
    if not root_dir.exists():
        raise SystemExit(f"Directory not found: {root_dir}")

    handler = make_handler(root_dir)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Mask editor serving videos under {root_dir}")
    print(f"Open http://<server-ip>:{args.port}/ in your browser")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
