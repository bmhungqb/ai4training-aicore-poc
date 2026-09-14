#!/usr/bin/env python3
"""Local server for the Expert Timestamp Adjustment Web Tool.

Allows annotators to watch the expert video frame-by-frame, inspect physical kinematic
action segments (SEA-RAFT optical flow), compare with manual ground truth, and adjust
scene start/end timestamps with instant preview and 1-click saving back to expert.json.

Usage:
    python tools/expert_timestamp_editor/server.py [cong_doan] [--port 8766]
"""
from __future__ import annotations

import argparse
import datetime
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_HTML = Path(__file__).resolve().parent / "index.html"


def find_expert_video(cd_dir: Path) -> Path | None:
    cand_symlink = cd_dir / "expert.mp4"
    if cand_symlink.exists():
        return cand_symlink
    # Look for .mp4 that is referenced in expert.json
    exp_json = cd_dir / "expert.json"
    if exp_json.is_file():
        try:
            data = json.loads(exp_json.read_text(encoding="utf-8"))
            vp = Path(data["expert_video"][0]["file_path"])
            if vp.is_file():
                return vp
            if (REPO_ROOT / vp).is_file():
                return REPO_ROOT / vp
        except Exception:
            pass
    mp4s = sorted(cd_dir.glob("*.mp4"))
    return mp4s[0] if mp4s else None


def make_handler(cd: str, port: int):
    cd_dir = REPO_ROOT / "data" / str(cd)
    expert_json_path = cd_dir / "expert.json"
    chuyen1_path = cd_dir / "chuyen1_segment.json"
    kinematic_path = cd_dir / "kinematic_expert" / "action_segments.json"
    if not kinematic_path.is_file():
        # Fallback to kinematic folder with video stem
        cand_kin = list((cd_dir / "kinematic").glob("*/action_segments.json"))
        if cand_kin:
            kinematic_path = cand_kin[0]

    video_file = find_expert_video(cd_dir)

    class ExpertTimestampHandler(BaseHTTPRequestHandler):
        def _send_json(self, data: dict, status: int = 200) -> None:
            buf = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(buf)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()
            self.wfile.write(buf)

        def do_OPTIONS(self) -> None:
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _serve_file_with_range(self, file_path: Path, content_type: str) -> None:
            """Serve a file supporting HTTP 206 Partial Content (Range requests) for video seeking."""
            if not file_path.is_file():
                self.send_error(404, f"File not found: {file_path.name}")
                return

            file_size = file_path.stat().st_size
            range_header = self.headers.get("Range")

            if not range_header:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(file_size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with open(file_path, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
                return

            # Parse Range header: bytes=start-end
            match = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if not match:
                self.send_error(416, "Requested Range Not Satisfiable")
                return

            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else file_size - 1
            if start >= file_size or end >= file_size or start > end:
                self.send_error(416, "Requested Range Not Satisfiable")
                return

            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(file_path, "rb") as f:
                f.seek(start)
                bytes_to_send = length
                buf_size = 64 * 1024
                while bytes_to_send > 0:
                    read_len = min(buf_size, bytes_to_send)
                    data = f.read(read_len)
                    if not data:
                        break
                    self.wfile.write(data)
                    bytes_to_send -= len(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)

            if parsed.path in ("/", "/index.html"):
                if not INDEX_HTML.is_file():
                    self.send_error(404, "index.html not found")
                    return
                html_bytes = INDEX_HTML.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.end_headers()
                self.wfile.write(html_bytes)
                return

            if parsed.path == "/video":
                if video_file and video_file.is_file():
                    self._serve_file_with_range(video_file, "video/mp4")
                else:
                    self.send_error(404, "Video file not found")
                return

            if parsed.path == "/api/data":
                expert_data = {}
                if expert_json_path.is_file():
                    try:
                        expert_data = json.loads(expert_json_path.read_text(encoding="utf-8"))
                    except Exception as e:
                        expert_data = {"error": str(e)}

                chuyen1_data = {}
                if chuyen1_path.is_file():
                    try:
                        chuyen1_data = json.loads(chuyen1_path.read_text(encoding="utf-8"))
                    except Exception:
                        pass

                kinematic_data = {}
                if kinematic_path.is_file():
                    try:
                        kinematic_data = json.loads(kinematic_path.read_text(encoding="utf-8"))
                    except Exception:
                        pass

                payload = {
                    "cong_doan": cd,
                    "video_name": video_file.name if video_file else "",
                    "video_url": "/video",
                    "expert_data": expert_data,
                    "chuyen1_data": chuyen1_data,
                    "kinematic_data": kinematic_data,
                }
                self._send_json(payload)
                return

            # Static files fallback
            requested = (REPO_ROOT / parsed.path.lstrip("/")).resolve()
            if requested.is_file() and str(requested).startswith(str(REPO_ROOT)):
                mime, _ = mimetypes.guess_type(str(requested))
                self._serve_file_with_range(requested, mime or "application/octet-stream")
                return

            self.send_error(404, f"Not found: {parsed.path}")

        def do_HEAD(self) -> None:
            self.do_GET()

        def do_POST(self) -> None:
            parsed = urlparse(self.path)

            if parsed.path == "/api/save":
                length = int(self.headers.get("Content-Length", 0))
                try:
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                except Exception as e:
                    self._send_json({"ok": False, "error": f"Invalid JSON: {e}"}, status=400)
                    return

                new_scenes = payload.get("scenes")
                if not isinstance(new_scenes, list):
                    self._send_json({"ok": False, "error": "Missing or invalid 'scenes' array"}, status=400)
                    return

                if not expert_json_path.is_file():
                    self._send_json({"ok": False, "error": f"File not found: {expert_json_path}"}, status=404)
                    return

                try:
                    expert_data = json.loads(expert_json_path.read_text(encoding="utf-8"))
                    # Create timestamped backup
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = expert_json_path.with_suffix(f".json.bak_{ts}")
                    shutil.copy2(expert_json_path, backup_path)

                    # Update scenes in expert_video[0]
                    expert_data["expert_video"][0]["scenes"] = new_scenes
                    expert_json_path.write_text(json.dumps(expert_data, indent=2, ensure_ascii=False), encoding="utf-8")

                    print(f"[SAVE] Successfully saved {len(new_scenes)} scenes to {expert_json_path}")
                    print(f"[BACKUP] Created backup at {backup_path}")

                    self._send_json({
                        "ok": True,
                        "message": f"Saved {len(new_scenes)} scenes successfully!",
                        "backup_file": backup_path.name,
                    })
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, status=500)
                return

            if parsed.path == "/api/rerun_step1":
                try:
                    print("[RERUN] Running tools/run_expert_selection.py...")
                    p1 = subprocess.run([sys.executable, str(REPO_ROOT / "tools/run_expert_selection.py"), str(cd)],
                                        capture_output=True, text=True, cwd=str(REPO_ROOT))
                    print(p1.stdout)
                    if p1.returncode != 0:
                        raise RuntimeError(f"run_expert_selection error: {p1.stderr}")

                    print("[RERUN] Running tools/visualize_expert_grid.py...")
                    p2 = subprocess.run([sys.executable, str(REPO_ROOT / "tools/visualize_expert_grid.py"), str(cd)],
                                        capture_output=True, text=True, cwd=str(REPO_ROOT))
                    print(p2.stdout)
                    if p2.returncode != 0:
                        raise RuntimeError(f"visualize_expert_grid error: {p2.stderr}")

                    self._send_json({
                        "ok": True,
                        "message": "Re-ran Step 1.1, 1.2 and Grid Visualization successfully!",
                    })
                except Exception as e:
                    self._send_json({"ok": False, "error": str(e)}, status=500)
                return

            self.send_error(404, "Not found")

    return ExpertTimestampHandler


def main():
    parser = argparse.ArgumentParser(description="Serve Expert Timestamp Adjustment Web Tool.")
    parser.add_argument("cong_doan", nargs="?", default="1", help="Cong doan ID (default: 1)")
    parser.add_argument("--port", type=int, default=8766, help="Port to serve on (default: 8766)")
    args = parser.parse_args()

    handler_cls = make_handler(args.cong_doan, args.port)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler_cls)
    url = f"http://localhost:{args.port}"

    print("=" * 65)
    print(" 🎬 EXPERT TIMESTAMP ADJUSTMENT WEB TOOL")
    print("=" * 65)
    print(f" Công đoạn : {args.cong_doan}")
    print(f" URL       : {url}")
    print(" Mở URL trên trong trình duyệt để bắt đầu điều chỉnh timestamp.")
    print(" Nhấn Ctrl+C để dừng server.")
    print("=" * 65)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
