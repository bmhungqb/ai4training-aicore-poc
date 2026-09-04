#!/usr/bin/env python3
"""Download every công đoạn's video from the shared (public, no-credential)
Google Drive folder into its own folder under data/, named by the sheet's
"Số thứ tự công đoạn" (operation sequence number).

Source of truth for which file belongs to which operation: the "Link" column
in the public Google Sheet (cong_doan_sp1.xlsx), matched by exact filename
against every video found anywhere in the Drive folder tree (recursively —
videos are nested under per-source subfolders like video-sp1-chuyen1/cd4/...,
and the same operation number can map to differently-named subfolders across
sources, so matching by filename is the only reliable key).

Usage:
    python -m tools.download_videos \
        --sheet-id 1BDx6UMDxh9Z0mT-r3l2go01t3w5CLoTg --gid 1709514550 \
        --drive-folder-id 157NMvEcMDU5bHiPGAPuvR6lQfzN5k_Mq \
        --out-dir data

Requires: gdown (public Drive folder listing/download, no OAuth/service
account needed since the folder is shared publicly).
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import urllib.request
from pathlib import Path

try:
    import gdown
except ImportError:
    raise SystemExit("Missing dependency: pip install gdown")

SHEET_EXPORT_URL = "https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/{folder_id}"

COL_SEQ = "Số thứ tự công đoạn"
COL_NAME = "Tên công đoạn"
COL_LINK = "Link"


def fetch_sheet_rows(sheet_id: str, gid: str) -> list[dict]:
    """Download the sheet as CSV (public export, no auth) and parse rows."""
    url = SHEET_EXPORT_URL.format(sheet_id=sheet_id, gid=gid)
    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode("utf-8")

    # the sheet has a leading blank row before the real header
    lines = [ln for ln in text.splitlines()]
    reader = csv.DictReader(lines[1:] if lines and lines[0].strip(",") == "" else lines)
    rows = []
    for row in reader:
        if not row.get(COL_SEQ, "").strip():
            continue
        rows.append(row)
    return rows


def list_drive_videos(folder_id: str, video_exts: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")) -> dict[str, str]:
    """Recursively list every video file under the public Drive folder.
    Returns {filename: file_id}. If the same filename appears more than once
    across subfolders, the first one found wins and a warning is printed —
    check --report for duplicates if that matters for your dataset."""
    url = DRIVE_FOLDER_URL.format(folder_id=folder_id)
    items = gdown.download_folder(url=url, skip_download=True, quiet=True)

    by_name: dict[str, str] = {}
    dupes: dict[str, list[str]] = {}
    for item in items:
        name = Path(item.path).name
        if not name.lower().endswith(video_exts):
            continue
        if name in by_name and by_name[name] != item.id:
            dupes.setdefault(name, [by_name[name]]).append(item.id)
            continue
        by_name[name] = item.id

    if dupes:
        print(f"\nWARNING: {len(dupes)} filename(s) appear more than once in Drive "
              f"(kept the first match found):")
        for name, ids in dupes.items():
            print(f"  {name}: {ids}")

    return by_name


def normalize_filename(raw: str) -> str:
    """The sheet's Link column sometimes has leading/trailing whitespace."""
    return raw.strip()


DEFAULT_TABS: dict[str, str] = {
    "Chuyền 1": "1709514550",
    "Chuyền 2": "1369687799",
    "Chuyền 3": "1306500353",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet-id", default="1BDx6UMDxh9Z0mT-r3l2go01t3w5CLoTg",
                    help="Google Sheets file ID (cong_doan_sp1.xlsx)")
    ap.add_argument("--gid", nargs="*", default=None,
                    help="sheet tab gid(s). Defaults to all 3 tabs if --all-tabs is used or if no gid is given.")
    ap.add_argument("--all-tabs", action="store_true", default=True,
                    help="download from all tabs (Chuyền 1, Chuyền 2, Chuyền 3)")
    ap.add_argument("--drive-folder-id", default="157NMvEcMDU5bHiPGAPuvR6lQfzN5k_Mq",
                    help="public Google Drive folder ID containing the video subfolders")
    ap.add_argument("--out-dir", default="data", help="root folder to download into")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the target file already exists")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gids_to_process: list[tuple[str, str]] = []
    if args.gid:
        for g in args.gid:
            # find tab name if known
            tab_name = next((name for name, tab_gid in DEFAULT_TABS.items() if tab_gid == g), f"Tab {g}")
            gids_to_process.append((tab_name, g))
    else:
        gids_to_process = list(DEFAULT_TABS.items())

    print(f"\nListing videos under Drive folder {args.drive_folder_id} (this walks the whole "
          f"folder tree, may take a minute)...")
    videos_by_name = list_drive_videos(args.drive_folder_id)
    print(f"Found {len(videos_by_name)} unique video file(s) in Drive.")

    total_ok, total_skipped, total_not_found = 0, 0, []

    for tab_name, gid in gids_to_process:
        print(f"\n" + "=" * 50)
        print(f"Processing {tab_name} (gid={gid})...")
        rows = fetch_sheet_rows(args.sheet_id, gid)
        print(f"Found {len(rows)} công đoạn row(s).")

        ok, skipped, not_found = 0, 0, []
        for row in rows:
            seq = row[COL_SEQ].strip()
            # remove float formatting if any (e.g. '1.0' -> '1')
            if seq.endswith(".0"):
                seq = seq[:-2]
            name = normalize_filename(row.get(COL_LINK, ""))
            op_name = row.get(COL_NAME, "").strip()
            if not name:
                print(f"[{tab_name} | CĐ {seq}] SKIP: empty Link column ({op_name})")
                not_found.append((tab_name, seq, op_name, "(empty Link)"))
                continue

            file_id = videos_by_name.get(name)
            if file_id is None:
                print(f"[{tab_name} | CĐ {seq}] NOT FOUND in Drive: {name!r} ({op_name})")
                not_found.append((tab_name, seq, op_name, name))
                continue

            dest_dir = out_dir / seq
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / name

            if dest_path.exists() and not args.force:
                print(f"[{tab_name} | CĐ {seq}] already downloaded -> {dest_path}")
                skipped += 1
                continue

            print(f"[{tab_name} | CĐ {seq}] downloading {name} -> {dest_path}")
            gdown.download(id=file_id, output=str(dest_path), quiet=False)
            ok += 1

        total_ok += ok
        total_skipped += skipped
        total_not_found.extend(not_found)
        print(f"{tab_name} summary: Downloaded: {ok} | already present: {skipped} | not found: {len(not_found)}")

    print("\n" + "=" * 70)
    print(f"ALL TABS TOTAL: Downloaded: {total_ok} | already present: {total_skipped} | not found / empty: {len(total_not_found)}")
    if total_not_found:
        print("\nNot found or empty link rows:")
        for tab_name, seq, op_name, name in total_not_found:
            print(f"  [{tab_name} | CĐ {seq}] {op_name} -> {name!r}")


if __name__ == "__main__":
    main()
