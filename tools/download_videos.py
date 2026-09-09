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
    # Download Chuyền 2 only (gid=1369687799):
    python -m tools.download_videos --chuyen 2

    # Download Chuyền 1 and 2:
    python -m tools.download_videos --chuyen 1 2

    # Download all chuyền (default):
    python -m tools.download_videos

    # Dry-run inspection without downloading:
    python -m tools.download_videos --chuyen 2 --dry-run

    # Force re-scan Google Drive folder index:
    python -m tools.download_videos --refresh-drive-cache

Requires: gdown (public Drive folder listing/download, no OAuth/service
account needed since the folder is shared publicly).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
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

DEFAULT_TABS: dict[str, str] = {
    "Chuyền 1": "1709514550",
    "Chuyền 2": "1369687799",
    "Chuyền 3": "1306500353",
}

TAB_ALIASES: dict[str, str] = {
    "1": "Chuyền 1",
    "chuyen1": "Chuyền 1",
    "chuyen 1": "Chuyền 1",
    "chuyền 1": "Chuyền 1",
    "2": "Chuyền 2",
    "chuyen2": "Chuyền 2",
    "chuyen 2": "Chuyền 2",
    "chuyền 2": "Chuyền 2",
    "3": "Chuyền 3",
    "chuyen3": "Chuyền 3",
    "chuyen 3": "Chuyền 3",
    "chuyền 3": "Chuyền 3",
}


def normalize_filename(raw: str) -> str:
    """The sheet's Link column sometimes has leading/trailing whitespace or quotes."""
    return raw.strip().strip("\"'")


def extract_drive_id(link: str) -> str | None:
    """Extract Google Drive file ID if the link is a full URL."""
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", link)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", link)
    if m:
        return m.group(1)
    return None


def fetch_sheet_raw(sheet_id: str, gid: str) -> str:
    """Download the sheet tab as raw CSV text."""
    url = SHEET_EXPORT_URL.format(sheet_id=sheet_id, gid=gid)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8")


def parse_sheet_rows_from_text(text: str) -> list[dict]:
    """Parse rows from CSV text, handling leading blank lines and skipping empty seq rows."""
    lines = [ln for ln in text.splitlines()]
    reader = csv.DictReader(lines[1:] if lines and lines[0].strip(",") == "" else lines)
    rows = []
    for row in reader:
        if not row.get(COL_SEQ, "").strip():
            continue
        rows.append(row)
    return rows


def fetch_sheet_rows(sheet_id: str, gid: str) -> list[dict]:
    """Download the sheet as CSV (public export, no auth) and parse rows."""
    text = fetch_sheet_raw(sheet_id, gid)
    return parse_sheet_rows_from_text(text)


def parse_cell_value(val: str | None):
    """Convert raw string cell values to typed representation matching cong_doan_all.json."""
    if val is None:
        return None
    s = val.strip()
    if not s:
        return None
    if s.upper() == "TRUE":
        return True
    if s.upper() == "FALSE":
        return False
    try:
        return float(s)
    except ValueError:
        return s


def update_sheet_cache(out_dir: Path, tab_name: str, raw_csv: str, rows: list[dict]) -> None:
    """Save raw CSV to data/sheets/{Tab}.csv and update data/sheets/cong_doan_all.json."""
    sheets_dir = out_dir / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_slug = tab_name.replace(" ", "_")
    csv_file = sheets_dir / f"{csv_slug}.csv"
    csv_file.write_text(raw_csv, encoding="utf-8")

    # Update cong_doan_all.json
    all_json_file = sheets_dir / "cong_doan_all.json"
    all_data: dict[str, list] = {}
    if all_json_file.exists():
        try:
            all_data = json.loads(all_json_file.read_text(encoding="utf-8"))
        except Exception:
            all_data = {}

    typed_rows = []
    for r in rows:
        row_dict = {}
        for k, v in r.items():
            row_dict[k] = parse_cell_value(v)
        typed_rows.append(row_dict)

    all_data[tab_name] = typed_rows
    all_json_file.write_text(json.dumps(all_data, indent=2, ensure_ascii=False), encoding="utf-8")


def list_drive_videos(
    folder_id: str,
    cache_path: Path | None = None,
    refresh_cache: bool = False,
    video_exts: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
) -> dict[str, str]:
    """Recursively list every video file under the public Drive folder.
    Returns {filename: file_id}. If cache_path is provided and valid, loads from
    cache unless refresh_cache is True."""
    if cache_path and cache_path.exists() and not refresh_cache:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached:
                print(f"Loaded {len(cached)} video(s) from Drive cache ({cache_path}).")
                return cached
        except Exception as err:
            print(f"Warning: Failed to load drive cache from {cache_path}: {err}")

    print(f"Walking Drive folder tree {folder_id} (this may take 30-45 seconds)...")
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

    if cache_path:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(by_name, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Saved Drive video index cache ({len(by_name)} files) to {cache_path}.")
        except Exception as err:
            print(f"Warning: Failed to write drive cache to {cache_path}: {err}")

    return by_name


def resolve_tabs_to_process(
    chuyen_args: list[str] | None = None,
    gid_args: list[str] | None = None,
    all_tabs: bool = False,
) -> list[tuple[str, str]]:
    """Determine which sheet tabs to process based on CLI args."""
    if chuyen_args:
        resolved = []
        for c in chuyen_args:
            norm = c.strip().lower()
            if norm == "all":
                return list(DEFAULT_TABS.items())
            tab_name = TAB_ALIASES.get(norm)
            if not tab_name:
                # Direct check against DEFAULT_TABS keys
                tab_name = next((k for k in DEFAULT_TABS if k.lower() == norm), None)
            if tab_name and tab_name in DEFAULT_TABS:
                resolved.append((tab_name, DEFAULT_TABS[tab_name]))
            else:
                raise SystemExit(
                    f"Invalid chuyền: '{c}'. Valid options: 1, 2, 3, 'Chuyền 1', 'Chuyền 2', 'Chuyền 3', or 'all'."
                )
        return resolved

    if gid_args:
        resolved = []
        for g in gid_args:
            tab_name = next((name for name, tab_gid in DEFAULT_TABS.items() if tab_gid == g), f"Tab {g}")
            resolved.append((tab_name, g))
        return resolved

    # Default to all configured tabs
    return list(DEFAULT_TABS.items())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--sheet-id",
        default="1BDx6UMDxh9Z0mT-r3l2go01t3w5CLoTg",
        help="Google Sheets file ID (cong_doan_sp1.xlsx)",
    )
    ap.add_argument(
        "-c", "--chuyen", "--tab",
        dest="chuyen",
        nargs="*",
        default=None,
        help="chuyền to process (e.g. 2, 'chuyen 2', 1 2, or all). Overrides --gid.",
    )
    ap.add_argument(
        "--gid",
        nargs="*",
        default=None,
        help="sheet tab gid(s). Defaults to all 3 tabs if neither --chuyen nor --gid is given.",
    )
    ap.add_argument(
        "--all-tabs",
        action="store_true",
        help="download from all tabs (Chuyền 1, Chuyền 2, Chuyền 3)",
    )
    ap.add_argument(
        "--drive-folder-id",
        default="157NMvEcMDU5bHiPGAPuvR6lQfzN5k_Mq",
        help="public Google Drive folder ID containing the video subfolders",
    )
    ap.add_argument(
        "--out-dir",
        default="data",
        help="root folder to download into (default: data)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the target file already exists",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="list files that would be downloaded without actually downloading",
    )
    ap.add_argument(
        "--refresh-drive-cache",
        action="store_true",
        help="force re-scanning public Drive folder index instead of using local cache",
    )
    ap.add_argument(
        "--no-sheet-cache",
        action="store_true",
        help="skip updating local CSV and cong_doan_all.json in data/sheets",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    drive_cache_file = out_dir / ".drive_index_cache.json"

    gids_to_process = resolve_tabs_to_process(
        chuyen_args=args.chuyen,
        gid_args=args.gid,
        all_tabs=args.all_tabs,
    )

    print(f"\nTabs to process: {', '.join(t[0] for t in gids_to_process)}")

    videos_by_name = list_drive_videos(
        folder_id=args.drive_folder_id,
        cache_path=drive_cache_file,
        refresh_cache=args.refresh_drive_cache,
    )
    print(f"Indexed {len(videos_by_name)} unique video file(s) in Drive.")

    total_ok, total_skipped, total_not_found = 0, 0, []

    for tab_name, gid in gids_to_process:
        print(f"\n" + "=" * 60)
        print(f"Processing {tab_name} (gid={gid})...")
        raw_csv = fetch_sheet_raw(args.sheet_id, gid)
        rows = parse_sheet_rows_from_text(raw_csv)
        print(f"Found {len(rows)} công đoạn row(s).")

        if not args.no_sheet_cache:
            update_sheet_cache(out_dir, tab_name, raw_csv, rows)
            print(f"Updated local sheet cache for {tab_name} in {out_dir / 'sheets'}")

        ok, skipped, not_found = 0, 0, []
        refreshed_during_run = False

        for row in rows:
            seq = row[COL_SEQ].strip()
            # remove float formatting if any (e.g. '1.0' -> '1')
            if seq.endswith(".0"):
                seq = seq[:-2]
            link_val = row.get(COL_LINK, "")
            name = normalize_filename(link_val)
            op_name = row.get(COL_NAME, "").strip()

            if not name:
                print(f"[{tab_name} | CĐ {seq}] SKIP: empty Link column ({op_name})")
                not_found.append((tab_name, seq, op_name, "(empty Link)"))
                continue

            file_id = extract_drive_id(name) or videos_by_name.get(name)

            # If not in cache, refresh Drive index once in case newly uploaded
            if file_id is None and not refreshed_during_run and not args.refresh_drive_cache:
                print(f"\n[{tab_name} | CĐ {seq}] '{name}' not found in local Drive cache. Refreshing Drive index...")
                videos_by_name = list_drive_videos(
                    folder_id=args.drive_folder_id,
                    cache_path=drive_cache_file,
                    refresh_cache=True,
                )
                refreshed_during_run = True
                file_id = videos_by_name.get(name)

            if file_id is None:
                print(f"[{tab_name} | CĐ {seq}] NOT FOUND in Drive: {name!r} ({op_name})")
                not_found.append((tab_name, seq, op_name, name))
                continue

            dest_dir = out_dir / seq
            dest_dir.mkdir(parents=True, exist_ok=True)
            # If name is a URL, sanitize or infer filename
            clean_filename = Path(name).name if not name.startswith("http") else f"{seq}.mp4"
            dest_path = dest_dir / clean_filename

            if dest_path.exists() and not args.force:
                print(f"[{tab_name} | CĐ {seq}] already downloaded -> {dest_path}")
                skipped += 1
                continue

            if args.dry_run:
                print(f"[{tab_name} | CĐ {seq}] [DRY-RUN] would download {clean_filename} (id={file_id}) -> {dest_path}")
                ok += 1
                continue

            print(f"[{tab_name} | CĐ {seq}] downloading {clean_filename} -> {dest_path}")
            gdown.download(id=file_id, output=str(dest_path), quiet=False)
            ok += 1

        total_ok += ok
        total_skipped += skipped
        total_not_found.extend(not_found)
        action_verb = "Would download" if args.dry_run else "Downloaded"
        print(f"{tab_name} summary: {action_verb}: {ok} | already present: {skipped} | not found: {len(not_found)}")

    print("\n" + "=" * 70)
    summary_verb = "WOULD DOWNLOAD" if args.dry_run else "DOWNLOADED"
    print(f"ALL SELECTED TABS: {summary_verb}: {total_ok} | already present: {total_skipped} | not found / empty: {len(total_not_found)}")
    if total_not_found:
        print("\nNot found or empty link rows:")
        for tab_name, seq, op_name, name in total_not_found:
            print(f"  [{tab_name} | CĐ {seq}] {op_name} -> {name!r}")


if __name__ == "__main__":
    main()
