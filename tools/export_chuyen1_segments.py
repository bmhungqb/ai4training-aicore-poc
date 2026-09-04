#!/usr/bin/env python3
"""Export Chuyền 1 ground-truth timestamp segments from Google Sheet into data/{id}/chuyen1_segment.json."""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def parse_sheet_xlsx(xlsx_path: Path) -> dict[str, dict]:
    with zipfile.ZipFile(xlsx_path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            tree = ET.fromstring(z.read('xl/sharedStrings.xml'))
            strings = [''.join(node.itertext()) for node in tree.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si')]

        rels_tree = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        rel_map = {r.get('Id'): r.get('Target') for r in rels_tree.findall('{http://schemas.openxmlformats.org/package/2006/relationships}Relationship')}

        wb_tree = ET.fromstring(z.read('xl/workbook.xml'))
        sheets = wb_tree.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheet')

        results = {}
        for s in sheets:
            if s.get('state') == 'hidden':
                continue
            name = s.get('name')
            rid = s.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
            sheet_tree = ET.fromstring(z.read('xl/' + rel_map[rid]))
            rows = sheet_tree.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row')

            cd_title = ''
            current_section = None
            segments = []

            for r in rows:
                vals = {}
                for c in r.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c'):
                    coord = c.get('r')
                    col = ''.join([ch for ch in coord if ch.isalpha()])
                    t = c.get('t')
                    v = c.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
                    val = v.text if v is not None else ''
                    if t == 's' and val.isdigit():
                        val = strings[int(val)]
                    vals[col] = val
                    if coord == 'A2':
                        cd_title = val.strip()

                stt_val = vals.get('A', '').strip()
                time_str = vals.get('B', '').strip()
                dur_val = vals.get('C', '').strip()
                name_val = vals.get('D', '').strip()
                note_val = vals.get('E', '').strip()

                # Section headers (e.g. "Tay Phải", "Tay Trái", "Câu dây")
                if not stt_val.isdigit() and name_val and not any(w in name_val.lower() for w in ['thao tác', 'thời gian', 'cđ']):
                    current_section = name_val
                    continue

                if not stt_val.isdigit() and not (time_str and '-' in time_str):
                    continue

                parts = re.split(r'[-–—]', time_str)
                if len(parts) == 2:
                    try:
                        t_start = float(parts[0].strip().replace(',', '.'))
                        t_end = float(parts[1].strip().replace(',', '.'))
                        dur = float(dur_val.replace(',', '.')) if dur_val else round(t_end - t_start, 2)
                        
                        # Fix known typo in sheet Tra tay chính: 312 - 214 -> 312 - 314
                        if t_start > t_end and dur > 0:
                            t_end = round(t_start + dur, 2)

                        segments.append({
                            'stt': int(stt_val) if stt_val.isdigit() else len(segments) + 1,
                            'name': name_val,
                            'section': current_section,
                            'timestamp_start': t_start,
                            'timestamp_end': t_end,
                            'duration': dur,
                            'note': note_val if note_val else None
                        })
                    except Exception as err:
                        print(f"Error parsing row in {name}: {vals} -> {err}")

            results[name] = {
                'sheet_name': name,
                'cd_title': cd_title,
                'segments': segments
            }
        return results


def main() -> None:
    xlsx_path = Path('downloaded_sheet.xlsx')
    if not xlsx_path.exists():
        raise SystemExit(f"Missing {xlsx_path}")

    # Sheet name to metadata mapping
    sheet_mapping = {
        'Diễu TP 4 cạnh túi lai x2': {'id': 1, 'video': 'cam-03_20260805_073527_cut_0_0-0_57.mp4'},
        'Khóa lưỡi gà lai': {'id': 2, 'video': 'cam-03_20260805_074035_cut_1_15-3_16.mp4'},
        'Ráp chèn tay lót': {'id': 3, 'video': 'cam-03_20260805_074444_cut_0_7-0_44.mp4'},
        'May đáp túi lai': {'id': 4, 'video': 'cam-03_20260805_074840_cut_0_2-0_31.mp4'},
        'Rập lược TP cầu dk': {'id': 5, 'video': 'cam-03_20260807_023331_cut_0_33-2_47.mp4'},
        'Ráp ngang đô sau': {'id': 6, 'video': 'cam-03_20260807_024253_cut_1_24-2_55.mp4'},
        'Tra tay lót': {'id': 8, 'video': 'cam-03_20260805_081345_cut_0_0-5_24.mp4'},
        'Tra cổ chính': {'id': 9, 'video': 'cam-03_20260805_082208_cut_3_30-5_29.mp4'},
        'Tra tay chính': {'id': 11, 'alt_id': 10, 'video': 'cam-03_20260807_030000_cut_4_54-12_29.mp4'},
    }

    parsed = parse_sheet_xlsx(xlsx_path)
    print(f"Parsed {len(parsed)} visible sheets.")

    for sheet_name, info in parsed.items():
        mapping = sheet_mapping.get(sheet_name)
        if not mapping:
            print(f"Warning: No mapping for sheet {sheet_name}")
            continue

        target_id = mapping['id']
        alt_id = mapping.get('alt_id')
        video_name = mapping['video']
        segments = info['segments']

        total_dur = round(segments[-1]['timestamp_end'] - segments[0]['timestamp_start'], 2) if segments else 0

        # Build expert scenes list compatible with expert.json
        expert_scenes = [
            {
                'scene_index': i,
                'timestamp_start': seg['timestamp_start'],
                'timestamp_end': seg['timestamp_end'],
                'operations': [
                    {
                        'operation_id': seg['stt'],
                        'name': seg['name']
                    }
                ]
            }
            for i, seg in enumerate(segments)
        ]

        out_data = {
            'cong_doan_id': target_id,
            'sheet_title': info['cd_title'],
            'sheet_name': sheet_name,
            'chuyen': 'Chuyền 1',
            'video_file': video_name,
            'total_segments': len(segments),
            'timestamp_start': segments[0]['timestamp_start'] if segments else 0,
            'timestamp_end': segments[-1]['timestamp_end'] if segments else 0,
            'total_duration_s': total_dur,
            'segments': segments,
            'expert_scenes': expert_scenes
        }

        # Write to primary target_id
        target_dir = Path(f"data/{target_id}")
        target_dir.mkdir(parents=True, exist_ok=True)
        out_file = target_dir / "chuyen1_segment.json"
        out_file.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"Saved: {out_file} ({len(segments)} segments, duration {total_dur}s)")

        # If alt_id exists (e.g. id=10 for Tra tay chính which is folder data/10), also save there
        if alt_id is not None:
            alt_dir = Path(f"data/{alt_id}")
            alt_dir.mkdir(parents=True, exist_ok=True)
            alt_file = alt_dir / "chuyen1_segment.json"
            alt_data = dict(out_data)
            alt_data['cong_doan_id'] = alt_id
            alt_data['note'] = f"Sheet header is CĐ 10, but video corresponds to CĐ 11 ({video_name}) in data/11"
            alt_file.write_text(json.dumps(alt_data, indent=2, ensure_ascii=False), encoding='utf-8')
            print(f"Saved (copy for CĐ {alt_id}): {alt_file}")


if __name__ == '__main__':
    main()
