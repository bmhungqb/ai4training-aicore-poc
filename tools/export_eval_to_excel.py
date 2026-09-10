import json
from pathlib import Path
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

def build_excel_report(out_path: str = "evaluation_result_9cd.xlsx"):
    cds = ["1", "2", "3", "4", "5", "6", "8", "9", "10"]
    data_dir = Path("data")
    result_dir = Path("data_result")
    
    cd_results = []
    all_steps_detail = []
    primary_tau = 0.5
    
    for cd in cds:
        gt_f = data_dir / cd / "chuyen1_segment.json"
        gt_data = json.loads(gt_f.read_text(encoding="utf-8"))
        vfile = gt_data.get("video_file", "")
        vstem = Path(vfile).stem
        
        pred_candidates = list(result_dir.glob(f"*/kinematic/{vstem}/action_segments.json"))
        if not pred_candidates:
            continue
        pred_f = pred_candidates[0]
        pred_data = json.loads(pred_f.read_text(encoding="utf-8"))
        
        gt_segs = gt_data.get("segments", [])
        pred_segs = pred_data.get("segments", [])
        
        gt_bounds = sorted(list({round(float(s[k]), 3) for s in gt_segs for k in ["timestamp_start", "timestamp_end"]}))
        
        pred_bounds = []
        if pred_segs:
            pred_bounds.append(round(float(pred_segs[0]["start_time_s"]), 3))
            for i in range(1, len(pred_segs)):
                t_trans = (float(pred_segs[i-1]["end_time_s"]) + float(pred_segs[i]["start_time_s"])) / 2.0
                pred_bounds.append(round(t_trans, 3))
            pred_bounds.append(round(float(pred_segs[-1]["end_time_s"]), 3))
        pred_bounds = sorted(list(set(pred_bounds)))
        
        gt_hits = sum(1 for g in gt_bounds if any(abs(g - p) <= primary_tau for p in pred_bounds))
        rec_05 = (gt_hits / len(gt_bounds) * 100) if gt_bounds else 0
        
        both_count = 0
        either_count = 0
        
        for s in gt_segs:
            t0 = float(s["timestamp_start"])
            t1 = float(s["timestamp_end"])
            
            p0, err0 = min(((p, abs(p - t0)) for p in pred_bounds), key=lambda x: x[1]) if pred_bounds else (None, 999)
            p1, err1 = min(((p, abs(p - t1)) for p in pred_bounds), key=lambda x: x[1]) if pred_bounds else (None, 999)
            
            hit0 = (err0 <= primary_tau)
            hit1 = (err1 <= primary_tau)
            if hit0 and hit1:
                both_count += 1
                status = "Khớp CẢ 2 đầu"
            elif hit0:
                either_count += 1
                status = "Khớp mốc Start"
            elif hit1:
                either_count += 1
                status = "Khớp mốc End"
            else:
                status = "Không khớp"
                
            if hit0 and hit1:
                either_count += 1
                
            all_steps_detail.append({
                "cd": int(cd),
                "cd_title": gt_data.get("sheet_title", f"CĐ {cd}"),
                "stt": s.get("stt", 0),
                "name": s.get("name", ""),
                "section": s.get("section", "") or "",
                "t0": t0,
                "t1": t1,
                "dur": round(t1 - t0, 2),
                "p0": p0,
                "err0": round(err0, 3) if p0 is not None else None,
                "hit0": "ĐẠT" if hit0 else "LỆCH",
                "p1": p1,
                "err1": round(err1, 3) if p1 is not None else None,
                "hit1": "ĐẠT" if hit1 else "LỆCH",
                "status": status
            })
            
        cd_results.append({
            "cd": int(cd),
            "title": gt_data.get("sheet_title", f"CĐ {cd}"),
            "video": vfile,
            "gt_steps": len(gt_segs),
            "gt_bounds": len(gt_bounds),
            "pred_segs": len(pred_segs),
            "pred_bounds": len(pred_bounds),
            "split_ratio": round(len(pred_segs) / max(1, len(gt_segs)), 1),
            "rec_05": rec_05,
            "both_count": both_count,
            "both_pct": both_count / len(gt_segs) * 100 if gt_segs else 0,
            "either_count": either_count,
            "either_pct": either_count / len(gt_segs) * 100 if gt_segs else 0,
            "gt_bounds_raw": gt_bounds,
            "pred_bounds_raw": pred_bounds,
            "gt_segs_raw": gt_segs,
        })

    windows = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    sweep_results = []
    total_gt_b = sum(r["gt_bounds"] for r in cd_results)
    total_steps = sum(r["gt_steps"] for r in cd_results)
    total_pred_segs = sum(r["pred_segs"] for r in cd_results)
    
    for w in windows:
        m_rec = sum(sum(1 for g in r["gt_bounds_raw"] if any(abs(g - p) <= w for p in r["pred_bounds_raw"])) / r["gt_bounds"] for r in cd_results) / len(cd_results) * 100
        u_hits = sum(sum(1 for g in r["gt_bounds_raw"] if any(abs(g - p) <= w for p in r["pred_bounds_raw"])) for r in cd_results)
        u_rec = u_hits / total_gt_b * 100
        
        both = sum(sum(1 for s in r["gt_segs_raw"] if any(abs(s["timestamp_start"] - p) <= w for p in r["pred_bounds_raw"]) and any(abs(s["timestamp_end"] - p) <= w for p in r["pred_bounds_raw"])) for r in cd_results)
        both_pct = both / total_steps * 100
        
        either = sum(sum(1 for s in r["gt_segs_raw"] if any(abs(s["timestamp_start"] - p) <= w for p in r["pred_bounds_raw"]) or any(abs(s["timestamp_end"] - p) <= w for p in r["pred_bounds_raw"])) for r in cd_results)
        either_pct = either / total_steps * 100
        
        sweep_results.append({
            "window": w,
            "macro_rec": m_rec,
            "micro_rec": u_rec,
            "u_hits": u_hits,
            "both_pct": both_pct,
            "either_pct": either_pct
        })

    wb = openpyxl.Workbook()
    
    font_title = Font(name="Calibri", size=15, bold=True, color="1F4E78")
    font_sub = Font(name="Calibri", size=11, italic=True, color="595959")
    font_section = Font(name="Calibri", size=12, bold=True, color="1F4E78")
    font_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    font_bold = Font(name="Calibri", size=11, bold=True)
    font_regular = Font(name="Calibri", size=11)
    
    fill_header = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    fill_header_sub = PatternFill(start_color="2F5597", end_color="2F5597", fill_type="solid")
    fill_summary = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    fill_highlight = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    fill_alt = PatternFill(start_color="F9FBFD", end_color="F9FBFD", fill_type="solid")
    fill_hit = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    fill_miss = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
    
    border_thin = Side(border_style="thin", color="D9D9D9")
    border_box = Border(left=border_thin, right=border_thin, top=border_thin, bottom=border_thin)
    border_top_double = Border(
        left=border_thin, right=border_thin,
        top=Side(border_style="thin", color="1F4E78"),
        bottom=Side(border_style="double", color="1F4E78")
    )
    
    align_center = Alignment(horizontal="center", vertical="center")
    align_left = Alignment(horizontal="left", vertical="center")
    align_right = Alignment(horizontal="right", vertical="center")

    # SHEET 1: Tong_Quan_9CD
    ws1 = wb.active
    ws1.title = "Tong_Quan_9CD"
    ws1.views.sheetView[0].showGridLines = True
    
    ws1.cell(row=1, column=1, value="BÁO CÁO ĐÁNH GIÁ ĐỘ ALIGN CỦA ACTION SEGMENTATION (STAGE 1: SAM + SEA-RAFT)").font = font_title
    ws1.cell(row=2, column=1, value="Dữ liệu: Chuyền 1 (9 công đoạn độc lập: CĐ 1, 2, 3, 4, 5, 6, 8, 9, 10) | Cửa sổ chuẩn: ±0.5s").font = font_sub
    
    ws1.cell(row=4, column=1, value="1. ĐÁNH GIÁ CHI TIẾT TỪNG CÔNG ĐOẠN (WINDOW = ±0.5 GIÂY)").font = font_section
    
    t1_headers = [
        "CĐ", "Tên công đoạn", "Số bước GT", "Số đoạn máy cắt", 
        "Tỷ lệ cắt máy / GT", "Boundary Recall", "Khớp CẢ 2 đầu (%)", "Khớp ÍT NHẤT 1 đầu (%)"
    ]
    for col_idx, h in enumerate(t1_headers, start=1):
        c = ws1.cell(row=5, column=col_idx, value=h)
        c.font = font_header
        c.fill = fill_header
        c.alignment = align_center
        c.border = border_box

    for idx, r in enumerate(cd_results):
        r_idx = 6 + idx
        fill = fill_alt if (idx % 2 == 1) else None
        vals = [
            r["cd"], r["title"], r["gt_steps"], r["pred_segs"],
            f"{r['split_ratio']}x", r["rec_05"] / 100.0, r["both_pct"] / 100.0, r["either_pct"] / 100.0
        ]
        for col_idx, val in enumerate(vals, start=1):
            c = ws1.cell(row=r_idx, column=col_idx, value=val)
            c.font = font_regular
            c.border = border_box
            if fill:
                c.fill = fill
            if col_idx == 1:
                c.alignment = align_center
                c.number_format = "#,##0"
            elif col_idx == 2:
                c.alignment = align_left
            elif col_idx in [3, 4]:
                c.alignment = align_right
                c.number_format = "#,##0"
            elif col_idx == 5:
                c.alignment = align_center
            elif col_idx in [6, 7, 8]:
                c.alignment = align_right
                c.number_format = "0.0%"

    # Summary rows
    ws1.cell(row=15, column=1, value="").border = border_box
    ws1.cell(row=15, column=2, value="TRUNG BÌNH CỘNG (MACRO AVERAGE)").font = font_bold
    ws1.cell(row=15, column=3, value=total_steps).number_format = "#,##0"
    ws1.cell(row=15, column=4, value=total_pred_segs).number_format = "#,##0"
    ws1.cell(row=15, column=5, value=f"{round(total_pred_segs / total_steps, 1)}x").alignment = align_center
    ws1.cell(row=15, column=6, value=sum(r["rec_05"] for r in cd_results) / len(cd_results) / 100.0).number_format = "0.0%"
    ws1.cell(row=15, column=7, value=sum(r["both_count"] for r in cd_results) / total_steps).number_format = "0.0%"
    ws1.cell(row=15, column=8, value=sum(r["either_count"] for r in cd_results) / total_steps).number_format = "0.0%"
    
    for c_idx in range(1, 9):
        c = ws1.cell(row=15, column=c_idx)
        c.font = font_bold
        c.fill = fill_summary
        c.border = border_box
        if c_idx in [3, 4, 6, 7, 8]:
            c.alignment = align_right

    u_hits_05 = sum(sum(1 for g in r["gt_bounds_raw"] if any(abs(g - p) <= primary_tau for p in r["pred_bounds_raw"])) for r in cd_results)
    u_rec_05 = u_hits_05 / total_gt_b
    
    ws1.cell(row=16, column=1, value="").border = border_top_double
    ws1.cell(row=16, column=2, value="TỔNG HỢP TOÀN BỘ (MICRO AGGREGATE)").font = font_bold
    ws1.cell(row=16, column=3, value=total_steps).number_format = "#,##0"
    ws1.cell(row=16, column=4, value=total_pred_segs).number_format = "#,##0"
    ws1.cell(row=16, column=5, value=f"{round(total_pred_segs / total_steps, 1)}x").alignment = align_center
    ws1.cell(row=16, column=6, value=u_rec_05).number_format = "0.0%"
    ws1.cell(row=16, column=7, value=sum(r["both_count"] for r in cd_results) / total_steps).number_format = "0.0%"
    ws1.cell(row=16, column=8, value=sum(r["either_count"] for r in cd_results) / total_steps).number_format = "0.0%"
    
    for c_idx in range(1, 9):
        c = ws1.cell(row=16, column=c_idx)
        c.font = font_bold
        c.fill = fill_summary
        c.border = border_top_double
        if c_idx in [3, 4, 6, 7, 8]:
            c.alignment = align_right

    # Table 2: Sweep
    ws1.cell(row=19, column=1, value="2. TIẾN TRÌNH RECALL & KHỚP THAO TÁC THEO DẢI DUNG SAI THỜI GIAN").font = font_section
    t2_headers = [
        "Cửa sổ dung sai (±s)", "Macro Recall (%)", "Micro Recall (%)", 
        "Số mốc GT trúng", "Khớp CẢ 2 đầu (%)", "Khớp ÍT NHẤT 1 đầu (%)"
    ]
    for col_idx, h in enumerate(t2_headers, start=1):
        c = ws1.cell(row=20, column=col_idx, value=h)
        c.font = font_header
        c.fill = fill_header_sub
        c.alignment = align_center
        c.border = border_box

    for idx, sw in enumerate(sweep_results):
        r_idx = 21 + idx
        is_primary = (sw["window"] == 0.5)
        fill = fill_highlight if is_primary else (fill_alt if idx % 2 == 1 else None)
        font_c = font_bold if is_primary else font_regular
        
        row_vals = [
            f"±{sw['window']:.2f}s",
            sw["macro_rec"] / 100.0,
            sw["micro_rec"] / 100.0,
            f"{sw['u_hits']} / {total_gt_b}",
            sw["both_pct"] / 100.0,
            sw["either_pct"] / 100.0
        ]
        for col_idx, val in enumerate(row_vals, start=1):
            c = ws1.cell(row=r_idx, column=col_idx, value=val)
            c.font = font_c
            c.border = border_box
            if fill:
                c.fill = fill
            if col_idx in [1, 4]:
                c.alignment = align_center
            elif col_idx in [2, 3, 5, 6]:
                c.alignment = align_right
                c.number_format = "0.0%"

    col_widths_s1 = {"A": 8, "B": 48, "C": 16, "D": 18, "E": 20, "F": 18, "G": 20, "H": 24}
    for col_letter, width in col_widths_s1.items():
        ws1.column_dimensions[col_letter].width = width

    # SHEET 2: Chi_Tiet_Tung_Thao_Tac
    ws2 = wb.create_sheet(title="Chi_Tiet_Tung_Thao_Tac")
    ws2.views.sheetView[0].showGridLines = True
    
    ws2.cell(row=1, column=1, value="BẢNG ĐỐI CHIẾU CHI TIẾT TỪNG THAO TÁC (349 THAO TÁC TRÊN 9 CÔNG ĐOẠN)").font = font_title
    ws2.cell(row=2, column=1, value="Đối sánh mốc bắt đầu (Start) và kết thúc (End) của Ground Truth với ranh giới gần nhất của Stage 1 (Dung sai: ±0.5s)").font = font_sub
    
    headers_s2 = [
        "CĐ", "Tên công đoạn", "STT", "Phần thân / Vị trí", "Tên thao tác",
        "GT Start (s)", "Vết cắt gần nhất (s)", "Sai lệch Start (s)", "Trạng thái Start",
        "GT End (s)", "Vết cắt gần nhất (s)", "Sai lệch End (s)", "Trạng thái End",
        "Thời lượng GT (s)", "Kết luận khớp thao tác"
    ]
    for col_idx, h in enumerate(headers_s2, start=1):
        c = ws2.cell(row=4, column=col_idx, value=h)
        c.font = font_header
        c.fill = fill_header
        c.alignment = align_center
        c.border = border_box

    for idx, st in enumerate(all_steps_detail):
        r_idx = 5 + idx
        fill_status = fill_hit if st["status"] == "Khớp CẢ 2 đầu" else (fill_highlight if "Khớp" in st["status"] else fill_miss)
        alt_fill = fill_alt if idx % 2 == 1 else None
        
        row_vals = [
            st["cd"], st["cd_title"], st["stt"], st["section"], st["name"],
            st["t0"], st["p0"], st["err0"], st["hit0"],
            st["t1"], st["p1"], st["err1"], st["hit1"],
            st["dur"], st["status"]
        ]
        for col_idx, val in enumerate(row_vals, start=1):
            c = ws2.cell(row=r_idx, column=col_idx, value=val)
            c.border = border_box
            c.font = font_regular
            if alt_fill and col_idx != 15:
                c.fill = alt_fill
            if col_idx in [1, 3, 9, 13]:
                c.alignment = align_center
            elif col_idx in [2, 4, 5]:
                c.alignment = align_left
            elif col_idx in [6, 7, 8, 10, 11, 12, 14]:
                c.alignment = align_right
                c.number_format = "0.00"
            elif col_idx == 15:
                c.alignment = align_center
                c.fill = fill_status
                if st["status"] == "Khớp CẢ 2 đầu":
                    c.font = font_bold

    col_widths_s2 = {
        "A": 8, "B": 32, "C": 8, "D": 20, "E": 38,
        "F": 14, "G": 20, "H": 18, "I": 16, "J": 14, "K": 20, "L": 18, "M": 16, "N": 16, "O": 22
    }
    for col_letter, width in col_widths_s2.items():
        ws2.column_dimensions[col_letter].width = width

    wb.save(out_path)
    print(f"Successfully generated clean, perfectly formatted Excel report: {out_path}")

if __name__ == "__main__":
    build_excel_report()
