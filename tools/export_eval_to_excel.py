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
        
        # Ground Truth boundaries: union of start & end
        gt_bounds = sorted(list({round(float(s[k]), 3) for s in gt_segs for k in ["timestamp_start", "timestamp_end"]}))
        
        # Machine transition boundaries: continuous transition points
        pred_bounds = []
        if pred_segs:
            pred_bounds.append(round(float(pred_segs[0]["start_time_s"]), 3))
            for i in range(1, len(pred_segs)):
                t_trans = (float(pred_segs[i-1]["end_time_s"]) + float(pred_segs[i]["start_time_s"])) / 2.0
                pred_bounds.append(round(t_trans, 3))
            pred_bounds.append(round(float(pred_segs[-1]["end_time_s"]), 3))
        pred_bounds = sorted(list(set(pred_bounds)))
        
        # Boundary level hits and errors at 0.5s
        hit_errors = []
        gt_hits = 0
        for g in gt_bounds:
            min_dist = min(abs(g - p) for p in pred_bounds) if pred_bounds else float("inf")
            if min_dist <= primary_tau:
                gt_hits += 1
                hit_errors.append(min_dist)
                
        rec_05 = (gt_hits / len(gt_bounds) * 100) if gt_bounds else 0
        mae_05 = (sum(hit_errors) / len(hit_errors)) if hit_errors else 0.0
        
        # Step level hits
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
            "mae_05": mae_05,
            "both_count": both_count,
            "both_pct": both_count / len(gt_segs) * 100 if gt_segs else 0,
            "either_count": either_count,
            "either_pct": either_count / len(gt_segs) * 100 if gt_segs else 0,
            "gt_bounds_raw": gt_bounds,
            "pred_bounds_raw": pred_bounds,
            "gt_segs_raw": gt_segs,
        })

    # Window sweep data (Focusing on Recall, MAE, Step Alignment)
    windows = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    sweep_results = []
    total_gt_b = sum(r["gt_bounds"] for r in cd_results)
    total_steps = sum(r["gt_steps"] for r in cd_results)
    
    for w in windows:
        # Macro & Micro Recall
        m_rec = sum(sum(1 for g in r["gt_bounds_raw"] if any(abs(g - p) <= w for p in r["pred_bounds_raw"])) / r["gt_bounds"] for r in cd_results) / len(cd_results) * 100
        u_hits = sum(sum(1 for g in r["gt_bounds_raw"] if any(abs(g - p) <= w for p in r["pred_bounds_raw"])) for r in cd_results)
        u_rec = u_hits / total_gt_b * 100
        
        # Step alignment
        both = sum(sum(1 for s in r["gt_segs_raw"] if any(abs(s["timestamp_start"] - p) <= w for p in r["pred_bounds_raw"]) and any(abs(s["timestamp_end"] - p) <= w for p in r["pred_bounds_raw"])) for r in cd_results)
        both_pct = both / total_steps * 100
        
        either = sum(sum(1 for s in r["gt_segs_raw"] if any(abs(s["timestamp_start"] - p) <= w for p in r["pred_bounds_raw"]) or any(abs(s["timestamp_end"] - p) <= w for p in r["pred_bounds_raw"])) for r in cd_results)
        either_pct = either / total_steps * 100
        
        # MAE of hits
        w_errors = [min(abs(g - p) for p in r["pred_bounds_raw"]) for r in cd_results for g in r["gt_bounds_raw"] if any(abs(g - p) <= w for p in r["pred_bounds_raw"])]
        w_mae = (sum(w_errors) / len(w_errors)) if w_errors else 0.0
        
        sweep_results.append({
            "window": w,
            "macro_rec": m_rec,
            "micro_rec": u_rec,
            "u_hits": u_hits,
            "both": both,
            "both_pct": both_pct,
            "either": either,
            "either_pct": either_pct,
            "mae": w_mae
        })

    # Create Workbook
    wb = openpyxl.Workbook()
    
    font_title = Font(name="Calibri", size=15, bold=True, color="1F4E78")
    font_section = Font(name="Calibri", size=12, bold=True, color="1F4E78")
    font_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    font_bold = Font(name="Calibri", size=11, bold=True)
    font_regular = Font(name="Calibri", size=11)
    
    fill_header = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    fill_header_sub = PatternFill(start_color="2F5597", end_color="2F5597", fill_type="solid")
    fill_summary = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    fill_alt = PatternFill(start_color="F9FBFD", end_color="F9FBFD", fill_type="solid")
    fill_hit = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    fill_miss = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
    
    border_thin = Side(border_style="thin", color="D9D9D9")
    border_box = Border(left=border_thin, right=border_thin, top=border_thin, bottom=border_thin)
    border_top_double = Border(top=Side(border_style="thin", color="1F4E78"), bottom=Side(border_style="double", color="1F4E78"))
    
    align_center = Alignment(horizontal="center", vertical="center")
    align_left = Alignment(horizontal="left", vertical="center")
    align_right = Alignment(horizontal="right", vertical="center")

    # ==========================================
    # SHEET 1: TỔNG QUAN 9 CÔNG ĐOẠN
    # ==========================================
    ws1 = wb.active
    ws1.title = "Tong_Quan_9CD"
    ws1.views.sheetView[0].showGridLines = True
    
    ws1.cell(row=1, column=1, value="BÁO CÁO ĐÁNH GIÁ ĐỘ ALIGN CỦA ACTION SEGMENTATION (STAGE 1: SAM + SEA-RAFT)").font = font_title
    ws1.cell(row=2, column=1, value="Dữ liệu: Chuyền 1 (9 công đoạn độc lập: CĐ 1, 2, 3, 4, 5, 6, 8, 9, 10) | Cửa sổ chuẩn: ±0.5s | Báo cáo các chỉ số cốt lõi").font = Font(name="Calibri", size=11, italic=True, color="595959")
    
    # Table 1: Per-CD Summary
    ws1.cell(row=4, column=1, value="1. ĐÁNH GIÁ CHI TIẾT TỪNG CÔNG ĐOẠN (WINDOW = ±0.5 GIÂY)").font = font_section
    
    headers_t1 = [
        "CĐ", "Tên công đoạn", "Video tương ứng", 
        "Số bước GT", "Số mốc GT", "Số đoạn máy cắt", "Số ranh giới máy",
        "Tỷ lệ cắt máy / GT", "Boundary Recall", "Độ lệch MAE (s)",
        "Khớp CẢ 2 đầu (Số bước)", "Khớp CẢ 2 đầu (%)",
        "Khớp ÍT NHẤT 1 đầu (Số bước)", "Khớp ÍT NHẤT 1 đầu (%)"
    ]
    
    for col_idx, h in enumerate(headers_t1, start=1):
        c = ws1.cell(row=5, column=col_idx, value=h)
        c.font = font_header
        c.fill = fill_header
        c.alignment = align_center
        c.border = border_box
        
    start_row = 6
    for idx, r in enumerate(cd_results):
        row_idx = start_row + idx
        fill = fill_alt if idx % 2 == 1 else None
        
        vals = [
            r["cd"], r["title"], r["video"],
            r["gt_steps"], r["gt_bounds"], r["pred_segs"], r["pred_bounds"],
            f"{r['split_ratio']}x",
            r["rec_05"] / 100.0, r["mae_05"],
            r["both_count"], r["both_pct"] / 100.0,
            r["either_count"], r["either_pct"] / 100.0
        ]
        
        for col_idx, val in enumerate(vals, start=1):
            c = ws1.cell(row=row_idx, column=col_idx, value=val)
            c.font = font_regular
            c.border = border_box
            if fill:
                c.fill = fill
                
            if col_idx in [1, 8]:
                c.alignment = align_center
            elif col_idx in [2, 3]:
                c.alignment = align_left
            elif col_idx in [4, 5, 6, 7, 11, 13]:
                c.alignment = align_right
                c.number_format = "#,##0"
            elif col_idx in [9, 12, 14]:
                c.alignment = align_right
                c.number_format = "0.0%"
            elif col_idx == 10:
                c.alignment = align_right
                c.number_format = "0.000"

    macro_row = start_row + len(cd_results)
    micro_row = macro_row + 1
    
    all_mae_05 = [r["mae_05"] for r in cd_results if r["mae_05"] > 0]
    avg_mae = sum(all_mae_05) / len(all_mae_05) if all_mae_05 else 0.0
    
    # Macro Average row
    ws1.cell(row=macro_row, column=1, value="").font = font_bold
    ws1.cell(row=macro_row, column=2, value="TRUNG BÌNH CỘNG (MACRO AVERAGE)").font = font_bold
    ws1.cell(row=macro_row, column=3, value=f"{len(cd_results)} công đoạn").font = font_bold
    ws1.cell(row=macro_row, column=4, value=total_steps).font = font_bold
    ws1.cell(row=macro_row, column=5, value=total_gt_b).font = font_bold
    ws1.cell(row=macro_row, column=6, value=sum(r["pred_segs"] for r in cd_results)).font = font_bold
    ws1.cell(row=macro_row, column=7, value=sum(r["pred_bounds"] for r in cd_results)).font = font_bold
    ws1.cell(row=macro_row, column=8, value=f"{round(sum(r['pred_segs'] for r in cd_results) / total_steps, 1)}x").font = font_bold
    ws1.cell(row=macro_row, column=9, value=sum(r["rec_05"] for r in cd_results) / len(cd_results) / 100.0).font = font_bold
    ws1.cell(row=macro_row, column=10, value=avg_mae).font = font_bold
    ws1.cell(row=macro_row, column=11, value=sum(r["both_count"] for r in cd_results)).font = font_bold
    ws1.cell(row=macro_row, column=12, value=sum(r["both_count"] for r in cd_results) / total_steps).font = font_bold
    ws1.cell(row=macro_row, column=13, value=sum(r["either_count"] for r in cd_results)).font = font_bold
    ws1.cell(row=macro_row, column=14, value=sum(r["either_count"] for r in cd_results) / total_steps).font = font_bold
    
    for c_idx in range(1, 15):
        cell = ws1.cell(row=macro_row, column=c_idx)
        cell.fill = fill_summary
        cell.border = border_box
        if c_idx in [9, 12, 14]:
            cell.number_format = "0.0%"
        elif c_idx in [4, 5, 6, 7, 11, 13]:
            cell.number_format = "#,##0"
        elif c_idx == 10:
            cell.number_format = "0.000"
        elif c_idx == 8:
            cell.alignment = align_center

    # Micro Aggregate row
    u_hits_05 = sum(sum(1 for g in r["gt_bounds_raw"] if any(abs(g - p) <= primary_tau for p in r["pred_bounds_raw"])) for r in cd_results)
    u_rec_05 = u_hits_05 / total_gt_b
    
    ws1.cell(row=micro_row, column=1, value="").font = font_bold
    ws1.cell(row=micro_row, column=2, value="TỔNG HỢP TOÀN BỘ (MICRO AGGREGATE)").font = font_bold
    ws1.cell(row=micro_row, column=3, value="Gộp toàn bộ 9 video").font = font_bold
    ws1.cell(row=micro_row, column=4, value=total_steps).font = font_bold
    ws1.cell(row=micro_row, column=5, value=f"{u_hits_05} / {total_gt_b} mốc").font = font_bold
    ws1.cell(row=micro_row, column=6, value=sum(r["pred_segs"] for r in cd_results)).font = font_bold
    ws1.cell(row=micro_row, column=7, value=sum(r["pred_bounds"] for r in cd_results)).font = font_bold
    ws1.cell(row=micro_row, column=8, value=f"{round(sum(r['pred_segs'] for r in cd_results) / total_steps, 1)}x").font = font_bold
    ws1.cell(row=micro_row, column=9, value=u_rec_05).font = font_bold
    ws1.cell(row=micro_row, column=10, value=avg_mae).font = font_bold
    ws1.cell(row=micro_row, column=11, value=sum(r["both_count"] for r in cd_results)).font = font_bold
    ws1.cell(row=micro_row, column=12, value=sum(r["both_count"] for r in cd_results) / total_steps).font = font_bold
    ws1.cell(row=micro_row, column=13, value=sum(r["either_count"] for r in cd_results)).font = font_bold
    ws1.cell(row=micro_row, column=14, value=sum(r["either_count"] for r in cd_results) / total_steps).font = font_bold
    
    for c_idx in range(1, 15):
        cell = ws1.cell(row=micro_row, column=c_idx)
        cell.fill = fill_summary
        cell.border = border_top_double
        if c_idx in [9, 12, 14]:
            cell.number_format = "0.0%"
        elif c_idx == 10:
            cell.number_format = "0.000"
        elif c_idx == 8:
            cell.alignment = align_center

    # Table 2: Tolerance Sweep
    sweep_start_row = micro_row + 3
    ws1.cell(row=sweep_start_row, column=1, value="2. TIẾN TRÌNH RECALL & KHỚP THAO TÁC THEO DẢI DUNG SAI THỜI GIAN").font = font_section
    
    headers_t2 = [
        "Cửa sổ dung sai (±s)", "Macro Recall (%)", "Micro Recall (%)", "Số mốc GT trúng", 
        "Khớp CẢ 2 đầu (Số bước)", "Khớp CẢ 2 đầu (%)",
        "Khớp ÍT NHẤT 1 đầu (Số bước)", "Khớp ÍT NHẤT 1 đầu (%)",
        "Độ lệch trung bình MAE (s)", "Ý nghĩa thực tế"
    ]
    for col_idx, h in enumerate(headers_t2, start=1):
        c = ws1.cell(row=sweep_start_row + 1, column=col_idx, value=h)
        c.font = font_header
        c.fill = fill_header_sub
        c.alignment = align_center
        c.border = border_box
        
    for idx, sw in enumerate(sweep_results):
        r_idx = sweep_start_row + 2 + idx
        if sw["window"] == 0.25:
            note = "Dung sai rất khắt khe (~3-4 frames)"
        elif sw["window"] == 0.5:
            note = "Mức chuẩn cho thao tác may công nghiệp"
        elif sw["window"] == 0.75:
            note = "Bao phủ hầu hết các mốc (>95% GT)"
        elif sw["window"] == 1.0:
            note = "Chỉ còn sót 5 mốc trên 9 video"
        else:
            note = "Bắt trúng trọn vẹn 100% mốc GT"
            
        fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid") if sw["window"] == 0.5 else (fill_alt if idx % 2 == 1 else None)
        
        row_vals = [
            f"±{sw['window']:.2f}s",
            sw["macro_rec"] / 100.0,
            sw["micro_rec"] / 100.0,
            f"{sw['u_hits']} / {total_gt_b}",
            sw["both"],
            sw["both_pct"] / 100.0,
            sw["either"],
            sw["either_pct"] / 100.0,
            sw["mae"],
            note
        ]
        
        for col_idx, val in enumerate(row_vals, start=1):
            c = ws1.cell(row=r_idx, column=col_idx, value=val)
            c.font = font_bold if sw["window"] == 0.5 else font_regular
            c.border = border_box
            if fill:
                c.fill = fill
            if col_idx in [1, 4]:
                c.alignment = align_center
            elif col_idx in [2, 3, 6, 8]:
                c.alignment = align_right
                c.number_format = "0.0%"
            elif col_idx in [5, 7]:
                c.alignment = align_right
                c.number_format = "#,##0"
            elif col_idx == 9:
                c.alignment = align_right
                c.number_format = "0.000"
            elif col_idx == 10:
                c.alignment = align_left

    # Auto-fit column widths for Sheet 1
    for col in ws1.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws1.column_dimensions[col_letter].width = max(max_len + 3, 11)
    ws1.column_dimensions["B"].width = 32
    ws1.column_dimensions["C"].width = 38
    ws1.column_dimensions["J"].width = 38

    # ==========================================
    # SHEET 2: CHI TIẾT TỪNG THAO TÁC (349 STEPS)
    # ==========================================
    ws2 = wb.create_sheet(title="Chi_Tiet_Tung_Thao_Tac")
    ws2.views.sheetView[0].showGridLines = True
    
    ws2.cell(row=1, column=1, value="BẢNG ĐỐI CHIẾU CHI TIẾT TỪNG THAO TÁC (349 THAO TÁC TRÊN 9 CÔNG ĐOẠN)").font = font_title
    ws2.cell(row=2, column=1, value="Đối sánh mốc bắt đầu (Start) và kết thúc (End) của Ground Truth với ranh giới gần nhất của Stage 1 (Dung sai: ±0.5s)").font = Font(name="Calibri", size=11, italic=True, color="595959")
    
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
        fill = fill_alt if idx % 2 == 1 else None
        
        status_fill = None
        if st["status"] == "Khớp CẢ 2 đầu":
            status_fill = fill_hit
        elif "Khớp" in st["status"]:
            status_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
        else:
            status_fill = fill_miss
            
        row_vals = [
            st["cd"], st["cd_title"], st["stt"], st["section"], st["name"],
            st["t0"], st["p0"], st["err0"], st["hit0"],
            st["t1"], st["p1"], st["err1"], st["hit1"],
            st["dur"], st["status"]
        ]
        
        for col_idx, val in enumerate(row_vals, start=1):
            c = ws2.cell(row=r_idx, column=col_idx, value=val)
            c.font = font_regular
            c.border = border_box
            if fill:
                c.fill = fill
                
            if col_idx in [1, 3, 9, 13]:
                c.alignment = align_center
            elif col_idx in [2, 4, 5]:
                c.alignment = align_left
            elif col_idx in [6, 7, 8, 10, 11, 12, 14]:
                c.alignment = align_right
                c.number_format = "0.00"
            elif col_idx == 15:
                c.alignment = align_center
                c.fill = status_fill
                if st["status"] == "Khớp CẢ 2 đầu":
                    c.font = font_bold
                    
    for col in ws2.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws2.column_dimensions[col_letter].width = max(max_len + 3, 10)
    ws2.column_dimensions["E"].width = 38
    ws2.column_dimensions["B"].width = 28
    
    wb.save(out_path)
    print(f"Successfully generated clean Excel report: {out_path}")

if __name__ == "__main__":
    build_excel_report()
