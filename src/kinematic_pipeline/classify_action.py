import os
import sys
import cv2
import base64
import requests
import json
import time
import numpy as np

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Define the classes & discriminative criteria
CLASSES = [
    "process/manipulate: The core transformative action is actively occurring (e.g., active sewing, cutting, shaping). Fabric is advancing through the needle/presser foot, and hands are feeding, pushing, or guiding the fabric as the machine runs. NOTE: Guiding the material edge WHILE stitching counts as 'process/manipulate', NOT 'adjust/align'.",
    "adjust/align: The machine is PAUSED / STOPPED. Hands are repositioning, smoothing wrinkles, matching edges together, or placing the piece under the presser foot in preparation before stitching begins.",
    "move/transfer: Hands or material travel across space with no sewing or transformation occurring (e.g., reaching for a tool/scissors, picking up raw fabric from a bin, rotating/flipping the entire large garment, moving finished part away).",
    "check/end: Visually or physically inspecting the completed seam for defects, trimming loose threads, or releasing the finished piece at the end of the action cycle."
]

def analyze_frames(frames, motion_info: dict | None = None, model="qwen/qwen3.7-flash", max_retries=3):
    motion_context_str = ""
    if motion_info:
        l_ang = motion_info.get("left_mean_angle")
        l_spd = motion_info.get("left_mean_speed")
        r_ang = motion_info.get("right_mean_angle")
        r_spd = motion_info.get("right_mean_speed")
        trigger = motion_info.get("trigger_type", "")
        
        l_str = f"Angle θ={l_ang:+.0f}°, Speed={l_spd:.2f} px/f" if l_ang is not None else "IDLE/Stationary"
        r_str = f"Angle θ={r_ang:+.0f}°, Speed={r_spd:.2f} px/f" if r_ang is not None else "IDLE/Stationary"
        
        motion_context_str = f"""
### MULTI-MODAL MOTION SIGNALS (MEASURED FROM OPTICAL FLOW):
- Left Hand Kinematics: {l_str}
- Right Hand Kinematics: {r_str}
- Boundary Trigger: {trigger}
Use these physical motion measurements to accurately confirm whether material is advancing under the needle (process/manipulate) or stopped/stationary (adjust/align).
"""

    prompt_text = f"""You are an expert industrial engineering AI analyzing video frames of a manual sewing/manufacturing workstation.
Classify the physical hand movement sequence into EXACTLY ONE of the 4 standard industrial work classes:
{motion_context_str}
### CLASSES & DISCRIMINATIVE RULES:
{chr(10).join(['- ' + cls for cls in CLASSES])}

### DECISION FLOW TO AVOID BIAS:
- Q1: Is the machine actively sewing / is fabric being fed through the needle? ➜ "process/manipulate"
- Q2: Is the machine stopped while hands realign, smooth, or match fabric edges? ➜ "adjust/align"
- Q3: Are hands reaching across space, grabbing raw material, or picking up tools away from the needle? ➜ "move/transfer"
- Q4: Is the worker inspecting the finished seam quality or concluding the step? ➜ "check/end"

Output strictly in JSON format with exactly two keys: "description" (detailed physical description of hand motions) and "class" (one of: "process/manipulate", "adjust/align", "move/transfer", "check/end")."""

    content = [{"type": "text", "text": prompt_text}]

    for frame in frames:
        # Resize to maintain max side of 1024 for API limits to save bandwidth/tokens
        h, w = frame.shape[:2]
        max_dim = 1024
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        base64_image = base64.b64encode(buffer).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{base64_image}"
        
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        })

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            content_resp = result['choices'][0]['message']['content']
            usage = result.get('usage', {})
            return content_resp, usage
        except Exception as e:
            print(f"  [Attempt {attempt+1}/{max_retries}] API call error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                return None, None

def extract_json(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())

def process_video(video_path, frames_per_chunk=10, test_one=False):
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY environment variable is not set.")
        return
        
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25.0 # Default if unable to read
        
    print(f"Processing video {video_path} (FPS: {fps:.2f}), Chunk Size: {frames_per_chunk} frames")
    
    output_file = f"{os.path.splitext(video_path)[0]}_results.json"
    results = []
    
    frame_count = 0
    chunk_frames = []
    start_frame_for_chunk = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Process remaining frames if any
            if chunk_frames:
                process_chunk(chunk_frames, start_frame_for_chunk, fps, results)
            break
            
        chunk_frames.append(frame)
        
        if len(chunk_frames) == frames_per_chunk:
            print(f"Analyzing chunk from frame {start_frame_for_chunk} to {frame_count} ({(start_frame_for_chunk)/fps:.2f}s - {frame_count/fps:.2f}s)...")
            process_chunk(chunk_frames, start_frame_for_chunk, fps, results)
            
            if test_one:
                print("Test mode: stopping after 1 chunk.")
                break
                
            chunk_frames = []
            start_frame_for_chunk = frame_count + 1
            
            # Save incrementally
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            sys.stdout.flush()
            
            # Rate limiting sleep
            time.sleep(1)
            
        frame_count += 1
        
    cap.release()
    
    # Final save just in case
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Finished processing. Results saved to {output_file}")
    sys.stdout.flush()

def sample_segment_frames(start_f, end_f, min_frames=5, max_frames=15):
    """
    Lấy mẫu các frames trong 1 action segment:
    - Số lượng: tối thiểu min_frames, tối đa max_frames.
    - Phân bố: đầu (start_f), giữa (các frame trung gian), cuối (end_f).
    """
    total_seg = end_f - start_f + 1
    if total_seg <= 0:
        return []
    target_count = min(max_frames, max(min_frames, total_seg))
    # np.linspace đảm bảo điểm đầu = start_f, điểm cuối = end_f, các điểm giữa phân bố đều
    indices = np.linspace(start_f, end_f, num=target_count, endpoint=True)
    indices = np.round(indices).astype(int)
    # Loại bỏ frame trùng nếu segment quá ngắn nhưng giữ nguyên thứ tự
    unique_indices = sorted(list(dict.fromkeys(indices)))
    return unique_indices


def classify_segmented_actions(video_path, boundaries, motion_data=None, fps=None, min_frames=5, max_frames=15,
                                model="qwen/qwen3.7-flash", output_file=None):
    """
    Classify each segmented action independently using the VLM, with Multi-Modal Motion signals.
    boundaries: list of boundary frame indices or path to direction_boundaries.npy / action_boundaries.npy
    """
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY environment variable is not set.")
        return []

    if isinstance(boundaries, (str, os.PathLike)):
        b_data = np.load(boundaries, allow_pickle=True).item()
        boundaries = b_data.get("multimodal_boundaries") or b_data.get("boundaries", [])
        motion_data = b_data
        if fps is None:
            fps = float(b_data.get("fps", 25.0))
    elif isinstance(boundaries, dict):
        motion_data = boundaries
        boundaries = motion_data.get("multimodal_boundaries") or motion_data.get("boundaries", [])
        if fps is None:
            fps = float(motion_data.get("fps", 25.0))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return []

    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Build seamless segment edges [0, b1, b2, ..., N]
    edges = [0] + sorted([int(b) for b in boundaries if 0 < int(b) < total_frames]) + [total_frames]
    edges = sorted(list(dict.fromkeys(edges)))
    n_segments = len(edges) - 1

    print(f"\nClassifying {n_segments} segmented actions with Multi-Modal Kinematics in {video_path} (FPS: {fps:.2f})...")
    results = []

    for seg_idx in range(n_segments):
        start_f = edges[seg_idx]
        end_f   = edges[seg_idx + 1] - 1
        dur     = (end_f - start_f + 1) / max(fps, 1.0)

        # Trích xuất thông tin động học (Multi-Modal Kinematic prior) cho segment này
        motion_info = {}
        if motion_data:
            l_mean_angs = motion_data.get("left_mean_angles")
            r_mean_angs = motion_data.get("right_mean_angles")
            l_spds = motion_data.get("left_speeds")
            r_spds = motion_data.get("right_speeds")
            b_details = motion_data.get("boundary_details", [])

            if l_mean_angs is not None and len(l_mean_angs) > end_f:
                valid_l = l_mean_angs[start_f:end_f + 1]
                valid_l = valid_l[np.isfinite(valid_l)]
                motion_info["left_mean_angle"] = float(np.mean(valid_l)) if len(valid_l) > 0 else None
            if r_mean_angs is not None and len(r_mean_angs) > end_f:
                valid_r = r_mean_angs[start_f:end_f + 1]
                valid_r = valid_r[np.isfinite(valid_r)]
                motion_info["right_mean_angle"] = float(np.mean(valid_r)) if len(valid_r) > 0 else None
            if l_spds is not None and len(l_spds) > end_f:
                motion_info["left_mean_speed"] = float(np.mean(l_spds[start_f:end_f + 1]))
            if r_spds is not None and len(r_spds) > end_f:
                motion_info["right_mean_speed"] = float(np.mean(r_spds[start_f:end_f + 1]))

            matched_det = next((d for d in b_details if d.get("frame") == edges[seg_idx + 1]), None)
            if matched_det:
                motion_info["trigger_type"] = matched_det.get("type", "JOINT_MAG_DIR")

        # Lấy mẫu theo phân bố đầu - giữa - cuối (min 5, max 15)
        sample_indices = sample_segment_frames(start_f, end_f, min_frames=min_frames, max_frames=max_frames)

        frames = []
        for fi in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        print(f"Segment {seg_idx+1}/{n_segments}: frames {start_f}–{end_f} ({dur:.2f}s) | {len(frames)} frames {sample_indices} → VLM...")

        result_entry = {
            "segment":             seg_idx,
            "start_frame":         start_f,
            "end_frame":           end_f,
            "start_timestamp_sec": round(start_f / fps, 3),
            "end_timestamp_sec":   round(end_f   / fps, 3),
            "duration_sec":        round(dur, 3),
            "motion_info":         motion_info,
            "class":               "unknown",
            "description":         "",
            "estimated_cost_usd":  0.0,
        }

        if frames:
            result_text, usage = analyze_frames(frames, motion_info=motion_info, model=model)
            if result_text:
                try:
                    parsed = extract_json(result_text)
                    result_entry["class"] = parsed.get("class", "unknown")
                    result_entry["description"] = parsed.get("description", "")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        result_entry["estimated_cost_usd"] = (prompt_tokens + completion_tokens) / 1_000_000 * 0.4
                        result_entry["tokens_usage"] = usage
                    print(f"  ➜ Class: [{result_entry['class']}] {result_entry['description'][:60]}")
                except Exception as e:
                    print(f"  Could not parse JSON output: {e}")
        results.append(result_entry)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def reclassify_unknown_segments(video_path, results_json_path, min_frames=5, max_frames=15,
                                model="qwen/qwen3.7-flash"):
    """
    Find segments with class == 'unknown' in results_json_path and re-classify them.
    """
    with open(results_json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    unknown_indices = [i for i, r in enumerate(results) if r.get("class") == "unknown" or not r.get("class")]
    if not unknown_indices:
        print(f"No 'unknown' segments found in {results_json_path}.")
        return results

    print(f"Found {len(unknown_indices)} unknown segments to reclassify: {unknown_indices}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    for i in unknown_indices:
        entry = results[i]
        start_f = entry["start_frame"]
        end_f = entry["end_frame"]
        dur = (end_f - start_f + 1) / max(fps, 1.0)

        sample_indices = sample_segment_frames(start_f, end_f, min_frames=min_frames, max_frames=max_frames)
        frames = []
        for fi in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        print(f"\nReclassifying Segment {entry.get('segment', i)+1}: frames {start_f}–{end_f} ({dur:.2f}s) | {len(frames)} frames {sample_indices} → VLM...")

        if frames:
            result_text, usage = analyze_frames(frames, model=model)
            if result_text:
                try:
                    parsed = extract_json(result_text)
                    entry["class"] = parsed.get("class", "unknown")
                    entry["description"] = parsed.get("description", "")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        entry["estimated_cost_usd"] = (prompt_tokens + completion_tokens) / 1_000_000 * 0.4
                        entry["tokens_usage"] = usage
                    print(f"  ➜ Updated: [{entry['class']}] {entry['description'][:60]}")
                except Exception as e:
                    print(f"  Parse error: {e}")
            time.sleep(1)

        # Save progress
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    cap.release()
    print(f"\nFinished reclassifying. Saved to {results_json_path}")
    return results

if __name__ == "__main__":
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser(description="Classify worker actions in a video.")
    parser.add_argument("video_file", help="Path to the video file")
    parser.add_argument("--boundaries", default=None, help="Path to action_boundaries.npy (for segmented classification)")
    parser.add_argument("--reclassify-json", default=None, help="Path to results JSON file with 'unknown' segments to reclassify")
    parser.add_argument("--min-frames", type=int, default=5, help="Minimum frames per segment (default: 5)")
    parser.add_argument("--max-frames", type=int, default=15, help="Maximum frames per segment (default: 15)")
    parser.add_argument("--chunk-size", type=int, default=10, help="Number of frames for fixed chunking mode (default: 10)")
    parser.add_argument("--test-one", action="store_true", help="Test mode: Process only the first chunk and exit")
    parser.add_argument("--model", default="qwen/qwen3.7-flash", help="VLM Model name")
    
    args = parser.parse_args()
    
    if args.reclassify_json:
        reclassify_unknown_segments(args.video_file, args.reclassify_json,
                                   min_frames=args.min_frames,
                                   max_frames=args.max_frames,
                                   model=args.model)
    elif args.boundaries:
        out_file = f"{os.path.splitext(args.video_file)[0]}_segmented_results.json"
        classify_segmented_actions(args.video_file, args.boundaries,
                                  min_frames=args.min_frames,
                                  max_frames=args.max_frames,
                                  model=args.model, output_file=out_file)
    else:
        process_video(args.video_file, args.chunk_size, args.test_one)

"""
python action_segment_direction.py \
    --flows output/test_run/video_10s_flow.npz \
    --masks output/test_run/video_10s_masks.npz \
    --video videos/video_10s.mp4 \
    --output output/test_run
"""