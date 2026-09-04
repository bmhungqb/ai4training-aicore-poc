# AI for Training pipeline

POC pipeline that evaluates a worker's video against an expert reference video: it learns the
standard from the expert, splits the worker's footage into the same operations, and reports where
the worker is off-standard or slower than the expert, and why.

## Features

- **Segments the worker video into physical actions, with no VLM.** Kinematic pre-segmentation
  (SAM3 + SEA-RAFT optical flow) finds where the worker's hand motions actually start/stop, before
  any expert knowledge or VLM call is involved.
- **Ignores other people caught in frame.** If a video shows more than one worker/expert, an optional
  ROI mask restricts SAM3/SEA-RAFT to the intended person's area only, so someone else's hands don't
  get picked up as motion noise — draw one per video with the bundled `tools/mask_editor` web tool.
  Falls back to the full frame when no mask is present. Works for both `worker.mp4` and `expert.mp4`.
- **Learns the standard from video, not from a written SOP.** A VLM watches automatically-selected
  expert frames and writes, per scene, how the operation is performed and what the product looks like
  before / during / after.
- **Classifies the worker's action segments against the expert's operations.** Each physical segment
  from Phase 1 is labeled by matching it to the expert's ordered scenes.
- **Flags off-standard technique.** The VLM marks a segment `off_standard` with a description when
  the worker's method deviates from the expert guideline.
- **Handles footage that doesn't match anything.** Stretches the model can't attribute to any
  expected operation become `UNKNOWN` segments instead of corrupting the timeline.
- **Compares timing against the expert.** Missing / extra operations and slow / fast segments are
  computed in code — no VLM cost.
- **Explains only what needs explaining.** A second, more expensive VLM pass runs on the slow
  segments only, pinpointing which sub-step is slow, the probable cause, and a suggested fix.
- **Tracks cost.** Every VLM call's token usage and USD cost is accumulated and reported per segment
  and per run.
- **Tunable per sub-step.** Every knob, path and model default lives in one file per sub-step under
  `src/config/`.
- **Debuggable.** `--visualize` renders the kinematic boundary video/plots (Phase 1) and a
  human-scannable segment timeline (Phase 2 classify step).

## The two phases

The pipeline is two phases: **Phase 1** finds where the worker's actions are (no VLM, no expert
knowledge); **Phase 2** is everything that needs the expert video and a VLM to make sense of them.

| Phase | Sub-step | Module | Input | Output | VLM |
|---|---|---|---|---|---|
| **1. Segmentation** | kinematic | `src/segmentation/kinematic.py` | `worker.mp4` | `action_segments.json` | ❌ |
| **2. Analysis** | expert | `src/analysis/expert_analysis.py` | `expert.json` + `expert.mp4` | `selected_frames.json`, `process_knowledge.json` | ✅ |
| | classify | `src/analysis/segment_classify.py` | `action_segments.json` + `selected_frames.json` + `worker.mp4` | `worker_segments.json` (+ clips) | ✅ |
| | macro | `src/analysis/macro_eval.py` | classify output + `selected_frames.json` | `macro_eval.json` | ❌ (pure code) |
| | micro | `src/analysis/micro_eval.py` | macro's slow segments | `micro_eval.json` | ✅ |

**Phase 1 (segment)** runs the SAM3 + SEA-RAFT kinematic pipeline (`src/kinematic_pipeline/`,
invoked as a subprocess) to detect physical motion boundaries — speed valleys / direction changes in
the worker's hand movement — and writes them as plain, unlabeled `action_segments.json`. This needs
no API key and no expert video.

**Phase 2 / expert** auto-selects each scene's reference frames from `expert.mp4` (via kinematic
action segmentation, no manual curation — see [Reference frame selection](#reference-frame-selection)),
asks the VLM for a per-scene guideline (how-to steps + product state), then synthesizes every
guideline into one process-wide reference (observable traits, start/end cues, easily-confused
neighbors).

**Phase 2 / classify** takes Phase 1's action segments and, for each one, asks the VLM which of the
expert's operations it matches (or `UNKNOWN`/`IDLE`), flagging `off_standard` technique where seen.
Consecutive segments matched to the same operation are merged.

**Phase 2 / macro** matches each classified segment back to its manifest scene by operation name and
derives `slow` / `fast` / `ok` from the duration ratio, plus the missing and extra operations — pure
code, no VLM.

**Phase 2 / micro** takes only the segments macro called slow and puts expert and worker frames side
by side, asking the VLM which small step is slow, why, and how to improve it.

## Modules

```
pipeline.py                 # CLI entry point: segment | analyze | all

src/
  segmentation/              # Phase 1 — no VLM, no expert knowledge
    kinematic.py              # KinematicSegmenter (subprocess into src/kinematic_pipeline/),
                              # action_segments.json read/write helpers

  analysis/                  # Phase 2 — everything VLM-based / expert-dependent
    expert_analysis.py        # expert sub-step (+ auto_select_frames_from_kinematic — auto
                              # reference frame selection from kinematic action segmentation)
    segment_classify.py       # classify sub-step (SegmentClassifier + cut_worker_segments +
                              # dump_timeline_debug)
    macro_eval.py             # macro sub-step
    micro_eval.py             # micro sub-step

  vlm_client.py               # OpenRouterClient (chat/chat_json, retries, token + cost tracking)
  manifest.py                 # selected_frames.json helpers used by every VLM sub-step
                              # (scene ordering, scene_op_name(), expected_duration(),
                              # guideline formatting)

  config/                     # every tunable, default path and default model, one file per sub-step
    common.py                        # DATA_DIR + frame crop boxes / resize widths
    phase1_segmentation.py           # Phase 1 — kinematic tunables, paths, DEBUG_DIR (--visualize)
    phase2_expert.py                 # Phase 2 / expert — model, paths, frame-sampling knobs
    phase2_classify.py               # Phase 2 / classify — model, paths, fps knobs
    phase2_macro.py                  # Phase 2 / macro — output path, slow/fast timing ratios
    phase2_micro.py                  # Phase 2 / micro — model, output path, frame caps

  utils/
    env.py                  # .env loader (no python-dotenv dependency)
    video.py                # cut_clip() (ffmpeg); extract_frames() (uniform sample-fps over the
                            # whole video), extract_frames_in_range() (uniform sample-fps within
                            # one scene's time range), extract_frames_by_index() (specific frame
                            # numbers), sample_window_frames_cached() (timestamp-cached sampling
                            # for the classify sub-step) — all OpenCV
    frames.py               # pick_evenly_spread(), encode_expert_frame(), encode_worker_frame()
                            # (crop to the work area, resize, base64-JPEG)
    message_content.py      # image_content()/text_content(), labeled_frames(),
                            # render_template_content() (prompt-template + image splicing)

  prompts/
    expert_analysis_prompts.py       # Phase 2 / expert system/user prompts
    kinematic_classify_prompts.py    # Phase 2 / classify system/user prompts
    evaluation_prompts.py            # Phase 2 / micro system/user prompts

  kinematic_pipeline/        # vendored SAM3 + SEA-RAFT kinematic sub-pipeline, invoked as a
                             # subprocess only from src/segmentation/kinematic.py — see its own
                             # files for the physical boundary-detection internals.
                             # searaft_core/ holds the upstream SEA-RAFT inference code
                             # (github.com/princeton-vl/SEA-RAFT, BSD-3-Clause).

tools/
  mask_editor/               # standalone web tool (stdlib http.server + a plain HTML/canvas page,
                             # no extra dependencies): browse videos under a folder, draw a ROI mask
                             # for each one, save it as "<video_stem>.mask.png" next to the video —
                             # see "Optional: mask out other people in frame" below.
```

Conventions:

- `prompts/*.py` and `config/*.py` hold **only constants** — no logic.
- Anything shared by 2+ sub-steps (video cutting/sampling, frame selection/encoding, message-part
  helpers, manifest reading) lives in `utils/` or `manifest.py`, never duplicated per sub-step.
- Phases and sub-steps communicate through files on disk, so each one can be run and inspected on its
  own.

## Data flow

```
data/worker.mp4 ──► Phase 1 (segment) ──► action_segments.json
                                                │
data/expert.json ─┐                            │
data/expert.mp4 ──┴─► Phase 2 / expert ─► selected_frames.json ─┬──────────┘
                       process_knowledge.json                   │
                                                                 ▼
                                        data/worker.mp4 ──► Phase 2 / classify ──► worker_segments.json
                                                                                          │
                                                                                          ▼
                                                                       Phase 2 / macro ──► macro_eval.json
                                                                                          │ slow
                                                                                          ▼
                                                                Phase 2 / micro ──► micro_eval.json
```

## Output layout

```
data/
  kinematic/
    pipe1_report.json, action_boundaries_dynamic.npy   # raw kinematic sub-pipeline output
    action_segments.json                # Phase 1 output, read by Phase 2 / classify
    debug/                              # --visualize: annotated video + boundary plots
  expert_scenes/
    frames/scene_NN/                # frames extracted by selected_frame_indices (expert sub-step output)
    selected_frames.json            # manifest: scenes + extracted frames + guideline (expert sub-step output)
    process_knowledge.json          # process-wide reference (expert sub-step output)
  worker_frames/                    # frames sampled from worker.mp4 on demand (classify sub-step)
  worker_segments/
    worker_segments.json            # one segment per Phase-1 action, with method deviation
                                    # (off_standard) flagged by the VLM (classify sub-step output) —
                                    # timing deviation is added later, in memory, by macro
    timeline_debug.json             # --visualize: human-scannable segment timeline
    cuts/                           # one clip per segment, only with --cut
  macro_eval.json                   # missing/extra/slow/fast summary (macro sub-step output)
  micro_eval.json                   # per-slow-segment root-cause analysis (micro sub-step output)
```

# Setup and run

## 1. Install

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENROUTER_API_KEY
```

Also needs `ffmpeg` on PATH (used to cut clips), plus whatever `src/kinematic_pipeline/` needs for
kinematic segmentation (torch, transformers, scipy — see that directory).

## 2. Prepare the inputs

Place these under `data/`:

- **`data/expert.mp4`** — the expert reference video.
- **`data/worker.mp4`** — the worker video to evaluate.
- **`data/expert.json`** — scene boundaries + operation names for the expert video (see
  `expert_video[0].file_path`, `expert_video[0].scenes`, top-level `operations`). Each scene needs
  `scene_index`, `timestamp_start`, `timestamp_end` and `operations` — this is ground truth, authored
  by hand (it's the human definition of the process being taught). `scene["selected_frame_indices"]`
  does **not** need to be filled in by hand: the expert sub-step auto-selects reference frames from
  kinematic action segmentation on `expert.mp4` and overwrites this field — see
  [Reference frame selection](#reference-frame-selection) below.

The bundled `data/expert.json` already has scene boundaries/operations filled in, so it runs straight
through.

### Optional: mask out other people in frame

If `worker.mp4` or `expert.mp4` also shows another worker/expert (e.g. two stations sharing one
camera), draw a ROI mask so kinematic segmentation only looks at the intended person:

```bash
python -m tools.mask_editor.server --dir data --port 8765
# then open http://<server-ip>:8765/ in a browser
```

The page lists every video under `--dir` (recursively), shows its first frame, and lets you paint
(brush) or draw a polygon over the area to KEEP. Saving writes `<video_stem>.mask.png` next to the
video (e.g. `data/worker.mask.png`) — no other config needed, `src/segmentation/kinematic.py` picks
it up automatically the next time Phase 1 (or the expert sub-step) runs on that video. No mask file =
full frame, unchanged behavior. Pass `--mask <path>` to `pipeline.py segment` to point at a mask
outside the default naming convention.

## 3. Run

```bash
python pipeline.py all      # both phases in one process (recommended)
```

Or one phase at a time:

```bash
python pipeline.py segment  # Phase 1: worker action segmentation (no VLM)
python pipeline.py analyze  # Phase 2: expert -> classify -> macro -> micro (needs Phase 1's output)
```

Or one sub-step at a time, re-reading its inputs from disk:

```bash
python pipeline.py segment --step kinematic
python pipeline.py analyze --step expert
python pipeline.py analyze --step classify
python pipeline.py analyze --step macro
python pipeline.py analyze --step micro
```

`all` / `analyze` (no `--step`) are recommended end-to-end: later sub-steps read earlier ones' results
from memory instead of re-reading JSON from disk.

Flags:

| Flag | Effect |
|---|---|
| `--step` | run only one sub-step of the given phase |
| `--vlm-model` | model for the expert sub-step (default `google/gemini-2.5-flash`) |
| `--model` | model for the classify/macro/micro sub-steps (default `qwen/qwen3.7-plus`) |
| `--cut` | also cut `worker.mp4` into one clip per segment, into `data/worker_segments/cuts/` |
| `--save-crop-frames` | save the cropped+resized frames actually sent to the VLM, for inspection |
| `--visualize` | write debug artifacts: annotated boundary video/plots (segment phase) and a segment timeline dump (classify sub-step) |
| `--force-segment` | force re-running Phase 1 segmentation even if an existing report exists |
| `--action-segments` | override the path to Phase 1's `action_segments.json` (classify sub-step input) |
| `--force-kinematic-expert` | force re-running kinematic action segmentation on `expert.mp4` even if cached |

## 4. Read the results

`macro_eval.json` (and the printed report) is the overview: missing, extra, slow and fast operations.
`micro_eval.json` has the per-slow-segment root cause and improvement suggestion. `worker_segments.json`
keeps the full trace, including every VLM call's window and cost. `action_segments.json` has the raw,
unlabeled physical boundaries from Phase 1.

## Running one sub-step's engine directly

`segment_classify.py` is also runnable standalone, with all paths overridable:

```bash
python -m src.analysis.segment_classify --cut
python -m src.analysis.segment_classify --manifest data/expert_scenes/selected_frames.json \
    --video data/worker.mp4 --out-dir data/worker_segments --model qwen/qwen3.7-plus \
    --action-segments data/kinematic/action_segments.json
```

## Tuning a sub-step

Retune a sub-step by editing its `src/config/phase*.py` — nothing else needs to change. For example,
to widen what the macro sub-step calls "slow" and re-run kinematic segmentation with a lower angle
tolerance:

```python
# src/config/phase2_macro.py
TIMING_SLOW_RATIO = 2.0      # was 1.5

# src/config/phase1_segmentation.py
ANGLE_TOLERANCE = 45.0       # was 60.0
```

CLI flags still win over the config defaults (`--model`, `--video`, `--out-dir`, ...).

## Reference frame selection

A NEW expert video only needs `expert.json`'s scene boundaries/operations authored by hand (ground
truth — see above). Reference frame selection within each scene is automatic, no manual curation:

1. Kinematic action segmentation runs on `expert.mp4` — the same no-VLM engine Phase 1 uses for
   `worker.mp4` — producing plain action segments (start/end time, no labels). Cached under
   `data/kinematic_expert/`.
2. Each action segment is assigned to the scene whose `[timestamp_start, timestamp_end]` contains
   the segment's midpoint (a scene/operation can own several action segments).
3. For each scene, `FRAMES_PER_ACTION_SEGMENT` frames (default 2) are sampled from every action
   segment assigned to it — uniformly spaced in time, picking the **sharpest** nearby frame (variance
   of the Laplacian — a standard, model-free blur metric; see `sharpness_score()` /
   `pick_sharpest_spread()` in `src/utils/frames.py`) to dodge motion-blurred frames instead of
   picking whatever lands exactly on the time step.
4. A scene with zero action segments in its time range (very short/static scene) falls back to a
   single uniform sample over the scene's whole time range.
5. The resulting absolute frame indices are written into `expert.json` as
   `scene["selected_frame_indices"]`, then `build_selection_manifest()` extracts those exact frames
   from `expert.mp4` by index into `selected_frames.json`, same as before.

This all happens automatically as part of `python pipeline.py analyze --step expert` (or
`all`/`analyze`) — no extra pass, no manual curation needed. Use `--force-kinematic-expert` to force
re-running kinematic segmentation on `expert.mp4` instead of reusing a cached report.

Tunables: `src/config/phase2_expert.py` (`FRAMES_PER_ACTION_SEGMENT`, `SHARPNESS_POOL_FACTOR`) and
`src/config/phase1_segmentation.py` (kinematic thresholds — same knobs used for `worker.mp4`).

**Camera angle**: both `encode_expert_frame()` and `encode_worker_frame()` (`src/utils/frames.py`)
crop every frame to a fixed work-area box before sending it to the VLM — `EXPERT_CROP_BOX =
(850, 120, 1800, 900)` for the bundled 1920×1080 `expert.mp4`, `WORKER_CROP_BOX = (20, 20, 400, 320)`
for the bundled 480×368 `worker.mp4` (wide enough to still see the fabric's shape/direction, not just
the needle); both live in `src/config/common.py`. A new expert or worker video shot from a different
angle or resolution needs its box adjusted there (or passed as `crop_box=None` to send the frame
uncropped) — neither box is derived from the video automatically.
