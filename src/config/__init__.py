"""Per-phase configuration: every tunable knob, default path and default model
lives here, one module per pipeline sub-step (mirroring `src/prompts/`).

    common.py               DATA_DIR + frame crop/resize settings (shared across sub-steps)
    phase1_segmentation.py  Phase 1 — worker action segmentation (kinematic)
    phase2_expert.py        Phase 2 / expert — expert analysis
    phase2_classify.py      Phase 2 / classify — worker segment classification
    phase2_macro.py         Phase 2 / macro — macro evaluation
    phase2_micro.py         Phase 2 / micro — micro evaluation

These modules hold only constants — no logic, no imports from the rest of
`src` — so a phase's behaviour can be re-tuned by editing one file. CLI flags
(`pipeline.py`, `python -m src.analysis.segment_classify`) still override the
defaults defined here.
"""
