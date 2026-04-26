# Edge Action Detection

This file documents the project-specific runtime commands for the shared offline and live benchmark paths on this repo.

## Offline Benchmark Path

The offline benchmark harness can run:

- one explicit video with `--video`
- a whole directory of videos with `--video-dir`

The shared runtime currently supports these pipeline modes:

- `always_on`
- `motion_only`
- `person_only`
- `motion_person_sia`

### Single Video

```bash
./.venv/bin/python tools/offline_benchmark_runner.py \
  --config configs/runtime_offline.json \
  --video path/to/your_video.mp4
```

### Whole Directory

```bash
./.venv/bin/python tools/offline_benchmark_runner.py \
  --config configs/runtime_offline.json \
  --video-dir sample_videos
```

### Fast Smoke Test Across A Directory

```bash
./.venv/bin/python tools/offline_benchmark_runner.py \
  --config configs/runtime_offline.json \
  --video-dir sample_videos \
  --max-frames 120 \
  --no-render
```

### Motion-Only Smoke Test

Set `pipeline_mode` to `motion_only` in the config or in a copied config file, then run:

```bash
./.venv/bin/python tools/offline_benchmark_runner.py \
  --config configs/runtime_offline.json \
  --video-dir sample_videos \
  --max-frames 120 \
  --no-render \
  --output-dir results/runtime/motion_only_smoke
```

### Person-Only Smoke Test

Use `configs/runtime_offline_person_only.json`, then run:

```bash
./.venv/bin/python tools/offline_benchmark_runner.py \
  --config configs/runtime_offline_person_only.json \
  --video-dir sample_videos \
  --max-frames 120 \
  --no-render \
  --output-dir results/runtime/person_only_smoke
```

### Motion+Person+SiA Smoke Test

Use `configs/runtime_offline_motion_person_sia.json`, then run:

```bash
./.venv/bin/python tools/offline_benchmark_runner.py \
  --config configs/runtime_offline_motion_person_sia.json \
  --video-dir sample_videos \
  --max-frames 120 \
  --no-render \
  --output-dir results/runtime/motion_person_sia_smoke
```

### Full Pipeline Benchmark

Use `configs/full_pipeline_runtime.json` with the dedicated benchmark harness:

```bash
./.venv/bin/python tools/full_pipeline_benchmark.py \
  --config configs/full_pipeline_runtime.json \
  --video-dir sample_videos \
  --max-frames 120 \
  --output-dir results/full_pipeline/manual_check
```

### Useful Offline Options

- `--recursive` to search subdirectories under `--video-dir`
- `--glob '*.avi'` or `--glob '*.mp4'` to restrict which files are picked up
- `--output-dir results/runtime/my_suite` to force a specific suite directory
- `--weights path/to/checkpoint.pt` to override the model checkpoint
- `--backend-name tensorrt --trt-engine-path path/to/sia_vision_fp16.engine` to run the shared runtime with the TensorRT vision backend
- `--resume` to skip videos that already have a matching completed run in the suite directory
- `--progress-every 60` to control how often frame progress is printed
- `--continue-on-error` to keep running remaining videos if one input fails

### Expected Offline Artifacts

For a suite run, expect:

- one suite directory under `results/runtime/`
- one subdirectory per input video

Suite-level files:

- `benchmark_summary.csv`
- `benchmark_summary.json`

Per-video files:

- `config.json`
- `metrics.json`
- `stage_timings.csv`
- `event_log.csv`
- `system_metrics.csv`
- `run_summary.txt`
- `runtime_offline.mp4` when rendering is enabled

Important note:

- if you pass `--no-render`, the benchmark still runs and writes metrics, but output videos are intentionally disabled
- `pipeline_mode` currently supports `always_on`, `motion_only`, `person_only`, and `motion_person_sia`
- `backend_name` currently supports `pytorch` and `tensorrt`
- `person_only` now defaults to `YOLOv8n` from `weights/yolov8n.pt`
- the shared offline benchmark summaries now record `output_ready_frames`, `motion_active_frames`, `person_active_frames`, and `person_detector_frames`
- full-pipeline runs now also save `event_log.csv` with scheduler transitions and gate edge events

### Offline TensorRT FP16 Example

```bash
./.venv/bin/python tools/offline_runtime_demo.py \
  --config configs/runtime_offline_motion_person_sia.json \
  --video sample_videos/hit.mp4 \
  --precision fp16 \
  --backend-name tensorrt \
  --trt-engine-path results/tensorrt_vision/fp16_check/sia_vision_fp16.engine \
  --max-frames 120 \
  --no-render
```

## Live Runtime Path

The live runtime path uses the shared runtime core on either:

- a real camera device
- a replayed video paced as if it were a live source

The default live config now targets the full gated path:

- `configs/runtime_live.json` uses `pipeline_mode = motion_person_sia`

If you want the older always-on live behavior, use:

- `configs/runtime_live_always_on.json`

### Default Camera

```bash
./.venv/bin/python tools/live_runtime_demo.py \
  --config configs/runtime_live.json
```

### Always-On Camera

```bash
./.venv/bin/python tools/live_runtime_demo.py \
  --config configs/runtime_live_always_on.json
```

### Fixed Duration

```bash
./.venv/bin/python tools/live_runtime_demo.py \
  --config configs/runtime_live.json \
  --max-seconds 30
```

### Fixed Frame Count

```bash
./.venv/bin/python tools/live_runtime_demo.py \
  --config configs/runtime_live.json \
  --max-frames 300
```

### Explicit Camera Device

```bash
./.venv/bin/python tools/live_runtime_demo.py \
  --config configs/runtime_live.json \
  --video-device 0
```

### Live Smoke Test Without Render Or Preview

```bash
./.venv/bin/python tools/live_runtime_demo.py \
  --config configs/runtime_live.json \
  --max-seconds 15 \
  --no-render
```

### Replay A Video As Live

This is the easiest way to validate the gated live path without a camera:

```bash
./.venv/bin/python tools/live_runtime_demo.py \
  --config configs/runtime_live.json \
  --video sample_videos/hit.mp4 \
  --simulate-live \
  --no-render
```

### Replay A Video As Live With Frame Dropping

Use this when you want the replay path to stay wall-clock paced even if inference falls behind:

```bash
./.venv/bin/python tools/live_runtime_demo.py \
  --config configs/runtime_live.json \
  --video sample_videos/hit.mp4 \
  --simulate-live \
  --drop-frames \
  --target-fps 30 \
  --no-render
```

### Live Replay TensorRT FP16 Example

```bash
./.venv/bin/python tools/live_runtime_demo.py \
  --config configs/runtime_live.json \
  --video sample_videos/hit.mp4 \
  --simulate-live \
  --precision fp16 \
  --backend-name tensorrt \
  --trt-engine-path results/tensorrt_vision/fp16_check/sia_vision_fp16.engine \
  --max-frames 120 \
  --no-render
```

### Expected Live Behavior

- startup logs should print the source, pipeline mode, and resolution
- a source FPS target should be printed
- if rendering is enabled, the output path should be printed
- a first-frame-received message should appear
- if rendering is enabled, a recording-started message should appear
- the run directory should contain:
  - `config.json`
  - `metrics.json`
  - `stage_timings.csv`
  - `event_log.csv`
  - `system_metrics.csv`
  - `run_summary.txt`
  - `runtime_live.mp4` when rendering is enabled

Important note:

- the live path now records scheduler transitions and gate events in `event_log.csv`
- live replay with `--simulate-live` is meant for validation, not for the final offline benchmark numbers
- the live output clears stale boxes when the pipeline goes inactive instead of leaving old detections on screen forever

## Notes

- the offline path is the main repeatable evaluation path for group benchmarking
- the live path is mainly for deployment smoke testing and qualitative camera validation
- the offline path supports `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, and `.m4v`

## Experiment Matrix

Use the experiment matrix runner when you want one summary table that records:

- gate permutation
- source mode
- precision or optimization variant
- timing deltas against an `always_on + fp32` baseline

## Backend Comparison

Use the backend comparison runner when you want one package that compares:

- PyTorch `fp32`
- PyTorch `fp16`
- TensorRT `fp16`

across the same source modes and pipeline modes, while also recording the exact engine path and cross-backend delta columns.

Example:

```bash
./.venv/bin/python tools/runtime_backend_compare.py \
  --video sample_videos/hit.mp4 \
  --source-modes offline,live_replay \
  --modes always_on,motion_person_sia \
  --pytorch-precisions fp32,fp16 \
  --include-tensorrt \
  --trt-engine-path results/tensorrt_vision/fp16_check/sia_vision_fp16.engine \
  --max-frames 120 \
  --no-render \
  --output-dir results/runtime/backend_compare_hit_smoke
```

Expected top-level artifacts:

- `comparison_summary.csv`
- `comparison_summary.json`
- `runs/`

### Offline + Live-Replay Matrix

```bash
./.venv/bin/python tools/runtime_experiment_matrix.py \
  --video sample_videos/hit.mp4 \
  --max-frames 60 \
  --no-render \
  --output-dir results/runtime/experiment_matrix_smoke
```

This writes:

- `matrix_summary.csv`
- `matrix_summary.json`
- `runs/`

Important note:

- the matrix summary records `backend_name` and `optimization_label` so future INT8 results can be compared in the same format as current FP32 and FP16 runs
- delta columns are computed against the `always_on + fp32` baseline for each source mode and backend
