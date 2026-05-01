# EdgeVLM Live Action Detection

This repository contains a Jetson-oriented runtime for deploying SiA, a simple architecture for open-vocabulary action detection, with optional adaptive motion/person/action gating.

The project code supports:

- SiA-only inference with the adaptive pipeline disabled.
- A three-stage pipeline: motion gate, person gate, and SiA action inference.
- PyTorch FP32, PyTorch FP16, TensorRT FP16, and TensorRT INT8 backend comparisons.
- Runtime metrics, stage timing logs, event logs, predictions, and JHMDB/MEVA pilot accuracy scripts.

## Environment

The project was run with Python 3.10 in a virtual environment on Jetson Orin.

Install Python dependencies:

```bash
python -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

The runtime expects CUDA-capable PyTorch for GPU inference. TensorRT runs require engine files generated with the commands below.

## Required Weights

Expected model weights, supplied separately from the repository:

```text
weights/avak_b16_11.pt
weights/yolov8n.pt
```

SiA uses `weights/avak_b16_11.pt`. The person gate uses `weights/yolov8n.pt`.

## TensorRT Engine Generation

TensorRT `.engine` files are generated locally from the SiA checkpoint; they are not expected to be committed or supplied as repository files.

Build a TensorRT FP16 engine:

```bash
./.venv/bin/python tools/trt_vision_benchmark.py \
  --mode fp16 \
  --weights weights/avak_b16_11.pt \
  --output-dir results/tensorrt_vision/fp16_check \
  --engine-path tmp/sia_vision_fp16.engine
```

Build a TensorRT INT8 engine:

```bash
./.venv/bin/python tools/trt_vision_benchmark.py \
  --mode int8 \
  --weights weights/avak_b16_11.pt \
  --output-dir results/tensorrt_vision/int8_check \
  --engine-path results/tensorrt_vision/int8_check/sia_vision_int8.engine \
  --calibration-videos sample_videos/hit.mp4 sample_videos/hallway.avi sample_videos/SurvellienceFootage.mp4
```

The INT8 command needs representative calibration videos. If TensorRT engines are unavailable, use PyTorch modes or pass script options that skip missing TensorRT engines where supported.

## Data

Datasets are not included in the repository.

Expected local data paths, supplied separately from the repository:

```text
data/jhmdb/videos/
anno/JHMDB-GT.pkl
meva-videos-short/
sample_videos/
```

The JHMDB runner expects the frame package to be converted into repo-compatible AVI files under `data/jhmdb/videos/`.

## Common Commands

Run SiA-only offline runtime:

```bash
./.venv/bin/python tools/offline_runtime_demo.py \
  --config configs/runtime_offline_always_on.json \
  --video sample_videos/hit.mp4 \
  --backend-name pytorch \
  --precision fp32 \
  --output-dir results/runtime/example_sia_only
```

Run the full three-stage pipeline:

```bash
./.venv/bin/python tools/offline_runtime_demo.py \
  --config configs/runtime_offline_motion_person_sia.json \
  --video sample_videos/SurvellienceFootage.mp4 \
  --backend-name tensorrt \
  --precision fp16 \
  --trt-engine-path tmp/sia_vision_fp16.engine \
  --output-dir results/runtime/example_full_pipeline
```

Run the JHMDB split-0 backend accuracy sweep:

```bash
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python tools/jhmdb_runtime_accuracy.py \
  --modes direct_sia runtime_sia full_pipeline \
  --split-index 0 \
  --limit 0 \
  --backend-sweep \
  --skip-missing-trt \
  --output-dir results/jhmdb_runtime_accuracy/split0_full_backend_sweep_warm \
  --continue-on-error
```

Run the MEVA pilot pipeline on short clips:

```bash
./.venv/bin/python tools/meva_pilot_pipeline.py \
  --video-dir meva-videos-short \
  --glob 'cut_*.mp4' \
  --modes sia_only \
  --no-render \
  --continue-on-error
```

## Reproducing Report Results

The final report uses several result families. The commands below regenerate the underlying artifacts when the required local videos, weights, JHMDB annotations, MEVA annotations, and TensorRT engines are available.

Before running the report commands, confirm these prerequisites:

- Python dependencies are installed in `.venv`.
- CUDA-capable PyTorch is available.
- TensorRT and its Python bindings are available for TensorRT engine generation and inference.
- The required weights listed above are present under `weights/`.
- The local data paths listed above are populated as needed for the selected experiment.
- TensorRT engines have been generated with the commands above before running TensorRT backend sweeps.

### 1. SiA-Only and Staged Backend Tables

The SiA-only and staged backend tables compare PyTorch FP32, PyTorch FP16, TensorRT FP16, and TensorRT INT8 in live-replay mode on `sample_videos/SurvellienceFootage.mp4`.

```bash
./.venv/bin/python tools/runtime_backend_compare.py \
  --video sample_videos/SurvellienceFootage.mp4 \
  --source-modes live_replay \
  --modes always_on,motion_person_sia \
  --pytorch-precisions fp32,fp16 \
  --include-tensorrt \
  --trt-engine-path tmp/sia_vision_fp16.engine \
  --include-tensorrt-int8 \
  --trt-int8-engine-path results/tensorrt_vision/int8_check/sia_vision_int8.engine \
  --output-dir results/runtime/backend_compare_surveillance_live \
  --no-render
```

Primary output:

```text
results/runtime/backend_compare_surveillance_live/comparison_summary.csv
results/runtime/backend_compare_surveillance_live/comparison_summary.json
```

Use rows with `source_mode=live_replay` and `pipeline_mode=always_on` for the SiA-only table. Use rows with `source_mode=live_replay` and `pipeline_mode=motion_person_sia` for the staged table.

### 2. Long-Clip Full-Pipeline Duty Cycle

The long-clip duty-cycle result measures how often the full motion/person/SiA pipeline wakes the expensive SiA tier on surveillance-style video.

```bash
./.venv/bin/python tools/full_pipeline_benchmark.py \
  --config configs/full_pipeline_runtime.json \
  --video sample_videos/SurvellienceFootage.mp4 \
  --output-dir results/full_pipeline/surveillance_long \
  --no-render
```

Primary output:

```text
results/full_pipeline/surveillance_long/benchmark_summary.csv
results/full_pipeline/surveillance_long/benchmark_summary.json
```

The report's duty-cycle table uses fields such as `frames_read`, `output_ready_frames`, `active_frames`, `motion_event_count`, `person_event_count`, and `sia_activation_count`.

### 3. Minimum-On-Time Hold Comparison

The minimum-on-time table compares the staged pipeline with no hold, motion hold only, person hold only, and both holds. These runs use the TensorRT FP16 engine and disable the SiA FPS cap with `--sia-target-fps 0`.

```bash
./.venv/bin/python tools/offline_runtime_demo.py \
  --config configs/runtime_offline_motion_person_sia.json \
  --video sample_videos/SurvellienceFootage.mp4 \
  --backend-name tensorrt \
  --trt-engine-path tmp/sia_vision_fp16.engine \
  --sia-target-fps 0 \
  --motion-min-on-time 0 \
  --person-min-on-time 0 \
  --output-dir results/runtime/min_on_time_staged_0_0 \
  --no-render

./.venv/bin/python tools/offline_runtime_demo.py \
  --config configs/runtime_offline_motion_person_sia.json \
  --video sample_videos/SurvellienceFootage.mp4 \
  --backend-name tensorrt \
  --trt-engine-path tmp/sia_vision_fp16.engine \
  --sia-target-fps 0 \
  --motion-min-on-time 60 \
  --person-min-on-time 0 \
  --output-dir results/runtime/min_on_time_staged_60_0 \
  --no-render

./.venv/bin/python tools/offline_runtime_demo.py \
  --config configs/runtime_offline_motion_person_sia.json \
  --video sample_videos/SurvellienceFootage.mp4 \
  --backend-name tensorrt \
  --trt-engine-path tmp/sia_vision_fp16.engine \
  --sia-target-fps 0 \
  --motion-min-on-time 0 \
  --person-min-on-time 60 \
  --output-dir results/runtime/min_on_time_staged_0_60 \
  --no-render

./.venv/bin/python tools/offline_runtime_demo.py \
  --config configs/runtime_offline_motion_person_sia.json \
  --video sample_videos/SurvellienceFootage.mp4 \
  --backend-name tensorrt \
  --trt-engine-path tmp/sia_vision_fp16.engine \
  --sia-target-fps 0 \
  --motion-min-on-time 60 \
  --person-min-on-time 60 \
  --output-dir results/runtime/min_on_time_staged_60_60 \
  --no-render
```

Primary output for each run:

```text
results/runtime/min_on_time_staged_<motion>_<person>/metrics.json
results/runtime/min_on_time_staged_<motion>_<person>/stage_timings.csv
```

The report table uses `person_detector_runs`, `sia_inference_iterations`, `effective_active_fps`, and `timings.active_loop.mean_ms` from each `metrics.json`.

### 4. MEVA Pilot Accuracy and IoU Diagnostic

The MEVA pilot evaluates short clips under `meva-videos-short/` against MEVA annotations available outside this repository.

```bash
./.venv/bin/python tools/meva_pilot_pipeline.py \
  --video-dir meva-videos-short \
  --glob 'cut_*.mp4' \
  --modes sia_only \
  --no-render \
  --continue-on-error
```

Primary output:

```text
results/meva_pilot/<timestamp>_meva_pilot/meva_pilot_summary.csv
results/meva_pilot/<timestamp>_meva_pilot/meva_pilot_summary.json
results/meva_pilot/<timestamp>_meva_pilot/all_predictions.csv
results/meva_pilot/<timestamp>_meva_pilot/ground_truth.csv
```

The report uses the aggregate MEVA f-mAP@0.5 as a secondary surveillance-domain stress test. The one-clip IoU diagnostic is derived from the same prediction and ground-truth artifacts by comparing the runtime box to the MEVA box on the inspected frame and again after the later runtime update frame. It is a diagnostic for runtime update cadence, not the primary accuracy metric.

### 5. JHMDB Split-0 Accuracy Preservation

The JHMDB split-0 sweep reproduces the report's accuracy-preservation table. It requires:

```text
data/jhmdb/videos/
anno/JHMDB-GT.pkl
gpt/GPT_HMDB21.json
```

Run the full split-0 backend sweep:

```bash
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python tools/jhmdb_runtime_accuracy.py \
  --modes direct_sia runtime_sia full_pipeline \
  --split-index 0 \
  --limit 0 \
  --backend-sweep \
  --skip-missing-trt \
  --output-dir results/jhmdb_runtime_accuracy/split0_full_backend_sweep_warm \
  --continue-on-error
```

Primary output:

```text
results/jhmdb_runtime_accuracy/split0_full_backend_sweep_warm/jhmdb_accuracy_summary.csv
results/jhmdb_runtime_accuracy/split0_full_backend_sweep_warm/jhmdb_accuracy_summary.json
```

This is the source for the report's direct SiA, runtime SiA, and full-pipeline f-mAP@0.5 rows.

### 6. Final Plot Regeneration

After generating backend-comparison and full-pipeline summaries, assemble the final package and regenerate the plots:

```bash
./.venv/bin/python tools/final_report_packager.py \
  --backend-compare-summary results/runtime/backend_compare_surveillance_live/comparison_summary.json \
  --full-pipeline-summary results/full_pipeline/surveillance_long/benchmark_summary.json \
  --output-dir results/final_package
```

Then generate plots from the packaged CSV/JSON artifacts:

```bash
./.venv/bin/python tools/final_plot_export.py \
  --final-metrics-json results/final_package/final_metrics_table.json \
  --backend-latency-csv results/final_package/figure_data/backend_latency_by_variant.csv \
  --stage-breakdown-csv results/final_package/figure_data/stage_breakdown_by_variant.csv \
  --pipeline-activity-csv results/final_package/figure_data/pipeline_activity_summary.csv \
  --event-timeline-csv results/final_package/figure_data/full_pipeline_event_timeline.csv \
  --output-dir results/final_package/plots
```

The report figures use plots such as:

```text
results/final_package/plots/backend_inference_latency.png
results/final_package/plots/backend_sia_active_fps.png
results/final_package/plots/pipeline_duty_cycle.png
```

## Outputs

Runtime runs write artifacts such as:

```text
metrics.json
stage_timings.csv
event_log.csv
sia_inference_frames.csv
predictions.csv
system_metrics.csv
runtime_offline.mp4
```

JHMDB accuracy runs write:

```text
jhmdb_accuracy_summary.csv
jhmdb_accuracy_summary.json
```

MEVA pilot runs write:

```text
meva_pilot_summary.csv
meva_pilot_summary.json
all_predictions.csv
ground_truth.csv
```
