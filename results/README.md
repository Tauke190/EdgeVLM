# Baseline Results

Phase 1 runs should write timestamped output folders under `results/baseline/`.

Expected per-run artifacts:

- `config.json`
- `metrics.json`
- `stage_timings.csv`
- `system_metrics.csv`
- `pred_video.mp4`
- `run_summary.txt`

Useful phase 1 commands:

- `./.venv/bin/python tools/baseline_runner.py --config configs/baseline_offline.json`
- `./.venv/bin/python tools/baseline_runner.py --config configs/baseline_offline.json --top-k-labels 1 --no-render`
- `./.venv/bin/python tools/fp16_benchmark.py --max-frames 300 --top-k-labels 1 --no-render`
- `./.venv/bin/python tools/live_baseline_runner.py --config configs/baseline_live.json --max-seconds 30`
- `./.venv/bin/python tools/compare_runs.py --baseline-run results/baseline/<run_a> --candidate-run results/baseline/<run_b>`

These run artifacts are intentionally left untracked by git. Keep only the lightweight directory structure and documentation in the repository.
