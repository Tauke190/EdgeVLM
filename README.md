# EdgeVLM — SiA Action Detection

PyTorch implementation of **SiA** for open-vocabulary spatio-temporal action detection on edge devices.
> **EZACT: Efficient Open Vocabulary Action Detection** — Z.H Sia and Y.S Rawat

## Setup
```bash
conda create -n ezact python=3.10 && conda activate ezact
pip install -r requirements.txt
```
Place weights in `weights/`: `avak_b16_11.pt` (SIA) and `yolov8n.pt` (person detector).

## Quick Start
```bash
bash scripts/avinash_multi_inference.bash [output_dir]
```
Runs Motion → Person → Action on `sample_videos/bus.avi`. Results in `[output_dir]/results/`.

**Custom video:**
```bash
python multi-tier_inference_avinash.py -F <video.mp4> --sia-precision fp16 --output-dir output/
```

| Flag | Default | Description |
|------|---------|-------------|
| `-F` | — | **Required.** Input video |
| `--act-map-file` | `gpt/MEVA_to_GPT_AVA.json` | Action label mapping |
| `--sia-precision` | `fp32` | Use `fp16` on edge GPUs |
| `--person-weights` | `weights/yolov8n.pt` | Accepts TensorRT `.engine` |
| `--action-stride` | `1` | Run action every N frames |
| `--skip-tier3` | off | Motion + person only |
| `--output-dir` | `.` | Output directory |
| `--debug` | off | Overlay tier states |

## Training
```bash
python train_avak.py -SIZE b16 -TRAIN AVAK \
  -AVA <ava_dir> -KINETICS <k700_dir> --TXTAUG --TXTLORA -JSON stats.json
```

## Evaluation
```bash
python val_avak.py -SIZE b16 -VAL UCF24 \
  -UCF24 <ucf24_dir> -ANNOUCF24 datasets/anno/UCF101v2-GT.pkl -JSON results.json
```
