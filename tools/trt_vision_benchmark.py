import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SYSTEM_DIST_PACKAGES = "/usr/lib/python3.10/dist-packages"
if SYSTEM_DIST_PACKAGES not in sys.path:
    sys.path.append(SYSTEM_DIST_PACKAGES)

import tensorrt as trt

from runtime.buffer import SlidingWindowBuffer
from runtime.preprocess import build_normalizer, resize_frame
from sia import get_sia
from tools.baseline_utils import ensure_dir, infer_git_commit, write_json


TRTEXEC_PATH = Path("/usr/src/tensorrt/bin/trtexec")
DEFAULT_CALIBRATION_SOURCES = [
    "sample_videos/hit.mp4",
    "sample_videos/hallway.avi",
    "sample_videos/SurvellienceFootage.mp4",
]


class VisionWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model.encode_vision(x)
        return outputs["pred_logits"], outputs["pred_boxes"], outputs["human_logits"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export, build, and benchmark the fixed-shape TensorRT vision path for SiA."
    )
    parser.add_argument(
        "--mode",
        choices=["fp16", "int8"],
        required=True,
        help="Engine precision mode to build and benchmark.",
    )
    parser.add_argument("--weights", default="weights/avak_b16_11.pt", help="Path to the SiA checkpoint.")
    parser.add_argument(
        "--output-dir",
        default="results/tensorrt_vision",
        help="Directory for exported ONNX, engines, and benchmark artifacts.",
    )
    parser.add_argument("--onnx-path", help="Optional explicit ONNX output path.")
    parser.add_argument("--engine-path", help="Optional explicit TensorRT engine output path.")
    parser.add_argument("--skip-export", action="store_true", help="Reuse an existing ONNX graph.")
    parser.add_argument("--skip-build", action="store_true", help="Reuse an existing TensorRT engine.")
    parser.add_argument("--benchmark-only", action="store_true", help="Skip export and build, benchmark the existing engine.")
    parser.add_argument("--num-frames", type=int, default=9, help="Temporal frames per clip. Default: 9")
    parser.add_argument("--buffer-max-len", type=int, default=72, help="Sliding buffer length. Default: 72")
    parser.add_argument("--img-height", type=int, default=240, help="Input height. Default: 240")
    parser.add_argument("--img-width", type=int, default=320, help="Input width. Default: 320")
    parser.add_argument("--det-token-num", type=int, default=20, help="Number of DET tokens. Default: 20")
    parser.add_argument("--duration", type=int, default=10, help="trtexec benchmark duration seconds. Default: 10")
    parser.add_argument("--iterations", type=int, default=50, help="trtexec benchmark minimum iterations. Default: 50")
    parser.add_argument("--warmup", type=int, default=200, help="trtexec warmup milliseconds. Default: 200")
    parser.add_argument("--workspace-mib", type=int, default=4096, help="TensorRT workspace in MiB. Default: 4096")
    parser.add_argument(
        "--calibration-videos",
        nargs="*",
        default=DEFAULT_CALIBRATION_SOURCES,
        help="Representative videos used for INT8 calibration.",
    )
    parser.add_argument("--max-calibration-clips", type=int, default=32, help="Maximum calibration clips. Default: 32")
    parser.add_argument(
        "--calibration-cache",
        help="Optional calibration cache path. Defaults under --output-dir.",
    )
    return parser.parse_args()


def load_model(weights_path, device):
    model = get_sia(size="b", pretrain=None, det_token_num=20, text_lora=True, num_frames=9)["sia"]
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def export_onnx(args, onnx_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)
    wrapper = VisionWrapper(model).to(device).eval()
    example = torch.randn(
        1,
        args.num_frames,
        3,
        args.img_height,
        args.img_width,
        device=device,
    )
    torch.onnx.export(
        wrapper,
        example,
        str(onnx_path),
        input_names=["video"],
        output_names=["pred_logits", "pred_boxes", "human_logits"],
        opset_version=17,
        do_constant_folding=False,
    )


def iter_calibration_clips(video_paths, num_frames, buffer_max_len, img_height, img_width, max_clips):
    normalizer = build_normalizer([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    clip_count = 0
    for video_path in video_paths:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            continue
        buffer = SlidingWindowBuffer(buffer_max_len, num_frames)
        try:
            while clip_count < max_clips:
                ret, frame = capture.read()
                if not ret:
                    break
                resized = resize_frame(frame, img_width, img_height)
                resized_chw = resized.transpose(2, 0, 1)
                buffer.push(resized_chw, resized_chw)
                if not buffer.ready():
                    continue
                sampled_clip = buffer.sampled_clip()
                clip_tensor = torch.from_numpy(sampled_clip).float() / 255.0
                clip_tensor = normalizer(clip_tensor).unsqueeze(0)
                yield np.ascontiguousarray(clip_tensor.numpy().astype(np.float32))
                clip_count += 1
                if clip_count >= max_clips:
                    break
        finally:
            capture.release()


class ClipEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, video_paths, num_frames, buffer_max_len, img_height, img_width, max_clips, cache_path):
        super().__init__()
        self.batch_size = 1
        self.cache_path = str(cache_path)
        self.clip_iter = iter_calibration_clips(
            video_paths,
            num_frames=num_frames,
            buffer_max_len=buffer_max_len,
            img_height=img_height,
            img_width=img_width,
            max_clips=max_clips,
        )
        self.device_tensor = torch.empty(
            (1, num_frames, 3, img_height, img_width),
            dtype=torch.float32,
            device="cuda",
        )

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        try:
            clip = next(self.clip_iter)
        except StopIteration:
            return None
        clip_tensor = torch.from_numpy(clip).to(device="cuda", dtype=torch.float32)
        self.device_tensor.copy_(clip_tensor)
        return [int(self.device_tensor.data_ptr())]

    def read_calibration_cache(self):
        if os.path.isfile(self.cache_path):
            with open(self.cache_path, "rb") as handle:
                return handle.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_path, "wb") as handle:
            handle.write(cache)


def build_engine(args, onnx_path, engine_path, calibration_cache_path=None):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as handle:
        if not parser.parse(handle.read()):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(f"TensorRT ONNX parse failed: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib * 1024 * 1024)

    if args.mode == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif args.mode == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        calibrator = ClipEntropyCalibrator(
            video_paths=[Path(path) for path in args.calibration_videos if Path(path).is_file()],
            num_frames=args.num_frames,
            buffer_max_len=args.buffer_max_len,
            img_height=args.img_height,
            img_width=args.img_width,
            max_clips=args.max_calibration_clips,
            cache_path=calibration_cache_path,
        )
        config.int8_calibrator = calibrator
    else:
        raise RuntimeError(f"Unsupported build mode '{args.mode}'.")

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build returned None.")
    with open(engine_path, "wb") as handle:
        handle.write(serialized_engine)


def parse_trtexec_summary(stdout_text):
    patterns = {
        "throughput_qps": r"Throughput:\s+([0-9.]+)\s+qps",
        "latency_mean_ms": r"Latency:.*mean = ([0-9.]+) ms",
        "gpu_compute_mean_ms": r"GPU Compute Time:.*mean = ([0-9.]+) ms",
        "enqueue_mean_ms": r"Enqueue Time:.*mean = ([0-9.]+) ms",
    }
    metrics = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout_text)
        metrics[key] = float(match.group(1)) if match else None
    return metrics


def benchmark_engine(engine_path, duration, iterations, warmup):
    command = [
        str(TRTEXEC_PATH),
        f"--loadEngine={engine_path}",
        f"--warmUp={warmup}",
        f"--duration={duration}",
        f"--iterations={iterations}",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "trtexec benchmark failed:\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    metrics = parse_trtexec_summary(completed.stdout)
    metrics["raw_stdout"] = completed.stdout
    return metrics


def benchmark_pytorch_fp16(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)
    wrapper = VisionWrapper(model).to(device).eval()
    example = torch.randn(1, args.num_frames, 3, args.img_height, args.img_width, device=device)
    warmup = 20
    iterations = 100
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for _ in range(warmup):
                wrapper(example)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                wrapper(example)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
    return {
        "iterations": iterations,
        "mean_ms": round(elapsed / iterations * 1000.0, 3),
        "qps": round(iterations / elapsed, 3),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    onnx_path = Path(args.onnx_path) if args.onnx_path else output_dir / "sia_vision_fixed.onnx"
    engine_suffix = f"sia_vision_{args.mode}.engine"
    engine_path = Path(args.engine_path) if args.engine_path else output_dir / engine_suffix
    calibration_cache_path = (
        Path(args.calibration_cache)
        if args.calibration_cache
        else output_dir / f"sia_vision_{args.mode}.calib"
    )

    if not args.benchmark_only and not args.skip_export:
        export_onnx(args, onnx_path)
    if not args.benchmark_only and not args.skip_build:
        build_engine(args, onnx_path, engine_path, calibration_cache_path)

    trt_metrics = benchmark_engine(engine_path, args.duration, args.iterations, args.warmup)
    summary = {
        "git_commit": infer_git_commit(),
        "mode": args.mode,
        "weights": args.weights,
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "calibration_cache_path": str(calibration_cache_path) if args.mode == "int8" else None,
        "input_shape": [1, args.num_frames, 3, args.img_height, args.img_width],
        "trt_metrics": trt_metrics,
    }
    if args.mode == "fp16":
        summary["pytorch_fp16_baseline"] = benchmark_pytorch_fp16(args)

    summary_path = output_dir / f"{args.mode}_benchmark_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
