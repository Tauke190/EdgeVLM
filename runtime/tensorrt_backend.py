from pathlib import Path
import sys

import torch


SYSTEM_DIST_PACKAGES = "/usr/lib/python3.10/dist-packages"
if SYSTEM_DIST_PACKAGES not in sys.path:
    sys.path.append(SYSTEM_DIST_PACKAGES)

import tensorrt as trt


TRT_TO_TORCH_DTYPE = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT8: torch.int8,
    trt.DataType.BOOL: torch.bool,
}


class TensorRTVisionBackend:
    def __init__(self, engine_path, device):
        if not str(device).startswith("cuda"):
            raise RuntimeError("TensorRT runtime backend requires a CUDA device.")

        self.engine_path = Path(engine_path)
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        self.device = device
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        with open(self.engine_path, "rb") as handle:
            self.engine = self.runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context: {self.engine_path}")

        self.input_name = None
        self.output_names = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)

        if self.input_name is None:
            raise RuntimeError("TensorRT engine does not expose an input tensor.")

        self.input_shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(self.input_name))
        self.output_buffers = {}
        for name in self.output_names:
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
            dtype = TRT_TO_TORCH_DTYPE.get(self.engine.get_tensor_dtype(name))
            if dtype is None:
                raise RuntimeError(
                    f"Unsupported TensorRT output dtype for '{name}': {self.engine.get_tensor_dtype(name)}"
                )
            self.output_buffers[name] = torch.empty(shape, dtype=dtype, device=self.device)
            self.context.set_tensor_address(name, int(self.output_buffers[name].data_ptr()))

    def infer(self, clip_tensor):
        actual_shape = tuple(int(dim) for dim in clip_tensor.shape)
        if actual_shape != self.input_shape:
            raise RuntimeError(
                f"TensorRT engine expected input shape {self.input_shape}, got {actual_shape}"
            )

        if clip_tensor.device.type != "cuda":
            clip_tensor = clip_tensor.to(device=self.device)
        if clip_tensor.dtype != torch.float32:
            clip_tensor = clip_tensor.float()
        clip_tensor = clip_tensor.contiguous()

        self.context.set_tensor_address(self.input_name, int(clip_tensor.data_ptr()))
        stream = torch.cuda.current_stream(device=clip_tensor.device).cuda_stream
        ok = self.context.execute_async_v3(stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned failure.")

        return {name: tensor for name, tensor in self.output_buffers.items()}
