import numpy as np


class SlidingWindowBuffer:
    def __init__(self, buffer_max_len, num_frames):
        self.buffer_max_len = int(buffer_max_len)
        self.num_frames = int(num_frames)
        self.sample_stride = max(1, self.buffer_max_len // self.num_frames)
        self.sample_indices = np.arange(0, self.buffer_max_len, self.sample_stride)[: self.num_frames]
        self.frame_buffer = []
        self.plot_buffer = []

    @property
    def mid_index(self):
        return self.buffer_max_len // 2

    def push(self, resized_chw, original_chw):
        self.frame_buffer.append(resized_chw)
        self.plot_buffer.append(original_chw)
        if len(self.frame_buffer) > self.buffer_max_len:
            self.frame_buffer.pop(0)
            self.plot_buffer.pop(0)

    def ready(self):
        return len(self.frame_buffer) >= self.buffer_max_len

    def sampled_clip(self):
        if not self.ready():
            return None
        return np.array(self.frame_buffer)[self.sample_indices]

    def render_frame(self):
        if not self.ready():
            return None
        return self.plot_buffer[self.mid_index].transpose(1, 2, 0).astype(np.uint8)
