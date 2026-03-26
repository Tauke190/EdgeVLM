import csv
import re
import shutil
import subprocess
import threading
import time


TEGRSTATS_RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")
TEGRSTATS_GPU_RE = re.compile(r"GR3D_FREQ (\d+)%")


class SystemMonitor:
    def __init__(self, output_csv_path, sample_interval_s=1.0):
        self.output_csv_path = output_csv_path
        self.sample_interval_s = sample_interval_s
        self.rows = []
        self._stop_event = threading.Event()
        self._thread = None
        self._tegrastats_process = None
        self._source = self._detect_source()
        self._start_time = None

    @property
    def source(self):
        return self._source

    def _detect_source(self):
        if shutil.which("tegrastats"):
            return "tegrastats"
        if shutil.which("nvidia-smi"):
            return "nvidia-smi"
        return "none"

    def start(self):
        self._start_time = time.perf_counter()
        if self._source == "tegrastats":
            self._thread = threading.Thread(target=self._run_tegrastats, daemon=True)
            self._thread.start()
        elif self._source == "nvidia-smi":
            self._thread = threading.Thread(target=self._run_nvidia_smi, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._tegrastats_process is not None:
            self._tegrastats_process.terminate()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._write_rows()

    def _elapsed_s(self):
        return round(time.perf_counter() - self._start_time, 3)

    def _append_row(self, row):
        self.rows.append(
            {
                "timestamp_s": self._elapsed_s(),
                "source": self._source,
                "gpu_util_percent": row.get("gpu_util_percent"),
                "memory_used_mb": row.get("memory_used_mb"),
                "memory_total_mb": row.get("memory_total_mb"),
                "temperature_c": row.get("temperature_c"),
                "raw": row.get("raw"),
            }
        )

    def _run_tegrastats(self):
        self._tegrastats_process = subprocess.Popen(
            ["tegrastats", "--interval", str(int(self.sample_interval_s * 1000))],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        assert self._tegrastats_process.stdout is not None
        for line in self._tegrastats_process.stdout:
            if self._stop_event.is_set():
                break
            line = line.strip()
            if not line:
                continue
            row = {"raw": line}
            ram_match = TEGRSTATS_RAM_RE.search(line)
            gpu_match = TEGRSTATS_GPU_RE.search(line)
            if ram_match:
                row["memory_used_mb"] = int(ram_match.group(1))
                row["memory_total_mb"] = int(ram_match.group(2))
            if gpu_match:
                row["gpu_util_percent"] = int(gpu_match.group(1))
            self._append_row(row)

    def _run_nvidia_smi(self):
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                first_line = result.stdout.strip().splitlines()[0]
                gpu_util, mem_used, mem_total, temperature = [part.strip() for part in first_line.split(",")]
                self._append_row(
                    {
                        "gpu_util_percent": int(gpu_util),
                        "memory_used_mb": int(mem_used),
                        "memory_total_mb": int(mem_total),
                        "temperature_c": int(temperature),
                        "raw": first_line,
                    }
                )
            except Exception as exc:
                self._append_row({"raw": f"monitor_error={exc}"})
            self._stop_event.wait(self.sample_interval_s)

    def _write_rows(self):
        with open(self.output_csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "timestamp_s",
                    "source",
                    "gpu_util_percent",
                    "memory_used_mb",
                    "memory_total_mb",
                    "temperature_c",
                    "raw",
                ],
            )
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)
