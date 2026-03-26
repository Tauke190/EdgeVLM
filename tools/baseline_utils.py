import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path


COLORS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_actions(actions_json_path, actions_override=None):
    if actions_override:
        return [action.replace("_", " ") for action in actions_override]

    data = load_json(actions_json_path)
    return list(data.keys())


def timestamp_slug():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def make_run_dir(output_root, run_name):
    run_dir = Path(output_root) / f"{timestamp_slug()}_{run_name}"
    ensure_dir(run_dir)
    return run_dir


def resolve_color(name):
    if name not in COLORS:
        raise ValueError(f"Unsupported color '{name}'. Expected one of: {sorted(COLORS)}")
    return COLORS[name]


def summarize_series(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "max_ms": None,
        }

    ordered = sorted(clean)
    count = len(ordered)
    p95_index = min(count - 1, max(0, int(round(0.95 * (count - 1)))))
    return {
        "count": count,
        "mean_ms": round(sum(ordered) / count * 1000.0, 3),
        "median_ms": round(ordered[count // 2] * 1000.0, 3),
        "p95_ms": round(ordered[p95_index] * 1000.0, 3),
        "max_ms": round(max(ordered) * 1000.0, 3),
    }


def infer_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def write_run_summary(path, summary_lines):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")


def to_builtin(value):
    if isinstance(value, dict):
        return {key: to_builtin(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [to_builtin(inner) for inner in value]
    if hasattr(value, "item"):
        return value.item()
    return value
