# EdgeVLM setup with `uv` on Jetson AGX Orin

This guide is for setting up this repository on a fresh Jetson OS install.

It is based on the working environment already present in this repo:

- JetPack / L4T: `R36.4.7`
- Python: `3.10.12`
- Virtual environment tool: `uv`
- Working package set includes `torch 2.8.0`, `torchvision 0.23.0`, and `opencv-contrib-python 4.11.0.86`

## 1. Confirm the base system

Run:

```bash
cat /etc/nv_tegra_release
python3 -V
```

Expected baseline:

- JetPack 6.x / L4T `R36.4.x`
- Python `3.10.x`

## 2. Install system packages

Start from a fresh terminal:

```bash
sudo apt update
sudo apt install -y \
  git \
  curl \
  build-essential \
  cmake \
  pkg-config \
  python3-dev \
  python3-pip \
  python3-venv \
  libopenblas-dev \
  libjpeg-dev \
  zlib1g-dev \
  libavformat-dev \
  libavcodec-dev \
  libavutil-dev \
  libswscale-dev \
  libgtk2.0-dev \
  libgtk-3-dev
```

These cover the native build dependencies most likely to matter for this repo on Jetson.

## 3. Install `uv`

If `uv` is not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the shell, or load the updated shell config, then confirm:

```bash
uv --version
```

## 4. Clone the repository

```bash
git clone <YOUR_REPO_URL> EdgeVLM
cd EdgeVLM
```

If you already have the repo, just `cd` into it.

## 5. Create the virtual environment with `uv`

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
```

This matches the repo-local `.venv` pattern already in use here.

## 6. Install the project requirements

Install the declared Python dependencies:

```bash
uv pip install -r requirements.txt
```

Then install the extra package this repo also needs:

```bash
uv pip install PyYAML
```

Why `PyYAML`: `utils/config.py` imports `yaml`, but `PyYAML` is not listed in `requirements.txt`.

## 7. Verify the environment

Run these checks from the activated `.venv`:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -c "import cv2; print(cv2.__version__)"
python -c "from sia import get_sia; from datasets import avatextaug; print('imports ok', len(avatextaug))"
```

On the working environment in this repo, those checks resolve to:

- `torch 2.8.0`
- CUDA `12.6`
- `torch.cuda.is_available() == True`
- `cv2 4.11.0`

If those imports succeed, the environment is in good shape for the demo.

## 8. Add the model weights

Create the weights directory if needed:

```bash
mkdir -p weights
```

Place the downloaded checkpoint files under `weights/`.

The demo scripts in this repo currently expect paths like:

- `weights/avak_b16_11.pt`

If your file has a different name, pass it explicitly where the script supports `-weights`.

## 9. Run the demo

Offline video demo:

```bash
source .venv/bin/activate
python demo.py -F video.mp4
```

Webcam / OpenCV camera demo:

```bash
source .venv/bin/activate
python opencv_demo_frame.py -weights weights/avak_b16_11.pt
```

## 10. Re-enter the environment later

From the repo root:

```bash
source .venv/bin/activate
```

## 11. Troubleshooting

If `uv` is not found after install:

```bash
echo $PATH
which uv
```

If `torch.cuda.is_available()` is `False`, the Jetson CUDA stack is not being seen correctly by the installed PyTorch build.

If `cv2` fails to import, reinstall the Python packages inside the venv:

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install PyYAML
```

If you want exact package parity with the working environment on this machine, compare against:

- `torch==2.8.0`
- `torchvision==0.23.0`
- `opencv-contrib-python==4.11.0.86`
- `decord==0.6.0`
- `timm==1.0.25`
- `transformers==5.3.0`

## 12. Notes

This guide reflects the environment that is already working in this repository today. It is not a generic Jetson compatibility guide; it is the shortest repeatable path to recreating the setup that currently imports and runs here.
