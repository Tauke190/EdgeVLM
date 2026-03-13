# Remote demo guide

This guide is for a user who is already on the VPN and already has SSH access to the Jetson.

The goal is:

1. SSH into the Jetson
2. Set up this repository
3. Create the `uv` environment
4. Start Sunshine on the virtual display
5. Run the demo on that virtual display
6. Connect from Moonlight and watch the output

This guide assumes the Jetson already has the one-time machine setup completed:

- Jetson OS is installed and booting normally
- `uv` is installed and available in `PATH`
- Sunshine is installed at `/usr/bin/sunshine-v0.23.1`
- `Xvfb`, `openbox`, and `xclock` are installed

If the machine is not at that point yet, use [UV_SETUP_JETSON.md](/home/anodyine/repos/EdgeVLM/UV_SETUP_JETSON.md) first for the repo and Python environment, then come back to this guide.

## 1. SSH into the Jetson

From your laptop or workstation:

```bash
ssh <jetson-user>@<jetson-host-or-vpn-ip>
```

Example:

```bash
ssh anodyine@10.0.0.25
```

## 2. Clone the repository

If the repo is not already on the machine for your user:

```bash
git clone <YOUR_REPO_URL> ~/repos/EdgeVLM
cd ~/repos/EdgeVLM
```

If the repo already exists:

```bash
cd ~/repos/EdgeVLM
git pull
```

## 3. Create the `uv` environment

From the repo root:

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
uv pip install -r requirements.txt
uv pip install PyYAML
```

Why `PyYAML`: the repo imports `yaml`, but `PyYAML` is not currently listed in `requirements.txt`.

## 4. Verify the Python environment

Run:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import cv2; print(cv2.__version__)"
python -c "from sia import get_sia; from datasets import avatextaug; print('imports ok', len(avatextaug))"
```

If those imports succeed, the environment is ready.

## 5. Make sure the weights are present

This repo expects checkpoint files under `weights/`.

Check:

```bash
ls -lah weights
```

The common demo checkpoint path in this repo is:

```bash
weights/avak_b16_11.pt
```

If the weights are stored elsewhere, copy them into `weights/` or pass the correct path with `-weights`.

## 6. Start Sunshine and the virtual display

From the repo root:

```bash
chmod +x start-sunshine.sh
./start-sunshine.sh
```

What this script does:

- applies some network tuning
- starts `Xvfb` on display `:99`
- starts `openbox` on `:99`
- starts `xclock` to keep the display active
- launches Sunshine bound to `DISPLAY=:99`

At the end, the important detail is:

```bash
DISPLAY=:99
```

Anything graphical you want to show through Moonlight must run on that display.

## 7. Run the demo on the virtual display

Open a second SSH session to the Jetson, or use `tmux` if you prefer.

Then:

```bash
ssh <jetson-user>@<jetson-host-or-vpn-ip>
cd ~/repos/EdgeVLM
source .venv/bin/activate
```

For the offline video demo:

```bash
DISPLAY=:99 python demo.py -F video.mp4
```

For the webcam / USB camera demo:

```bash
DISPLAY=:99 python opencv_demo_frame.py -weights weights/avak_b16_11.pt
```

Important: the demo waits for a keypress at startup because it opens a window and prints `Press any key to start buffering`.

That means you usually want Moonlight connected before expecting the demo to proceed visually.

## 8. Connect with Moonlight

On the client machine:

1. Open Moonlight
2. Add the Jetson by hostname or VPN IP
3. Start the connection

If this is the first connection, Sunshine and Moonlight may require pairing.

Common pairing flow:

1. Moonlight shows a PIN
2. You approve that PIN on the Sunshine side

Depending on your Sunshine setup, approval is usually done through the Sunshine web UI on the Jetson:

```text
https://<jetson-host-or-vpn-ip>:47990
```

If the pair is already established, Moonlight should connect directly.

## 9. Watch and interact with the demo

Once Moonlight is connected, you should see the virtual desktop from `DISPLAY=:99`.

If the demo window is open and waiting, send a keypress through Moonlight to let buffering start.

During the demo:

- `q` exits the OpenCV window
- the demo window should remain visible through Moonlight because both Sunshine and the demo are running on `:99`

## 10. Recommended terminal layout

This works well:

- SSH session 1: `./start-sunshine.sh`
- SSH session 2: `DISPLAY=:99 python ...`
- Moonlight: view the display output

That separation makes it easier to restart the demo without restarting Sunshine every time.

## 11. Troubleshooting

If Moonlight connects but the screen is blank:

```bash
ps -ef | grep -E 'Xvfb|openbox|sunshine'
```

Make sure all three are running.

If the demo does not appear in Moonlight, make sure it was launched with:

```bash
DISPLAY=:99
```

If the demo exits immediately, check the terminal output in the SSH session. The most common causes are:

- missing weights
- wrong weight path
- camera not available for `opencv_demo_frame.py`

If Sunshine is running but pairing fails, check whether the Sunshine web UI is reachable:

```text
https://<jetson-host-or-vpn-ip>:47990
```

If `start-sunshine.sh` fails on `sudo`, the user needs sudo access on the Jetson because the script currently changes sysctl values, adjusts MTU, kills old processes, and launches Sunshine with `sudo`.

## 12. Minimal repeat workflow

After the repo and `.venv` already exist, the usual remote workflow is just:

```bash
ssh <jetson-user>@<jetson-host-or-vpn-ip>
cd ~/repos/EdgeVLM
./start-sunshine.sh
```

Then in another SSH session:

```bash
ssh <jetson-user>@<jetson-host-or-vpn-ip>
cd ~/repos/EdgeVLM
source .venv/bin/activate
DISPLAY=:99 python demo.py -F video.mp4
```

Then connect from Moonlight to the Jetson and watch the output there.
