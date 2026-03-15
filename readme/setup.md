````markdown

## 1. Model Selection

All code lives in the `models/` folder. The relevant factory function:

```python
def build_vision_encoder(self, lora: bool = False):
    """Build vision encoder.
    Returns:
        (vision_encoder, vision_layernorm), each an nn.Module.
    """
    encoder_name = self.vision_encoder_name

    if encoder_name == "vit_l14":
        vision_encoder = clip_joint_l14()
    elif encoder_name == "vit_b16":
        vision_encoder = clip_joint_b16()
    else:
        raise NotImplementedError(f"Not implemented: {encoder_name}")

    return vision_encoder
````

---


## 2. Datasets and Annotations

LitSiA follows the **SiA** data format and protocol.

* **AVA / AVA-Kinetics**
  Frame-level action detection, same setup as SiA:

  * Person boxes and multi-label actions at key frames
  * AVA-style annotation files (`.csv` / `.pkl` as in SiA)

* Optional additional benchmarks:

  * **UCF101-24**
  * **JHMDB21 / HMDB51**

> Make sure paths and annotation formats match what SiA expects (e.g., AVA splits, key-frame indices, etc.).

---

## 3. Training and Evaluation

### 3.1 Backbone and Losses


* **Text encoder:** Frozen CLIP-style encoder (as in SiA), with GPT-augmented descriptors.
* **Head and loss:**

  * SiA-style DET tokens and detection head
  * Hungarian set-prediction loss (boxes + human / non-human + actions)

---

### 3.2 Example Slurm Training Script

` run.slurm`:

```bash
#!/bin/bash
#SBATCH --output=logs/run-%j.out
#SBATCH --nodes=1
#SBATCH --gres gpu:1 -C gmem80
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8G

python -u train_avak.py \
    -JSON avak_aws_stats_flt_b16_txtaug_txtlora.json \
    -AWS  avak_stats_flt_b16_txtaug_txtlora_assigned_ws.json \
    -SIZE t \                    # 't' = MobileViCLIP tiny
    -HEIGHT 240 \
    -WIDTH 320 \
    -BS 12 \
    -WORKERS 16 \
    -EPOCH 6 \
    -TRAIN AVA \
    -AVA sia/dummy/videos_15min \
    -KINETICS /datasets/kinetics-k700-2020 \
    -RATEKINETICS 8 \
    -RATEAVA 8 \
    -DET 20 \
    -FRAMES 8 \
    --TXTAUG \
    --TXTLORA \
    --SAVE
```

Notes:

* Replace dataset paths (`/datasets/...`, `/home/...`) with your own.
* `-DET 20` sets the number of DET queries.
* `--TXTAUG` and `--TXTLORA` enable GPT text augmentation and LoRA text tuning as in SiA.

---

### 3.3 Validation / Zero-Shot Evaluation

`val.sh`:

```bash
python -u val.py \
    -JSON avak_aws_stats_flt_b16_txtaug_txtlora.json \
    -PRETRAINED weights/avak_aws_stats_flt_b16_txtaug_txtlora \
    -SIZE t \
    -VAL AVA \
    -HEIGHT 240 \
    -WIDTH 320 \
    -BS 12 \
    -WORKERS 8 \
    -AVA sia/dummy/videos_15min \
    -RATEAVA 8 \
    -KINETICS /home/c3-0/datasets/Kinetics700 \
    -RATEKINETICS 8 \
    -UCF24 /home/c3-0/datasets/UCF101/videos \
    -RATEUCF24 7 \
    -ANNOUCF24 anno/UCF101v2-GT.pkl \
    -HMDB21 /home/c3-0/datasets/hmdb51/videos \
    -ANNOHMDB21 anno/JHMDB-GT.pkl \
    -DET 20 \
    -FRAMES 8 \
    --TXTAUG \
    --TXTLORA
```

Again, adapt all dataset and annotation paths for your environment.

---

## 4. Metrics

We follow **SiA / AVA** and report:

* **f-mAP@0.5** (often written as `f@0.5`):
  frame-level mean Average Precision at IoU 0.5 across all action classes.

```
```
