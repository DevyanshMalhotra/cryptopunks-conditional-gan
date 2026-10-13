# Conditional CryptoPunks Generation

Experimental conditional image generation using **NVIDIA's StyleGAN3 codebase configured with the StyleGAN2 architecture** (`--cfg=stylegan2`). The project includes data preparation, multi-hot attribute labeling and checkpoint-based image generation.

> The public repository is a compact reproduction companion, **not** a self-contained pretrained model. Original datasets, trained weights, and NVIDIA's upstream training code are not redistributed.

## Overview

- Convert CryptoPunks JPEG source images into 32 × 32 PNGs with nearest-neighbor resizing.
- Create a stable attribute vocabulary from JSONL metadata and export multi-hot conditioning vectors in `dataset.json` and `attributes.csv`.
- Train a conditional generator using NVIDIA's original trainer with `--cond=1` and adaptive discriminator augmentation (ADA).
- Load a saved `G_ema` generator checkpoint and render attribute-conditioned images.

### Sample outputs

| | | | |
| --- | --- | --- | --- |
| ![Sample 1](samples/generated_cryptopunk.png) | ![Sample 2](samples/generated_cryptopunk_1.png) | ![Sample 3](samples/generated_cryptopunk_2.png) | ![Sample 4](samples/generated_cryptopunk_3.png) |

Sample PNGs were retained from the original local project. Their model checkpoints are not included.

## Structure

```text
scripts/prepare_dataset.py    Source images + JSONL -> training PNGs and labels
generate.py                   Conditional inference from a saved checkpoint
tests/test_prepare_dataset.py Preprocessing unit test
samples/                      Existing sample PNGs
requirements.txt              Pillow for local preprocessing
```

## Setup

Use Python 3 and a compatible PyTorch/CUDA environment for training or generation.

```bash
git clone https://github.com/DevyanshMalhotra/cryptopunks-conditional-gan.git
cd cryptopunks-conditional-gan
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

git clone https://github.com/NVlabs/stylegan3.git
```

For the upstream model implementation, install the additional dependencies following [NVlabs/stylegan3](https://github.com/NVlabs/stylegan3).

Expected input paths (source images and metadata are **not** bundled):

```text
data/cryptopunks/
  transactions.jsonl
  images/
    cryptopunk0.jpg
    cryptopunk1.jpg
    ...
```

JSONL example:

```json
{"punk_id": 0, "type": ["Female"], "accessories": ["Earring"]}
```

### 1. Prepare the dataset

```bash
python scripts/prepare_dataset.py \
  --metadata data/cryptopunks/transactions.jsonl \
  --images data/cryptopunks/images \
  --output data/cryptopunks/pngs \
  --attributes-csv attributes.csv
```

The preprocessing script matches images by `punk_id`, builds a vocabulary of attributes, writes an `attributes.csv` header, and creates a StyleGAN-compatible `dataset.json`. Preserve the CSV from the original training run to ensure correct conditioning at inference time.

### 2. Package the image data

```bash
python stylegan3/dataset_tool.py \
  --source data/cryptopunks/pngs \
  --dest datasets/cryptopunks-32x32.zip
```

### 3. Train

An example training invocation matching the project's approach:

```bash
python stylegan3/train.py \
  --outdir=training-runs/cryptopunks \
  --cfg=stylegan2 \
  --data=datasets/cryptopunks-32x32.zip \
  --gpus=1 --batch=16 --cond=1 --gamma=8 \
  --mirror=1 --aug=ada --metrics=fid50k \
  --kimg=500 --cbase=16384 --mbstd-group=2
```

Adjust compute and batch settings for the available GPU and NVIDIA's supported versions. **Important:** `--cfg=stylegan2` selects StyleGAN2 architecture even though the trainer lives in the StyleGAN3 repository.

### 4. Generate

```bash
python generate.py \
  --upstream stylegan3 \
  --snapshot training-runs/cryptopunks/<run-directory>/network-snapshot-000500.pkl \
  --attributes-csv attributes.csv \
  --attribute Male --attribute Earring \
  --output samples/new-punk.png --seed 42
```

The snapshot and matching attribute CSV are required. For unconditioned selection, omit `--attribute`; the output uses a zero vector.

## Verification

The data preparation test does not require a CUDA GPU or pretrained checkpoint:

```bash
python -m unittest discover -s tests -v
```

Preprocessing tests pass on the packaged sample test inputs. Full GAN training, checkpoint inference and published research metrics have **not** been independently reproduced from this repository because the original training dataset and checkpoint are absent. The old experimental ONNX exporter was excluded rather than shipping a nonfunctional export path.

## Attribution and scope

Upstream architecture, trainer and dataset tools are provided by [NVIDIA StyleGAN3](https://github.com/NVlabs/stylegan3), subject to NVIDIA's original license. This repository contains project-specific preprocessing and inference code. CryptoPunks artwork and metadata remain subject to their respective rights; no full training dataset is redistributed here.
