# PRNU-Based Camera Identification Under Social Media Compression

**Full documentation (datasets, algorithms, architecture, execution):** [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)

## 1. Project overview

This repository implements a complete machine learning research pipeline for **device-level camera identification** using photo-response non-uniformity (PRNU) noise residuals, with emphasis on **robustness to JPEG compression and resizing** (including simulated WhatsApp and Flickr pipelines). It compares three approaches: **normalized cross-correlation (NCC)** against estimated sensor fingerprints, a **1-channel ResNet-18 CNN** on residual patches, and a **Siamese encoder** with centroid-based inference. Experiments are organized into groups A–D (core performance, ablations, robustness/calibration, and forensic analysis) and write structured JSON under `results/`.

## 2. Dataset setup (Kaggle CLI)

Install the [Kaggle API](https://github.com/Kaggle/kaggle-api) and place your `kaggle.json` credentials in `~/.kaggle/`.

**Dresden Image Database (primary):**

```bash
kaggle datasets download -d micscodes/dresden-image-database -p data/raw/dresden --unzip
```

**IEEE SP Camera Model Identification (competition data):**

```bash
kaggle competitions download -c sp-society-camera-model-identification -p data/raw/ieee_sp
unzip -q data/raw/ieee_sp/sp-society-camera-model-identification.zip -d data/raw/ieee_sp
```

Alternatively, run the bundled script from the project root:

```bash
chmod +x scripts/download_data.sh
./scripts/download_data.sh
```

**Social media simulation** (WhatsApp / Flickr) is implemented in code (`src/compression.py`); no extra download is required.

## 3. Environment setup

Using conda (recommended):

```bash
conda env create -f environment.yml
conda activate prnu_project
```

Using pip:

```bash
pip install -r requirements.txt
```

## 4. How to run all experiments (order)

From the `prnu_project` directory (so `configs/` and `src/` resolve correctly):

```bash
# Group A — core performance (clean, WhatsApp+Flickr test, JPEG-aug train)
python experiments/run_group_A.py --config configs/default.yaml

# Group B — ablations (Wiener window, CNN data fraction)
python experiments/run_group_B.py --config configs/default.yaml

# Group C — robustness sweeps + calibration (+ optional IEEE SP sweep)
python experiments/run_group_C.py --config configs/default.yaml

# Group D — per-device robustness, confusion / error taxonomy, CPU profiling
python experiments/run_group_D.py --config configs/default.yaml
```

Optional flags (all groups): `--max-devices N` to cap the number of devices for faster debugging, `--device cuda` when a GPU is available. Group C additionally supports `--skip-ieee` if IEEE SP data is not unpacked yet, and `--cnn-epochs` / `--siamese-epochs` overrides.

### Fast GPU workflow (Colab/Kaggle)

1. **Precompute residual cache once** (resume-safe):

```bash
python scripts/precompute_prnu.py \
  --config configs/colab_gpu_fast.yaml \
  --split train \
  --out-dir data/processed/residuals
```

2. **Train Group A with cached residuals + AMP**:

```bash
python experiments/run_group_A.py \
  --config configs/colab_gpu_fast.yaml \
  --device cuda \
  --num-workers 4 \
  --prefetch-factor 2 \
  --residual-root data/processed/residuals
```

3. **Notebook-friendly launcher** (auto-detects GPU):

```bash
python scripts/run_notebook_training.py \
  --config configs/colab_gpu_fast.yaml \
  --sample-fraction 0.20 \
  --max-devices 10 \
  --residual-root data/processed/residuals
```

## 5. Results schema

- **`results/group_A.json`**: `A1` clean test; `A2` train clean / test WhatsApp+Flickr (metrics averaged across the two simulators); `A3` train with JPEG augmentation / test social simulators. Each block reports `top1`, `top5`, `macro_f1` for `NCC`, `CNN`, and `Siamese`.
- **`results/group_B.json`**: `B1` Wiener window ablation for NCC; `B2` CNN validation accuracy vs. subsampled training images.
- **`results/group_C.json`**: `C1.jpeg_sweep` Dresden accuracies per JPEG `Q`, plus `AURC` and `RAD`; `C1.ieee_sp_crossdataset` mirrors the sweep on IEEE SP when data is present; `C2.resize_sweep` for resize stress; `C3` calibration (ECE, Brier, reliability bins, selective risk).
- **`results/group_D.json`**: `D1` per-device WhatsApp top-1 and fragile-device list; `D2` confusion matrix and same-family vs cross-family error rates; `D3` CPU timing and peak memory for NCC/CNN/Siamese at patch sizes 64², 128², 256².

Each JSON file includes the resolved `config` and `git_hash` for traceability.

## 6. Reproducibility

- **Randomness**: All stochastic operations are driven by `seed` in `configs/default.yaml` (Python, NumPy, Torch).
- **Hyperparameters**: Single source of truth is `configs/default.yaml` (patch size, Wiener window, training epochs, compression parameters, calibration bins).
- **Data splits**: Dresden uses **scene-aware** splits (60/20/20 by default) so images from the same scene do not cross train/val/test. Patches are indexed so **no patch from a test parent image appears in training** when splits are built at the image level.
- **Constraints**: No ImageNet pretraining, no EXIF features, no color augmentation; JPEG training augmentation in Group A3 uses **per-image** random quality via a per-path transform factory.
