# FastMRI: Attentive MRI Reconstruction

Deep learning-based MRI reconstruction on undersampled k-space data, using a U-Net model trained with SSIM and MSE loss functions on the NYU fastMRI single-coil knee dataset. Full project report: [fastMRI_Attentive_Reconstruction.pdf](./fastMRI_Attentive_Reconstruction.pdf)

---

## Overview

MRI acquisition requires long scan times (15–90 min) due to Nyquist-Shannon sampling requirements in k-space. This project accelerates MRI by training a U-Net to reconstruct high-quality images from **undersampled k-space data**, reducing acquisition time while preserving diagnostic image quality.

The baseline approach of applying inverse FFT directly to undersampled k-space (zero-filled IFFT) produces aliased, blurry images. Our trained U-Net produces significantly cleaner reconstructions with sharper tissue boundaries.

---

## Tech Stack

**Framework:** PyTorch

**Model:** U-Net (encoder-decoder with skip connections, 3×3 conv, ReLU, max-pooling, upsampling)

**Loss Functions:** SSIM (Structural Similarity Index), MSE (Mean Squared Error), combined weighted loss (α×MSE + β×SSIM)

**Optimizer:** Adam

**Data:** NYU fastMRI single-coil knee dataset (HDF5 `.h5` format), k-space + ground truth image pairs

**Signal Processing:** Inverse FFT (IFFT) for k-space → image domain reconstruction, k-space masking for undersampling simulation

**Metrics:** SSIM, MSE, visual comparison against zero-filled IFFT baseline and fully-sampled ground truth

---

## Approach

### K-Space and Image Reconstruction

MRI data is acquired in **k-space** (Fourier domain). The vertical axis encodes phase variations and the horizontal axis encodes frequency. Key properties:
- The **center** of k-space carries contrast information; the **periphery** carries edge/margin detail
- Undersampling k-space (via acceleration masks) reduces scan time but introduces aliasing artifacts
- **Inverse FFT** reconstructs the spatial image from k-space; zero-filled IFFT on undersampled data gives a blurry baseline

We apply a mask to simulate undersampling at a given **acceleration rate** (e.g., 8×), then train a U-Net to recover the fully-sampled image.

### U-Net Architecture

Encoder-decoder architecture with skip connections:
- **Contracting path:** repeated (3×3 conv → ReLU → 3×3 conv → ReLU → 2×2 max pool) blocks, doubling channels at each depth
- **Expansion path:** upsampling → concatenation with skip connection from contracting path → 3×3 conv → ReLU blocks
- Skip connections fuse low-level spatial detail with high-level semantic features, critical for image refinement tasks

### Loss Functions

| Loss | Formula | Behavior |
|------|---------|----------|
| **MSE** | (1/n)Σ(yᵢ − ŷᵢ)² | Pixel-level fidelity; can over-smooth |
| **SSIM** | Luminance × Contrast × Structure | Perceptual quality; captures local structural changes |
| **Combined** | α×MSE + β×SSIM | Balances pixel fidelity and perceptual quality |

SSIM is computed with a sliding window kernel; window size and loss weights (α, β) are tunable hyperparameters.

---

## Data Pipeline

1. **Ingestion** — Retrieve HDF5 (`.h5`) files from dataset directory; configurable max file limit
2. **Slice extraction** — Extract selected k-space and target image slices (configurable slice indices)
3. **IFFT reconstruction** — Apply inverse FFT to k-space slices to get spatial-domain images
4. **Normalization** — Mean normalization + clipping of extreme outliers (min-max normalization discarded due to outlier skew)
5. **Cropping & resizing** — Crop dark border regions to focus training on diagnostically relevant areas
6. **DataLoader** — Batch data for GPU training with configurable batch size

---

## Hyperparameters

| Parameter | Description | Values Tested |
|---|---|---|
| `ACCELERATE_RATE` | k-space undersampling factor | 8 |
| `DROPOUT_PROB` | Dropout probability in U-Net | 0.0, 0.35 |
| `WINDOW_SIZE` | SSIM loss kernel size | 7, 11 |
| `LR` | Adam learning rate | 0.0015 |
| `SLICES` | MRI slice indices used for training | [16, 17, 18, 19, 20] |
| `BATCH_SIZE` | Training batch size | 4 |
| `EPOCHS` | Training epochs | 10–100 |

---

## Results

### SSIM Loss Model
- **Window size 11 > window size 7** — larger kernel provides more stable SSIM estimates, reducing loss more efficiently
- **Dropout 0.35 > Dropout 0.0** — no dropout showed overfitting (training loss decreasing, validation flat); dropout 0.35 reduced the train/val gap. Optimal dropout likely in 0.25–0.3 range.
- Reconstruction showed clearer, smoother tissue boundaries vs zero-filled IFFT baseline; fine soft-tissue detail still limited by small training slice subset

### Combined MSE + SSIM Model
- Mean normalization + clipping (replacing min-max normalization) was key to stable training — extreme outliers in k-space data caused distribution skew
- Combining MSE and SSIM leverages pixel-level fidelity (MSE) and perceptual quality (SSIM); weighted loss balances trade-off between over-smoothing (pure MSE) and structural accuracy
- Output reconstruction showed improved detail and contrast compared to SSIM-only model

---

## Repository Structure

```
FastMRI-Reconstruction/
├── Fastmri_unet_dev/            # Main implementation
│   ├── Model/
│   │   ├── unet.py              # U-Net architecture
│   │   └── losses.py            # SSIM, MSE, combined loss functions
│   ├── Training/
│   │   └── unet_train.py        # Training loop, validation, checkpointing
│   ├── Data/                    # Data loading and preprocessing
│   ├── config_file.py           # Hyperparameter configuration
│   ├── main.py                  # Entry point
│   └── requirements.txt
├── fastmri/                     # fastMRI library (Meta/NYU)
├── fastmri_examples/            # Example scripts (U-Net, VarNet, CS, zero-filled)
├── data/                        # Dataset samples (full dataset downloaded separately)
├── fastMRI_Attentive_Reconstruction.pdf   # Full project report
├── fastMRI_tutorial.ipynb       # Dataset exploration tutorial
└── example.ipynb
```

---

## Setup

### 1. Install the fastMRI library

```bash
# Clone and install
pip install -e .
# Or follow the original fastMRI repo setup:
# https://github.com/facebookresearch/fastMRI
```

### 2. Download the dataset

The `data/` directory contains only sample files. Download the full NYU fastMRI single-coil knee dataset:

[https://fastmri.med.nyu.edu/](https://fastmri.med.nyu.edu/)

Update paths in `Fastmri_unet_dev/config_file.py` after downloading.

### 3. Explore the dataset

```bash
jupyter notebook fastMRI_tutorial.ipynb
```

### 4. Train the model

```bash
cd Fastmri_unet_dev
python main.py
```

Key config options in `config_file.py`: `IS_SSIM`, `DROPOUT_PROB`, `WINDOW_SIZE`, `ACCELERATE_RATE`, `EPOCHS`, `LR`.

---

## Limitations & Future Work

- Training limited to a small subset of slices due to GPU constraints — training on the full dataset would improve feature extraction
- Single-coil only — multi-coil reconstruction (exploiting coil sensitivity maps via SENSE/GRAPPA) would improve quality
- Loss function exploration — learned perceptual losses (e.g., LPIPS) or adversarial losses (GAN-based) could better capture semantic image quality
- Data augmentation not fully explored
- Alternative architectures: VarNet (variational network) and ADMM-Net are available in `fastmri_examples/` for comparison
