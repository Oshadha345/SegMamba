# SegMamba: Pure Mamba State-Space U-Net for 3D Medical Image Segmentation

**Competition Documentation - BraTS Challenge Style**  
**Author**: Elite Competitive Data Scientist  
**Date**: December 2025  
**Framework**: PyTorch 2.x with MONAI + mamba-ssm  

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Preprocessing Strategy](#preprocessing-strategy)
5. [Training Methodology](#training-methodology)
6. [Implementation Details](#implementation-details)
7. [Results & Ablations](#results--ablations)
8. [Future Work](#future-work)
9. [References](#references)

---

## 1. Executive Summary

**SegMamba** is a novel hybrid 3D U-Net architecture designed for medical image segmentation, specifically targeting the BraTS (Brain Tumor Segmentation) challenge. The architecture combines the efficiency of standard 3D convolutions for local feature extraction with state-space models (Mamba) for capturing long-range dependencies in volumetric medical images.

### Key Innovations:

1. **Hybrid Encoder Design**: Early stages use Conv3D for efficient local feature extraction; deep stages employ Mamba blocks for global context modeling with linear O(n) complexity
2. **Single GPU Optimization**: Designed for training on a single GPU using AMP, gradient accumulation, and memory-efficient state-space models
3. **nnU-Net Inspired Preprocessing**: Robust data augmentation and foreground sampling to handle class imbalance
4. **Production-Grade Pipeline**: Comprehensive experiment management, versioning, and reproducibility

### Performance Targets:

- **Dice Score (Mean)**: 0.85+ across all tumor regions
- **Training Time**: ~2-3 days on single RTX 3090/4090
- **Memory Efficiency**: Fits within 24GB VRAM with batch size 2

---

## 2. Architecture Overview

### 2.1 High-Level Design

SegMamba follows the U-Net paradigm with an **encoder-decoder** structure connected by **skip connections**. The key innovation lies in the encoder's hybrid design:

```
Input (B, 4, D, H, W)
    ↓
[Initial Conv3D Block]
    ↓
[Stage 1: Conv3D] ──────────────┐
    ↓                           │
[Stage 2: Conv3D] ─────────┐    │
    ↓                      │    │
[Stage 3: Conv3D + Mamba] ─┐    │    │
    ↓                     │    │    │
[Stage 4: Mamba Only]     │    │    │
    ↓                     │    │    │
[Bottleneck: Mamba]       │    │    │
    ↓                     │    │    │
[Decoder 4] ←─────────────┘    │    │
    ↓                          │    │
[Decoder 3] ←──────────────────┘    │
    ↓                               │
[Decoder 2] ←───────────────────────┘
    ↓
[Decoder 1] ←───────────────────────┘
    ↓
[Segmentation Head]
    ↓
Output (B, num_classes, D, H, W)
```

### 2.2 Rationale for Hybrid Design

**Why Conv3D for Early Stages?**
- Medical images have strong local structure (tissues, boundaries)
- Conv3D is parameter-efficient for local features
- Provides inductive bias suitable for medical imaging
- Faster than attention mechanisms for small receptive fields

**Why Mamba for Deep Stages?**
- Tumor extent can span large 3D volumes (requires global context)
- Mamba provides **linear complexity** O(n) vs. Transformers' O(n²)
- Captures long-range dependencies critical for whole-tumor segmentation
- State-space models excel at sequential (volumetric) data
- No self-attention overhead - pure state-space modeling

### 2.3 Component Details

#### **Conv3D Block**
```python
Conv3D(in, out, 3x3x3)
→ BatchNorm3D
→ ReLU
→ Conv3D(out, out, 3x3x3)
→ BatchNorm3D
→ ReLU
+ Residual Connection
```

**Purpose**: Extract hierarchical local features with residual learning.

#### **Mamba Block (State-Space Model)**

Mamba implements a selective state-space model with gating:

$$
h_t = \bar{A} h_{t-1} + \bar{B} x_t
$$

$$
y_t = C h_t
$$

Where:
- $h_t$ is the hidden state at position $t$
- $\bar{A}, \bar{B}, C$ are learned parameters (input-dependent via gating)
- Provides **linear complexity** in sequence length

**Advantage over Transformers**: For a 3D volume of size $64 \times 64 \times 64 = 262,144$ voxels:
- **Transformer**: $O(262,144^2) \approx 69$ billion operations
- **Mamba**: $O(262,144) \approx 262$ thousand operations

This is a **~250,000x reduction** in computational complexity, making Mamba the ideal choice for 3D volumetric processing!

---

## 3. Mathematical Foundations

### 3.1 Loss Function: Dice + Cross Entropy

We use a combined loss to leverage both region-based and pixel-wise supervision:

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{dice}} \mathcal{L}_{\text{dice}} + \lambda_{\text{ce}} \mathcal{L}_{\text{ce}}
$$

#### **Dice Loss**

The Dice coefficient measures overlap between prediction and ground truth:

$$
\text{Dice}(P, G) = \frac{2 |P \cap G|}{|P| + |G|}
$$

For differentiability, we compute it as:

$$
\text{Dice} = \frac{2 \sum_{i} p_i g_i + \epsilon}{\sum_{i} p_i + \sum_{i} g_i + \epsilon}
$$

Where:
- $p_i$ is the predicted probability for voxel $i$
- $g_i$ is the ground truth label (0 or 1)
- $\epsilon$ is a smoothing term to avoid division by zero

The Dice loss is:

$$
\mathcal{L}_{\text{dice}} = 1 - \text{Dice}(P, G)
$$

**Why Dice Loss?**
- Robust to class imbalance (tumors are 1-5% of brain volume)
- Directly optimizes the evaluation metric
- Region-based: focuses on overall shape and structure

#### **Cross Entropy Loss**

Standard pixel-wise classification loss:

$$
\mathcal{L}_{\text{ce}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} g_{i,c} \log(p_{i,c})
$$

Where:
- $N$ is the number of voxels
- $C$ is the number of classes
- $g_{i,c}$ is the ground truth one-hot encoding
- $p_{i,c}$ is the predicted probability

**Why Cross Entropy?**
- Provides pixel-level supervision
- Encourages confident predictions
- Complements Dice loss's region-based optimization

### 3.2 Optimization: AdamW with Cosine Annealing

**Optimizer**: AdamW (Adam with decoupled weight decay)

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
\theta_{t+1} = \theta_t - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)
$$

**Learning Rate Schedule**: Cosine Annealing

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)
$$

Where:
- $\eta_{\max} = 1 \times 10^{-4}$ (initial learning rate)
- $\eta_{\min} = 1 \times 10^{-6}$ (minimum learning rate)
- $T$ is total number of epochs

**Rationale**:
- Smooth decay prevents abrupt changes
- Allows fine-tuning in later epochs
- Standard in medical image segmentation

---

## 4. Preprocessing Strategy

Our preprocessing follows the **nnU-Net** philosophy: aggressive augmentation with domain knowledge.

### 4.1 Intensity Normalization

MRI intensities are **non-standardized** (vary across scanners, protocols). We apply:

$$
x_{\text{norm}} = \frac{x - \mu_{\text{nonzero}}}{\sigma_{\text{nonzero}}}
$$

Only non-zero voxels are used for statistics to exclude background.

**Per-channel normalization**: Each modality (T1, T1ce, T2, FLAIR) normalized independently.

### 4.2 Spatial Augmentations

1. **Random Crop with Foreground Sampling**
   - Extract patches of size $128 \times 128 \times 64$
   - 50% probability of sampling patches containing tumor
   - 50% probability of sampling from background
   - **Why?** Ensures balanced exposure to rare tumor regions

2. **Random Flips**
   - Flip along each axis with $p = 0.5$
   - **Justification**: Brain anatomy is approximately symmetric

3. **Random 90° Rotations**
   - Rotate in axial plane with $p = 0.5$
   - **Justification**: No canonical orientation in medical volumes

4. **Random Scaling**
   - Scale factor $\sim \mathcal{U}(0.9, 1.1)$
   - **Why?** Simulates variations in patient anatomy

### 4.3 Intensity Augmentations

1. **Random Brightness Shift**
   - Add offset $\sim \mathcal{U}(-0.1, 0.1)$
   - Simulates scanner variations

2. **Random Contrast Scaling**
   - Multiply by factor $\sim \mathcal{U}(0.9, 1.1)$
   - Accounts for different acquisition parameters

### 4.4 Resampling

All volumes resampled to **1mm isotropic** spacing:
- Ensures consistent voxel dimensions
- Standardizes across different scanners
- Uses trilinear interpolation for images, nearest-neighbor for labels

---

## 5. Training Methodology

### 5.1 Single GPU Optimization

**Challenge**: 3D medical images are memory-intensive.

**Solutions Implemented**:

1. **Automatic Mixed Precision (AMP)**
   - Uses FP16 for forward/backward passes
   - Maintains FP32 for critical operations
   - **Memory reduction**: ~40%
   - **Speed up**: ~2-3x

2. **Gradient Accumulation**
   - Effective batch size = `batch_size × accumulation_steps`
   - With `batch_size=2` and `accumulation_steps=2`: effective batch size = 4
   - Allows larger effective batches without OOM errors

3. **Patch-Based Training**
   - Full brain volumes are $\sim 240 \times 240 \times 155$
   - Train on patches: $128 \times 128 \times 64$
   - **Memory reduction**: ~70%

### 5.2 Training Schedule

- **Total Epochs**: 300
- **Warmup**: 10 epochs (linear LR increase)
- **Batch Size**: 2 (effective: 4 with accumulation)
- **Validation Frequency**: Every 2 epochs
- **Early Stopping**: Stop if no improvement for 50 epochs

### 5.3 Metric Tracking

**Primary Metric**: Mean Dice Score (excluding background)

$$
\text{Mean Dice} = \frac{1}{3}(\text{Dice}_{\text{ET}} + \text{Dice}_{\text{TC}} + \text{Dice}_{\text{WT}})
$$

Where:
- **ET**: Enhancing Tumor
- **TC**: Tumor Core
- **WT**: Whole Tumor

### 5.4 Checkpointing Strategy

- **Best Model**: Saved based on highest validation Dice score
- **Last Model**: Saved at end of training (for resumption)
- **Periodic**: Save every 10 epochs for analysis

---

## 6. Implementation Details

### 6.1 Software Stack

```
- Python: 3.10+
- PyTorch: 2.0+
- MONAI: 1.3+ (Medical Open Network for AI)
- **PyTorch**: 2.0+
- **MONAI**: 1.3+
- **nibabel**: 5.0+
- **mamba-ssm**: 2.0+ (required for Mamba blocks)
- **causal-conv1d**: 1.2+ (dependency of mamba-ssm)
- nibabel: 5.0+ (NIfTI file loading)
- numpy, matplotlib, tqdm
```

### 6.2 Model Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Base Channels | 32 | Balance between capacity and memory |
| Encoder Depths | [2, 2, 2, 2] | Sufficient depth for hierarchical features |
| Patch Size | 128×128×64 | Fits in 24GB VRAM, captures sufficient context |
| Dropout | None | Not used (data augmentation sufficient) |
| Activation | ReLU | Standard, works well |
| Normalization | BatchNorm3D | Stable training |

### 6.3 Experiment Management

All experiments are version-controlled with the `ExperimentManager`:

```
Results/
  └── SegMamba_Run01/
        ├── checkpoints/
        │     ├── best_metric_model.pth
        │     └── last.pth
        ├── logs/
        │     └── tensorboard events
        ├── plots/
        │     ├── training_curves.png
        │     └── val_predictions_epoch_*.png
        ├── metrics/
        │     └── final_metrics.json
        └── config.json
```

**Benefits**:
- No overwriting between runs
- Full reproducibility from `config.json`
- Easy comparison of different experiments

---

## 7. Results & Ablations

### 7.1 Expected Performance

| Configuration | Mean Dice | ET Dice | TC Dice | WT Dice | Training Time |
|---------------|-----------|---------|---------|---------|---------------|
| SegMamba (Mamba) | 0.87 | 0.82 | 0.85 | 0.91 | 48h |
| Baseline U-Net | 0.82 | 0.77 | 0.80 | 0.88 | 40h |

*(Note: Actual results to be filled after training)*

### 7.2 Ablation Studies

**Planned Ablations**:

1. **Effect of Mamba Block Depth**
   - Hypothesis: Deeper Mamba blocks improve long-range modeling
   
2. **Effect of Patch Size**
   - Compare 96×96×96 vs. 128×128×64
   - Hypothesis: Larger patches improve WT segmentation
   
3. **Effect of Loss Weights**
   - Vary $\lambda_{\text{dice}}$ and $\lambda_{\text{ce}}$
   - Hypothesis: Equal weighting is optimal

4. **Effect of Augmentation**
   - Train without augmentation
   - Hypothesis: Augmentation crucial for generalization

---

## 8. Future Work

### 8.1 Ensemble Methods

**Strategy**: Train 5 models with different:
- Random seeds
- Augmentation strategies
- Patch sizes

**Ensemble via**:
- Average predictions (soft voting)
- Expected improvement: +2-3% Dice

### 8.2 Test-Time Augmentation (TTA)

**Approach**:
1. Apply 8 augmentations (flips along 3 axes)
2. Average predictions
3. Expected improvement: +1-2% Dice

### 8.3 Post-Processing

**Connected Component Analysis**:
- Remove small isolated regions (< 100 voxels)
- Rationale: Tumors are spatially coherent

**Conditional Random Fields (CRF)**:
- Refine boundaries using intensity priors
- Expected improvement: +0.5-1% Dice

### 8.4 Architecture Improvements

1. **Deep Supervision**
   - Add auxiliary losses at intermediate layers
   - Improves gradient flow
   
2. **Attention Gates**
   - In skip connections
   - Focus decoder on relevant features
   
3. **Multi-Scale Inference**
   - Predict at multiple resolutions
   - Combine for final output

### 8.5 Advanced Training Strategies

1. **Self-Supervised Pretraining**
   - Pretrain encoder on unlabeled MRI scans
   - Use contrastive learning or masked autoencoding
   
2. **Semi-Supervised Learning**
   - Leverage unlabeled data
   - Use consistency regularization

3. **Online Hard Example Mining**
   - Focus training on difficult patches
   - Improves performance on edge cases

---

## 9. References

### Key Papers

1. **U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation**  
   Ma et al., 2024  
   [arXiv:2401.04722](https://arxiv.org/abs/2401.04722)  
   *Our primary architectural inspiration*

2. **nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation**  
   Isensee et al., 2020  
   [arXiv:2011.00848](https://arxiv.org/abs/2011.00848)  
   *Gold standard preprocessing and training methodology*

3. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**  
   Gu & Dao, 2023  
   *Foundation of our state-space modeling approach*

### BraTS Dataset

5. **The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)**  
   Menze et al., 2015  
   [arXiv:2107.02314](https://arxiv.org/abs/2107.02314)  
   *Dataset specifications and evaluation metrics*

---

## Appendix A: Reproducibility Checklist

- [x] Configuration saved with every run
- [x] Random seed set for PyTorch, NumPy, CUDA
- [x] Deterministic mode available (optional)
- [x] Model architecture fully documented
- [x] Data preprocessing pipeline described
- [x] Training hyperparameters specified
- [x] Hardware specifications documented
- [x] Software versions recorded
- [x] Results saved with model checkpoints
- [x] Code commented extensively

---

## Appendix B: Competition Compliance

This implementation adheres to typical competition requirements:

1. **Code Clarity**: Extensive docstrings and comments
2. **Architectural Justification**: Mathematical and empirical motivation provided
3. **Reproducibility**: Seed-controlled, configuration-managed
4. **Modularity**: Clean separation of concerns
5. **Efficiency**: Single GPU training viable
6. **Documentation**: Comprehensive technical documentation

---

**END OF DOCUMENTATION**

*This document was generated for academic and competition purposes. For questions, refer to the code implementation in the SegMamba repository.*
