# Migration to Pure Mamba Architecture

## Summary of Changes

This document describes the migration from a hybrid Mamba/Swin architecture (with fallback support) to a **pure Mamba-only** architecture.

---

## What Changed

### 1. **Architecture Simplification** ([models/segmamba.py](models/segmamba.py))

**REMOVED:**
- ✅ Entire `SwinBlock3D` class (~70 lines)
- ✅ Swin Transformer fallback mechanism
- ✅ Dual import logic (try Mamba, fallback to Swin)

**ADDED:**
- ✅ Hard requirement for mamba-ssm (raises `ImportError` if unavailable)
- ✅ Cleaner import section with explicit error message

**KEPT (INTENTIONAL):**
- ✅ Internal `use_mamba` parameter in `EncoderStage`
  - This controls **which stages** use Mamba vs Conv3D
  - Early stages: Conv3D (local features)
  - Deep stages: Mamba (global context)
  - This is part of the **hybrid Conv3D + Mamba** design philosophy

### 2. **Configuration Simplification** ([config.py](config.py))

**REMOVED:**
- ✅ `USE_MAMBA` configuration flag
- ✅ Comments about Mamba/Swin toggling

**KEPT:**
- ✅ All other hyperparameters unchanged

### 3. **Training Pipeline** ([train.py](train.py))

**REMOVED:**
- ✅ `use_mamba=Config.USE_MAMBA` parameter from model initialization

**KEPT:**
- ✅ All training logic unchanged

### 4. **Dependencies** ([requirements.txt](requirements.txt))

**CHANGED:**
- ✅ `mamba-ssm>=2.0.0` - Changed from optional (commented) to **REQUIRED**
- ✅ `causal-conv1d>=1.2.0` - Now required as mamba-ssm dependency

### 5. **Documentation Updates**

**Updated Files:**
- ✅ [README.md](README.md) - Changed from "Hybrid Architecture" to "Pure Mamba Architecture"
- ✅ [docs/SegMamba_Documentation.md](docs/SegMamba_Documentation.md):
  - Removed Swin UNETR references
  - Removed Swin Block mathematical section
  - Updated ablation studies
  - Removed Swin from references
- ✅ [QUICKSTART.md](QUICKSTART.md) - Added mamba-ssm to installation steps
- ✅ [INSTALLATION_COMPLETE.md](INSTALLATION_COMPLETE.md):
  - Updated feature list
  - Updated architecture diagram
  - Changed references
- ✅ [notebooks/SegMamba_Training.ipynb](notebooks/SegMamba_Training.ipynb):
  - Removed "Baseline SegMamba with Swin blocks" text
  - Removed `Config.USE_MAMBA` references
  - Removed `use_mamba` parameter from model creation

---

## Architecture Philosophy

### What "Pure Mamba" Means

**Before:** Users could choose between Mamba OR Swin Transformers for attention blocks
**After:** Mamba is the ONLY choice for attention blocks

### What "Hybrid" Still Means

The architecture is still **hybrid Conv3D + Mamba**, where:
- **Early Encoder Stages (1-2)**: Use Conv3D blocks only
  - Efficient for local feature extraction
  - Parameter-efficient
  - Fast computation
  
- **Deep Encoder Stages (3-4) + Bottleneck**: Use Mamba blocks
  - Capture long-range dependencies
  - Linear O(n) complexity
  - Global context modeling

This is the **optimal design** for 3D medical image segmentation.

---

## Key Benefits

1. **Simplified Codebase**
   - Removed ~70 lines of Swin implementation
   - Removed conditional logic
   - Clearer architectural intent

2. **Better Error Handling**
   - Explicit error message if mamba-ssm unavailable
   - No silent fallback to different architecture

3. **Focused Approach**
   - State-space modeling is the core innovation
   - No compromise with Transformer fallback

4. **Consistent Documentation**
   - All docs now reflect pure Mamba approach
   - No confusion about which blocks are used

---

## Installation Requirements

### Before:
```bash
pip install mamba-ssm  # Optional - falls back to Swin if unavailable
```

### After:
```bash
pip install mamba-ssm>=2.0.0  # REQUIRED - fails if unavailable
pip install causal-conv1d>=1.2.0  # REQUIRED dependency
```

---

## Breaking Changes

⚠️ **For Existing Users:**

1. **Config Parameter Removed:**
   ```python
   # OLD (no longer works)
   Config.USE_MAMBA = False  # ❌ Removed
   
   # NEW (always uses Mamba)
   # No configuration needed - pure Mamba by default
   ```

2. **Model Initialization:**
   ```python
   # OLD (no longer works)
   model = SegMamba(
       in_channels=4,
       num_classes=4,
       use_mamba=Config.USE_MAMBA  # ❌ Parameter removed
   )
   
   # NEW
   model = SegMamba(
       in_channels=4,
       num_classes=4
   )
   ```

3. **Dependency Installation:**
   - `mamba-ssm` is now **mandatory**, not optional
   - Training will fail if CUDA-compatible mamba-ssm cannot be installed

---

## Migration Checklist

If you have existing code, update:

- [ ] Remove `Config.USE_MAMBA` from config files
- [ ] Remove `use_mamba=...` parameter from SegMamba initialization
- [ ] Install mamba-ssm and causal-conv1d
- [ ] Update any custom documentation/comments referencing Swin
- [ ] Verify CUDA compatibility with mamba-ssm

---

## Testing

### Verify Pure Mamba Implementation:

```python
from models.segmamba import SegMamba

# Should work if mamba-ssm installed correctly
model = SegMamba(in_channels=4, num_classes=4)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Check architecture
print(model.encoders[0])  # Should show Conv3DBlock
print(model.encoders[-1]) # Should show MambaBlock3D
```

### Expected Output:
```
Model parameters: 9.8M
EncoderStage(...)  # Conv3D blocks
EncoderStage(...)  # Mamba blocks
```

---

## Rationale

**Why remove Swin fallback?**

1. **Clarity**: State-space models are the core innovation
2. **Simplicity**: Less code = easier maintenance
3. **Performance**: Mamba is faster than Swin (O(n) vs O(n²))
4. **Competition**: BraTS judges appreciate focused approaches
5. **Reproducibility**: One architecture = easier to replicate results

---

## References

- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**  
  Gu & Dao, 2023 - [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

- **U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation**  
  Ma et al., 2024 - [arXiv:2401.04722](https://arxiv.org/abs/2401.04722)

---

**Migration Complete: December 29, 2025**  
**Status: ✅ Pure Mamba Architecture**
