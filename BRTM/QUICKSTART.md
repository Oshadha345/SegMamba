# SegMamba Quick Start Guide

## 🚀 5-Minute Setup

### Step 1: Navigate to Project
```bash
cd /storage2/CV_Irradiance/VMamba/BRTM
```

### Step 2: Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install monai nibabel numpy matplotlib tqdm tensorboard
pip install mamba-ssm>=2.0.0 causal-conv1d>=1.2.0  # Required for Mamba blocks
```

### Step 3: Update Configuration
Edit `config.py`:
```python
RUN_NAME = "SegMamba_Run01"  # Change this for each run!
DATA_ROOT = Path("/path/to/your/BraTS/dataset")
```

### Step 4: Train
**Option A - Jupyter Notebook (Recommended):**
```bash
jupyter notebook notebooks/SegMamba_Training.ipynb
```

**Option B - Python Script:**
```bash
python train.py
```

---

## 📊 Monitor Training

### Check Progress
```bash
# View training curves
ls -lh results/SegMamba_Run01/plots/training_curves.png

# Check metrics
cat results/SegMamba_Run01/metrics/final_metrics.json
```

### Load Best Model
```python
import torch
from models import SegMamba

model = SegMamba()
checkpoint = torch.load("results/SegMamba_Run01/checkpoints/best_metric_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 🔧 Common Issues

### Issue: Out of Memory (OOM)
**Solution**: Reduce patch size in `config.py`
```python
PATCH_SIZE = (96, 96, 96)  # Instead of (128, 128, 64)
BATCH_SIZE = 1  # Instead of 2
```

### Issue: MONAI Not Found
**Solution**: Install MONAI
```bash
pip install monai
```

### Issue: Dataset Not Found
**Solution**: Check paths
```python
# In config.py
DATA_ROOT = Path("/storage2/CV_Irradiance/datasets/CVMD/BraTS")
print(DATA_ROOT.exists())  # Should be True
```

### Issue: Mamba SSM Install Fails
**Solution**: Use Swin fallback
```python
# In config.py
USE_MAMBA = False  # Uses Swin Transformer instead
```

---

## 📈 Expected Timeline

| Phase | Duration | What's Happening |
|-------|----------|------------------|
| Setup | 5 min | Install dependencies, verify paths |
| Data Loading | 2 min | First epoch initialization |
| Training | 48-72 hrs | Main training loop |
| Validation | 5 min/epoch | Compute metrics, save predictions |

---

## 🎯 Performance Targets

| Metric | Target | Competition-Winning |
|--------|--------|---------------------|
| Mean Dice | 0.85+ | 0.88+ |
| ET Dice | 0.80+ | 0.85+ |
| TC Dice | 0.83+ | 0.87+ |
| WT Dice | 0.90+ | 0.92+ |

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `config.py` | **MODIFY THIS** - All hyperparameters |
| `train.py` | Main training script |
| `models/segmamba.py` | Model architecture |
| `data/brats_dataset.py` | Data loading |
| `utils/experiment_manager.py` | Result organization |
| `notebooks/SegMamba_Training.ipynb` | Interactive training |

---

## 🔬 Advanced Features

### Multiple Runs
```bash
# Run 1
python train.py  # RUN_NAME = "SegMamba_Run01"

# Run 2 (change config.py first)
python train.py  # RUN_NAME = "SegMamba_Run02"
```

### Resume Training
```python
# In train.py or notebook
checkpoint = exp_manager.load_checkpoint("last.pth")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Inference
```python
model.eval()
with torch.no_grad():
    output = model(input_volume)
    prediction = torch.argmax(output, dim=1)
```

---

## 📚 Documentation

- **README.md**: Project overview
- **docs/SegMamba_Documentation.md**: Detailed technical documentation
- **requirements.txt**: All dependencies

---

## ✅ Pre-Training Checklist

- [ ] Python 3.10+ installed
- [ ] CUDA 11.8+ available
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset downloaded and structured correctly
- [ ] Paths updated in `config.py`
- [ ] `RUN_NAME` changed to unique value
- [ ] Enough disk space (50GB+ for results)
- [ ] GPU has 16GB+ VRAM

---

## 🆘 Getting Help

1. **Check logs**: `results/{RUN_NAME}/logs/`
2. **Review documentation**: `docs/SegMamba_Documentation.md`
3. **Verify configuration**: `Config.print_config()`
4. **Test data loading**: Run first cell of notebook

---

**Ready to train? Good luck! 🧠🔬**
