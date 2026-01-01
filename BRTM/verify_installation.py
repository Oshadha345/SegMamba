#!/usr/bin/env python3
"""
SegMamba Installation Verification Script
Checks if all dependencies are installed and project structure is correct.
"""

import sys
from pathlib import Path

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def check_imports():
    """Check if all required packages can be imported."""
    print_section("Checking Python Packages")
    
    packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'tqdm': 'TQDM',
        'nibabel': 'NiBabel (NIfTI support)',
    }
    
    optional_packages = {
        'monai': 'MONAI (Medical imaging - HIGHLY RECOMMENDED)',
        'mamba_ssm': 'Mamba SSM (State-space models - Optional)',
        'tensorboard': 'TensorBoard (Logging)',
    }
    
    print("\nRequired Packages:")
    all_required_ok = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT FOUND")
            all_required_ok = False
    
    print("\nOptional Packages:")
    for package, name in optional_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ○ {name} - Not installed (optional)")
    
    return all_required_ok

def check_project_structure():
    """Check if all project files exist."""
    print_section("Checking Project Structure")
    
    required_files = [
        'config.py',
        'train.py',
        'requirements.txt',
        'README.md',
        'models/__init__.py',
        'models/segmamba.py',
        'data/__init__.py',
        'data/brats_dataset.py',
        'utils/__init__.py',
        'utils/experiment_manager.py',
        'utils/metrics.py',
        'utils/visualization.py',
        'notebooks/SegMamba_Training.ipynb',
        'docs/SegMamba_Documentation.md',
    ]
    
    project_root = Path(__file__).parent
    all_files_ok = True
    
    for file in required_files:
        file_path = project_root / file
        if file_path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - NOT FOUND")
            all_files_ok = False
    
    return all_files_ok

def check_cuda():
    """Check CUDA availability."""
    print_section("Checking CUDA/GPU")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"  ✓ CUDA Available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU Memory: {total_memory:.2f} GB")
            
            if total_memory < 16:
                print(f"  ⚠️  Warning: GPU has < 16GB VRAM. Consider reducing batch size.")
            
            return True
        else:
            print(f"  ✗ CUDA Not Available")
            print(f"  Training will be very slow on CPU!")
            return False
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False

def check_config():
    """Check if configuration is set up."""
    print_section("Checking Configuration")
    
    try:
        from config import Config
        
        print(f"  Run Name: {Config.RUN_NAME}")
        print(f"  Data Root: {Config.DATA_ROOT}")
        print(f"  Data Root Exists: {Config.DATA_ROOT.exists()}")
        
        if not Config.DATA_ROOT.exists():
            print(f"  ⚠️  Warning: Data root does not exist. Update config.py with your dataset path.")
        
        print(f"  Patch Size: {Config.PATCH_SIZE}")
        print(f"  Batch Size: {Config.BATCH_SIZE}")
        print(f"  Use AMP: {Config.USE_AMP}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error loading config: {e}")
        return False

def test_model_import():
    """Test if model can be imported and created."""
    print_section("Testing Model Import")
    
    try:
        from models import SegMamba
        
        print("  ✓ SegMamba imported successfully")
        
        # Try to create model
        model = SegMamba(in_channels=4, num_classes=4, base_channels=16)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model created successfully")
        print(f"  Parameters: {param_count:,}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification checks."""
    print("\n" + "█" * 70)
    print("  SegMamba Installation Verification")
    print("█" * 70)
    
    checks = [
        ("Python Packages", check_imports),
        ("Project Structure", check_project_structure),
        ("CUDA/GPU", check_cuda),
        ("Configuration", check_config),
        ("Model Import", test_model_import),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
            results[name] = False
    
    # Summary
    print_section("Verification Summary")
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:8} - {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    
    if all_passed:
        print("\n✓ All checks passed! Ready to train.")
        print("\nNext steps:")
        print("  1. Update config.py with your dataset path")
        print("  2. Run: python train.py")
        print("  3. Or open: notebooks/SegMamba_Training.ipynb")
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Update dataset path in config.py")
        print("  - Ensure CUDA drivers are installed")
    
    print("\n" + "=" * 70 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
