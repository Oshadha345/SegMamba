"""
Configuration File for SegMamba Training
Centralized hyperparameters and paths for reproducibility.
"""

import torch
from pathlib import Path


class Config:
    """
    Centralized configuration for SegMamba training.
    
    Modify this class to change training parameters.
    All parameters are documented for competition reproducibility.
    """
    
    # ============================================================================
    # EXPERIMENT IDENTIFICATION
    # ============================================================================
    RUN_NAME = "SegMamba_Run01"  # CHANGE THIS FOR EACH NEW TRAINING RUN
    DESCRIPTION = "Baseline SegMamba with Conv3D encoder + Swin blocks in bottleneck"
    
    # ============================================================================
    # PATHS
    # ============================================================================
    # Dataset paths (UPDATE THESE TO YOUR DATASET LOCATION)
    DATA_ROOT = Path("/storage2/CV_Irradiance/datasets/CVMD/BraTS")  # Update this path
    TRAIN_DATA_PATH = "/storage2/CV_Irradiance/VMamba/BRTM/data/TrainingData"
    VAL_DATA_PATH = "/storage2/CV_Irradiance/VMamba/BRTM/data/ValidationData"
    
    # Results will be saved outside the code directory
    RESULTS_BASE_PATH = Path("/storage2/CV_Irradiance/VMamba/BRTM/results")
    
    # ============================================================================
    # MODEL ARCHITECTURE
    # ============================================================================
    IN_CHANNELS = 4  # T1, T1ce, T2, FLAIR
    NUM_CLASSES = 4  # Background, NCR/NET, ED, ET (or 3 for TC, WT, ET)
    
    # Spatial dimensions for 3D patches
    # Adjust based on GPU VRAM:
    # - (128, 128, 128): ~16-24GB VRAM
    # - (96, 96, 96): ~10-14GB VRAM  
    # - (128, 128, 64): Good balance for single GPU
    PATCH_SIZE = (128, 128, 64)  
    
    # Model architecture parameters
    BASE_CHANNELS = 32  # Starting number of channels (increases with depth)
    ENCODER_DEPTHS = [2, 2, 2, 2]  # Number of blocks per encoder stage
    USE_MAMBA = True  # Set to True if mamba_ssm installed; False uses Swin fallback
    USE_CHECKPOINT = False  # Gradient checkpointing (saves memory, slower)
    
    # ============================================================================
    # TRAINING HYPERPARAMETERS
    # ============================================================================
    # Single GPU optimization
    BATCH_SIZE = 2  # Reduce if OOM errors occur
    ACCUMULATION_STEPS = 2  # Effective batch size = BATCH_SIZE * ACCUMULATION_STEPS
    NUM_EPOCHS = 300
    
    # Learning rate schedule
    INITIAL_LR = 1e-4
    LR_SCHEDULER = "cosine"  # Options: "cosine", "poly", "step"
    WARMUP_EPOCHS = 10  # Linear warmup for first N epochs
    MIN_LR = 1e-6  # Minimum LR for cosine annealing
    
    # Optimizer
    OPTIMIZER = "AdamW"  # Options: "AdamW", "SGD"
    WEIGHT_DECAY = 1e-5
    
    # Loss function
    DICE_WEIGHT = 0.5  # Weight for Dice loss
    CE_WEIGHT = 0.5  # Weight for Cross Entropy loss
    
    # ============================================================================
    # DATA AUGMENTATION (nnU-Net inspired)
    # ============================================================================
    # Intensity augmentations
    NORMALIZE_INTENSITY = True
    INTENSITY_SHIFT_RANGE = 0.1  # Random brightness adjustment
    
    # Spatial augmentations
    RANDOM_FLIP_PROB = 0.5  # Probability of random flip
    RANDOM_ROTATE_PROB = 0.5  # Probability of random 90-degree rotation
    RANDOM_SCALE_RANGE = (0.9, 1.1)  # Random zoom
    
    # Foreground sampling (critical for class imbalance)
    POS_SAMPLE_RATIO = 0.5  # 50% of samples include tumor voxels
    NUM_SAMPLES = 4  # Number of patches per volume
    
    # ============================================================================
    # VALIDATION & CHECKPOINTING
    # ============================================================================
    VAL_INTERVAL = 2  # Validate every N epochs
    SAVE_INTERVAL = 10  # Save checkpoint every N epochs
    EARLY_STOPPING_PATIENCE = 50  # Stop if no improvement for N epochs
    
    # Metric for best model selection
    METRIC_NAME = "mean_dice"  # Options: "mean_dice", "dice_ET", "dice_TC", "dice_WT"
    METRIC_HIGHER_BETTER = True
    
    # ============================================================================
    # COMPUTATIONAL SETTINGS
    # ============================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # DataLoader workers (0 for debugging)
    PIN_MEMORY = True  # Pin memory for faster GPU transfer
    USE_AMP = True  # Automatic Mixed Precision (MANDATORY for 3D on single GPU)
    
    # Reproducibility
    SEED = 42
    DETERMINISTIC = False  # Set True for full reproducibility (slower)
    
    # ============================================================================
    # LOGGING
    # ============================================================================
    LOG_INTERVAL = 10  # Print training stats every N batches
    USE_TENSORBOARD = True
    SAVE_TRAIN_IMAGES = True  # Save sample predictions during training
    
    # ============================================================================
    # INFERENCE (for testing after training)
    # ============================================================================
    TEST_TIME_AUGMENTATION = True  # TTA for final submission
    SLIDING_WINDOW_INFERENCE = True  # For full volume inference
    INFERENCE_OVERLAP = 0.5  # Overlap ratio for sliding window
    
    @classmethod
    def get_config_dict(cls):
        """
        Return all configuration as a dictionary for saving.
        """
        config_dict = {}
        for key in dir(cls):
            if key.isupper():  # Only capture UPPERCASE attributes
                config_dict[key] = getattr(cls, key)
        return config_dict
    
    @classmethod
    def print_config(cls):
        """
        Pretty print configuration.
        """
        print("\n" + "="*70)
        print(f"{'SegMamba Configuration':^70}")
        print("="*70)
        
        sections = {
            "EXPERIMENT": ["RUN_NAME", "DESCRIPTION"],
            "PATHS": ["DATA_ROOT", "RESULTS_BASE_PATH"],
            "MODEL": ["IN_CHANNELS", "NUM_CLASSES", "PATCH_SIZE", "BASE_CHANNELS", "USE_MAMBA"],
            "TRAINING": ["BATCH_SIZE", "ACCUMULATION_STEPS", "NUM_EPOCHS", "INITIAL_LR"],
            "COMPUTE": ["DEVICE", "USE_AMP", "NUM_WORKERS"]
        }
        
        for section, keys in sections.items():
            print(f"\n{section}:")
            for key in keys:
                if hasattr(cls, key):
                    value = getattr(cls, key)
                    print(f"  {key:.<40} {value}")
        
        print("\n" + "="*70 + "\n")


# Create a global config instance
config = Config()


if __name__ == "__main__":
    # Test configuration
    Config.print_config()
