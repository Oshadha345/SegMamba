"""
ExperimentManager: Manages experiment artifacts and prevents overwriting
Author: Competitive ML Engineer
Purpose: Organize training runs with automatic directory creation and versioning
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import torch


class ExperimentManager:
    """
    Manages all artifacts for a single training run.
    
    Creates a structured directory for checkpoints, logs, plots, and metrics.
    Ensures no overwriting between different training runs.
    
    Directory Structure:
        Results/
          └── {RUN_NAME}/
                ├── checkpoints/    (model weights)
                ├── logs/           (tensorboard/CSV logs)
                ├── plots/          (visualizations)
                ├── metrics/        (evaluation results)
                └── config.json     (saved configuration)
    
    Args:
        run_name (str): Unique identifier for this training run
        base_path (str): Root directory for all results (default: './results')
        overwrite (bool): If False, raises error if run_name exists (default: False)
    """
    
    def __init__(
        self, 
        run_name: str,
        base_path: str = "./results",
        overwrite: bool = False
    ):
        self.run_name = run_name
        self.base_path = Path(base_path)
        self.run_path = self.base_path / run_name
        
        # Define subdirectories
        self.checkpoints_dir = self.run_path / "checkpoints"
        self.logs_dir = self.run_path / "logs"
        self.plots_dir = self.run_path / "plots"
        self.metrics_dir = self.run_path / "metrics"
        
        # Check for existing run
        if self.run_path.exists() and not overwrite:
            raise ValueError(
                f"Run '{run_name}' already exists at {self.run_path}. "
                f"Use overwrite=True or choose a different run_name."
            )
        
        # Create directory structure
        self._create_directories()
        
        # Log creation time
        self.created_at = datetime.now().isoformat()
        
        print(f"[ExperimentManager] Initialized run: {run_name}")
        print(f"[ExperimentManager] Results will be saved to: {self.run_path}")
    
    def _create_directories(self):
        """Create all subdirectories for the experiment."""
        directories = [
            self.checkpoints_dir,
            self.logs_dir,
            self.plots_dir,
            self.metrics_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"[ExperimentManager] Created directory structure at {self.run_path}")
    
    def save_config(self, config: Dict[str, Any]):
        """
        Save configuration dictionary to JSON file.
        
        Args:
            config (dict): Configuration dictionary to save
        """
        config_path = self.run_path / "config.json"
        
        # Add metadata
        config_with_meta = {
            "run_name": self.run_name,
            "created_at": self.created_at,
            "config": config
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_with_meta, f, indent=4)
        
        print(f"[ExperimentManager] Saved config to {config_path}")
    
    def save_checkpoint(
        self, 
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        filename: str = "checkpoint.pth",
        is_best: bool = False
    ):
        """
        Save model checkpoint with full training state.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer instance
            epoch: Current epoch number
            metrics: Dictionary of metrics (e.g., {'dice': 0.85, 'loss': 0.15})
            filename: Name for the checkpoint file
            is_best: If True, also saves as 'best_metric_model.pth'
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'run_name': self.run_name,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.checkpoints_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"[ExperimentManager] Saved checkpoint to {checkpoint_path}")
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoints_dir / "best_metric_model.pth"
            shutil.copy(checkpoint_path, best_path)
            print(f"[ExperimentManager] Saved best model to {best_path}")
    
    def load_checkpoint(self, filename: str = "best_metric_model.pth") -> Dict:
        """
        Load a checkpoint file.
        
        Args:
            filename: Name of checkpoint file to load
            
        Returns:
            Dictionary containing checkpoint data
        """
        checkpoint_path = self.checkpoints_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        print(f"[ExperimentManager] Loaded checkpoint from {checkpoint_path}")
        
        return checkpoint
    
    def get_checkpoint_path(self, filename: str) -> Path:
        """Return full path to checkpoint file."""
        return self.checkpoints_dir / filename
    
    def get_plot_path(self, filename: str) -> Path:
        """Return full path to plot file."""
        return self.plots_dir / filename
    
    def get_log_path(self, filename: str) -> Path:
        """Return full path to log file."""
        return self.logs_dir / filename
    
    def get_metrics_path(self, filename: str) -> Path:
        """Return full path to metrics file."""
        return self.metrics_dir / filename
    
    def save_metrics_json(self, metrics: Dict[str, Any], filename: str = "metrics.json"):
        """
        Save metrics dictionary to JSON file.
        
        Args:
            metrics: Dictionary of metrics to save
            filename: Name for the metrics file
        """
        metrics_path = self.metrics_dir / filename
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"[ExperimentManager] Saved metrics to {metrics_path}")
    
    def __repr__(self):
        return f"ExperimentManager(run_name='{self.run_name}', path='{self.run_path}')"
