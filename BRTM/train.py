"""
SegMamba Training Pipeline
Production-grade training loop with AMP, gradient accumulation, and experiment tracking.

Author: Competitive ML Engineer
Purpose: Train SegMamba for BraTS 3D medical image segmentation
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional
import time
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
from config import Config
from models import SegMamba
from data import create_dataloaders
from utils import (
    ExperimentManager,
    DiceCELoss,
    DiceMetric,
    visualize_batch,
    plot_training_curves
)


class SegMambaTrainer:
    """
    Complete training pipeline for SegMamba.
    
    Features:
        - Automatic Mixed Precision (AMP) for memory efficiency
        - Gradient accumulation for effective larger batch sizes
        - Best model checkpointing based on validation metrics
        - Training curve visualization and logging
        - Sanity check visualization before training
    
    Args:
        config: Configuration object
        experiment_manager: ExperimentManager instance
    """
    
    def __init__(self, config: Config, experiment_manager: ExperimentManager):
        self.config = config
        self.exp_manager = experiment_manager
        
        # Set random seeds for reproducibility
        self._set_seed(config.SEED)
        
        # Initialize model
        print("\n" + "="*70)
        print("Initializing SegMamba Model")
        print("="*70)
        
        self.model = SegMamba(
            in_channels=config.IN_CHANNELS,
            num_classes=config.NUM_CLASSES,
            base_channels=config.BASE_CHANNELS,
            encoder_depths=config.ENCODER_DEPTHS,
            use_checkpoint=config.USE_CHECKPOINT
        ).to(config.DEVICE)
        
        # Initialize loss function
        self.criterion = DiceCELoss(
            num_classes=config.NUM_CLASSES,
            dice_weight=config.DICE_WEIGHT,
            ce_weight=config.CE_WEIGHT
        ).to(config.DEVICE)
        
        # Initialize optimizer
        if config.OPTIMIZER == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.INITIAL_LR,
                weight_decay=config.WEIGHT_DECAY
            )
        elif config.OPTIMIZER == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.INITIAL_LR,
                momentum=0.9,
                weight_decay=config.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")
        
        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize AMP scaler
        self.scaler = GradScaler() if config.USE_AMP else None
        
        # Initialize metrics
        self.dice_metric = DiceMetric(num_classes=config.NUM_CLASSES, include_background=False)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.best_metric = -1.0 if config.METRIC_HIGHER_BETTER else float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        print(f"\n[Trainer] Model has {self.model.count_parameters():,} parameters")
        print(f"[Trainer] Using device: {config.DEVICE}")
        print(f"[Trainer] AMP enabled: {config.USE_AMP}")
        print(f"[Trainer] Gradient accumulation steps: {config.ACCUMULATION_STEPS}")
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        if self.config.DETERMINISTIC:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.LR_SCHEDULER == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.NUM_EPOCHS,
                eta_min=self.config.MIN_LR
            )
        elif self.config.LR_SCHEDULER == "poly":
            scheduler = optim.lr_scheduler.PolynomialLR(
                self.optimizer,
                total_iters=self.config.NUM_EPOCHS,
                power=0.9
            )
        elif self.config.LR_SCHEDULER == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=50,
                gamma=0.5
            )
        else:
            scheduler = None
        
        return scheduler
    
    def sanity_check(self, train_loader):
        """
        Perform sanity check by visualizing first batch.
        
        Args:
            train_loader: Training data loader
        """
        print("\n" + "="*70)
        print("Running Sanity Check")
        print("="*70)
        
        # Get first batch
        batch = next(iter(train_loader))
        images = batch['image']
        labels = batch['label']
        
        print(f"[Sanity Check] Batch image shape: {images.shape}")
        print(f"[Sanity Check] Batch label shape: {labels.shape}")
        print(f"[Sanity Check] Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"[Sanity Check] Label unique values: {torch.unique(labels).tolist()}")
        
        # Visualize
        save_path = self.exp_manager.get_plot_path("sanity_check_batch.png")
        visualize_batch(
            images=images,
            labels=labels,
            save_path=save_path,
            title="Sanity Check: Training Batch"
        )
        
        print(f"[Sanity Check] Visualization saved to {save_path}")
        
        # Test forward pass
        print("\n[Sanity Check] Testing forward pass...")
        self.model.eval()
        with torch.no_grad():
            images = images.to(self.config.DEVICE)
            output = self.model(images)
            print(f"[Sanity Check] Model output shape: {output.shape}")
            print(f"[Sanity Check] Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        self.model.train()
        print("[Sanity Check] ✓ Passed\n")
    
    def train_epoch(self, train_loader, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.NUM_EPOCHS}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.config.DEVICE)
            labels = batch['label'].to(self.config.DEVICE)
            
            # Forward pass with AMP
            if self.config.USE_AMP:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.config.ACCUMULATION_STEPS
                
                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.config.ACCUMULATION_STEPS
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.ACCUMULATION_STEPS == 0:
                if self.config.USE_AMP:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += loss.item() * self.config.ACCUMULATION_STEPS
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item() * self.config.ACCUMULATION_STEPS:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        avg_loss = epoch_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader, epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.dice_metric.reset()
        
        epoch_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(val_loader, desc=f"Validation")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.config.DEVICE)
            labels = batch['label'].to(self.config.DEVICE)
            
            # Forward pass
            if self.config.USE_AMP:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            self.dice_metric.update(outputs, labels)
            
            # Save sample predictions (first batch only)
            if batch_idx == 0 and self.config.SAVE_TRAIN_IMAGES:
                save_path = self.exp_manager.get_plot_path(f"val_predictions_epoch_{epoch:03d}.png")
                predictions = torch.softmax(outputs, dim=1)
                visualize_batch(
                    images=images,
                    labels=labels,
                    predictions=predictions,
                    save_path=save_path,
                    title=f"Validation Predictions - Epoch {epoch}"
                )
        
        # Compute final metrics
        avg_loss = epoch_loss / num_batches
        dice_results = self.dice_metric.compute()
        
        results = {
            'loss': avg_loss,
            'mean_dice': dice_results['mean_dice'],
            'dice_per_class': dice_results['dice_per_class']
        }
        
        return results
    
    def train(self, train_loader, val_loader):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print("\n" + "="*70)
        print("Starting Training")
        print("="*70)
        
        # Sanity check before training
        self.sanity_check(train_loader)
        
        start_time = time.time()
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            print(f"\nEpoch {epoch}/{self.config.NUM_EPOCHS}")
            print("-" * 70)
            
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            print(f"[Train] Loss: {train_loss:.4f}")
            
            # Validation
            if epoch % self.config.VAL_INTERVAL == 0:
                val_results = self.validate(val_loader, epoch)
                self.val_losses.append(val_results['loss'])
                self.val_metrics.append(val_results['mean_dice'])
                
                print(f"[Val] Loss: {val_results['loss']:.4f}")
                print(f"[Val] Mean Dice: {val_results['mean_dice']:.4f}")
                print(f"[Val] Dice per class: {[f'{d:.4f}' for d in val_results['dice_per_class']]}")
                
                # Check if best model
                current_metric = val_results[self.config.METRIC_NAME]
                is_best = False
                
                if self.config.METRIC_HIGHER_BETTER:
                    if current_metric > self.best_metric:
                        self.best_metric = current_metric
                        self.best_epoch = epoch
                        is_best = True
                        self.epochs_without_improvement = 0
                    else:
                        self.epochs_without_improvement += self.config.VAL_INTERVAL
                else:
                    if current_metric < self.best_metric:
                        self.best_metric = current_metric
                        self.best_epoch = epoch
                        is_best = True
                        self.epochs_without_improvement = 0
                    else:
                        self.epochs_without_improvement += self.config.VAL_INTERVAL
                
                # Save checkpoint
                if is_best:
                    print(f"[Checkpoint] New best model! {self.config.METRIC_NAME}: {current_metric:.4f}")
                
                self.exp_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=val_results,
                    filename=f"checkpoint_epoch_{epoch:03d}.pth",
                    is_best=is_best
                )
                
                # Plot training curves
                if len(self.val_losses) > 1:
                    plot_path = self.exp_manager.get_plot_path("training_curves.png")
                    plot_training_curves(
                        train_losses=self.train_losses,
                        val_losses=self.val_losses,
                        val_metrics=self.val_metrics,
                        save_path=plot_path,
                        metric_name=self.config.METRIC_NAME
                    )
                
                # Early stopping
                if self.epochs_without_improvement >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"\n[Early Stopping] No improvement for {self.config.EARLY_STOPPING_PATIENCE} epochs")
                    break
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "="*70)
        print("Training Complete")
        print("="*70)
        print(f"Total time: {elapsed_time / 3600:.2f} hours")
        print(f"Best {self.config.METRIC_NAME}: {self.best_metric:.4f} at epoch {self.best_epoch}")
        
        # Save final metrics
        final_metrics = {
            'best_metric': float(self.best_metric),
            'best_epoch': int(self.best_epoch),
            'total_epochs': epoch,
            'training_time_hours': elapsed_time / 3600,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }
        
        self.exp_manager.save_metrics_json(final_metrics, "final_metrics.json")
        
        print(f"\n[Results] All artifacts saved to: {self.exp_manager.run_path}")


def main():
    """
    Main training function.
    """
    # Load configuration
    config = Config()
    config.print_config()
    
    # Initialize experiment manager
    exp_manager = ExperimentManager(
        run_name=config.RUN_NAME,
        base_path=str(config.RESULTS_BASE_PATH),
        overwrite=False
    )
    
    # Save configuration
    exp_manager.save_config(config.get_config_dict())
    
    # Create data loaders
    print("\n" + "="*70)
    print("Loading Data")
    print("="*70)
    
    train_loader, val_loader = create_dataloaders(
        train_data_path=str(config.TRAIN_DATA_PATH),
        val_data_path=str(config.VAL_DATA_PATH),
        batch_size=config.BATCH_SIZE,
        patch_size=config.PATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Initialize trainer
    trainer = SegMambaTrainer(config, exp_manager)
    
    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
