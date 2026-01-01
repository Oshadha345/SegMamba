"""
Visualization Utilities for SegMamba
Provides functions for sanity checks, training plots, and segmentation overlays.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
from pathlib import Path
from typing import Optional, Tuple, List


def visualize_batch(
    images: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    predictions: Optional[torch.Tensor] = None,
    save_path: Optional[Path] = None,
    slice_idx: Optional[int] = None,
    title: str = "Batch Visualization"
):
    """
    Visualize a batch of 3D medical images (center slice).
    
    Args:
        images: Tensor of shape (B, C, D, H, W) - batch of 3D images
        labels: Optional ground truth labels (B, D, H, W) or (B, C, D, H, W)
        predictions: Optional model predictions (B, C, D, H, W)
        save_path: Path to save the figure
        slice_idx: Which slice to show (if None, uses center slice)
        title: Title for the plot
    """
    # Move to CPU and convert to numpy
    images = images.cpu().numpy()
    if labels is not None:
        labels = labels.cpu().numpy()
    if predictions is not None:
        predictions = predictions.cpu().numpy()
    
    batch_size = images.shape[0]
    num_channels = images.shape[1]
    
    # Determine slice index
    if slice_idx is None:
        slice_idx = images.shape[2] // 2  # Center slice
    
    # Create figure
    num_cols = num_channels + (1 if labels is not None else 0) + (1 if predictions is not None else 0)
    fig, axes = plt.subplots(batch_size, num_cols, figsize=(num_cols * 3, batch_size * 3))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    channel_names = ['T1', 'T1ce', 'T2', 'FLAIR'][:num_channels]
    
    for b in range(batch_size):
        col = 0
        
        # Plot each modality
        for c in range(num_channels):
            ax = axes[b, col]
            img_slice = images[b, c, slice_idx, :, :]
            ax.imshow(img_slice, cmap='gray')
            if b == 0:
                ax.set_title(f'{channel_names[c]}')
            ax.axis('off')
            col += 1
        
        # Plot ground truth
        if labels is not None:
            ax = axes[b, col]
            if labels.ndim == 5:  # One-hot encoded
                label_slice = np.argmax(labels[b, :, slice_idx, :, :], axis=0)
            else:
                label_slice = labels[b, slice_idx, :, :]
            
            ax.imshow(label_slice, cmap='jet', vmin=0, vmax=3)
            if b == 0:
                ax.set_title('Ground Truth')
            ax.axis('off')
            col += 1
        
        # Plot predictions
        if predictions is not None:
            ax = axes[b, col]
            pred_slice = np.argmax(predictions[b, :, slice_idx, :, :], axis=0)
            ax.imshow(pred_slice, cmap='jet', vmin=0, vmax=3)
            if b == 0:
                ax.set_title('Prediction')
            ax.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Visualization] Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_metrics: List[float],
    save_path: Path,
    metric_name: str = "Dice Score"
):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        val_metrics: List of validation metrics per epoch
        save_path: Path to save the figure
        metric_name: Name of the metric being plotted
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Metric curves
    ax2.plot(epochs, val_metrics, 'g-^', label=f'Val {metric_name}', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel(metric_name, fontsize=12)
    ax2.set_title(f'Validation {metric_name}', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Highlight best epoch
    best_epoch = np.argmax(val_metrics) + 1
    best_metric = max(val_metrics)
    ax2.axvline(x=best_epoch, color='k', linestyle='--', alpha=0.5)
    ax2.text(best_epoch, best_metric, f'Best: {best_metric:.4f}\nEpoch {best_epoch}',
             ha='center', va='bottom', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Visualization] Saved training curves to {save_path}")
    plt.close()


def save_segmentation_overlay(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    save_path: Path,
    slice_idx: Optional[int] = None,
    alpha: float = 0.4
):
    """
    Save an overlay of segmentation on the original image.
    
    Args:
        image: 3D image array (D, H, W) or (C, D, H, W)
        ground_truth: 3D segmentation array (D, H, W)
        prediction: 3D segmentation array (D, H, W)
        save_path: Path to save the overlay
        slice_idx: Which slice to visualize
        alpha: Transparency for overlay
    """
    if image.ndim == 4:
        image = image[0]  # Use first channel
    
    if slice_idx is None:
        slice_idx = image.shape[0] // 2
    
    img_slice = image[slice_idx, :, :]
    gt_slice = ground_truth[slice_idx, :, :]
    pred_slice = prediction[slice_idx, :, :]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Ground truth overlay
    axes[1].imshow(img_slice, cmap='gray')
    masked_gt = np.ma.masked_where(gt_slice == 0, gt_slice)
    axes[1].imshow(masked_gt, cmap='jet', alpha=alpha, vmin=0, vmax=3)
    axes[1].set_title('Ground Truth Overlay', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Prediction overlay
    axes[2].imshow(img_slice, cmap='gray')
    masked_pred = np.ma.masked_where(pred_slice == 0, pred_slice)
    axes[2].imshow(masked_pred, cmap='jet', alpha=alpha, vmin=0, vmax=3)
    axes[2].set_title('Prediction Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Visualization] Saved overlay to {save_path}")
    plt.close()


def visualize_3d_volume(
    volume: np.ndarray,
    save_path: Path,
    num_slices: int = 9,
    title: str = "3D Volume Slices"
):
    """
    Visualize multiple slices of a 3D volume.
    
    Args:
        volume: 3D array (D, H, W)
        save_path: Path to save the figure
        num_slices: Number of slices to show
        title: Title for the plot
    """
    depth = volume.shape[0]
    slice_indices = np.linspace(0, depth - 1, num_slices, dtype=int)
    
    rows = int(np.ceil(np.sqrt(num_slices)))
    cols = int(np.ceil(num_slices / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if num_slices > 1 else [axes]
    
    for idx, slice_idx in enumerate(slice_indices):
        axes[idx].imshow(volume[slice_idx, :, :], cmap='gray')
        axes[idx].set_title(f'Slice {slice_idx}', fontsize=10)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(slice_indices), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Visualization] Saved 3D volume visualization to {save_path}")
    plt.close()
