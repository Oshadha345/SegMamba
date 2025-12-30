"""
Metrics for Medical Image Segmentation
Implements Dice Score and other evaluation metrics for BraTS challenge.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List


def compute_dice_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-5,
    per_class: bool = True
) -> torch.Tensor:
    """
    Compute Dice Score (F1-score for segmentation).
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    
    Args:
        predictions: Predicted segmentation (B, C, D, H, W) after softmax
        targets: Ground truth one-hot encoded (B, C, D, H, W)
        smooth: Smoothing factor to avoid division by zero
        per_class: If True, returns dice per class; else returns mean
    
    Returns:
        Dice score tensor (C,) if per_class else scalar
    """
    # Flatten spatial dimensions
    preds_flat = predictions.contiguous().view(predictions.shape[0], predictions.shape[1], -1)
    targets_flat = targets.contiguous().view(targets.shape[0], targets.shape[1], -1)
    
    # Compute intersection and union
    intersection = (preds_flat * targets_flat).sum(dim=2)  # (B, C)
    union = preds_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (B, C)
    
    # Dice score
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # Average over batch
    dice = dice.mean(dim=0)  # (C,)
    
    if per_class:
        return dice
    else:
        return dice.mean()


class DiceMetric:
    """
    Accumulator for Dice scores across batches.
    """
    
    def __init__(self, num_classes: int = 4, include_background: bool = True):
        """
        Args:
            num_classes: Number of segmentation classes
            include_background: Whether to include background in average
        """
        self.num_classes = num_classes
        self.include_background = include_background
        self.reset()
    
    def reset(self):
        """Reset accumulated scores."""
        self.dice_scores = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update with a new batch.
        
        Args:
            predictions: Predicted logits (B, C, D, H, W)
            targets: Ground truth labels (B, D, H, W) or one-hot (B, C, D, H, W)
        """
        # Convert predictions to probabilities
        preds_prob = torch.softmax(predictions, dim=1)
        
        # Convert targets to one-hot if needed
        if targets.ndim == 4:
            targets_onehot = torch.nn.functional.one_hot(
                targets.long(), 
                num_classes=self.num_classes
            ).permute(0, 4, 1, 2, 3).float()
        else:
            targets_onehot = targets
        
        # Compute dice per class
        dice = compute_dice_score(preds_prob, targets_onehot, per_class=True)
        self.dice_scores.append(dice.cpu().numpy())
    
    def compute(self) -> dict:
        """
        Compute final metrics.
        
        Returns:
            Dictionary with mean dice and per-class dice scores
        """
        if len(self.dice_scores) == 0:
            return {'mean_dice': 0.0}
        
        # Average across all batches
        dice_per_class = np.mean(self.dice_scores, axis=0)
        
        # Compute mean dice
        if self.include_background:
            mean_dice = np.mean(dice_per_class)
        else:
            mean_dice = np.mean(dice_per_class[1:])  # Exclude background
        
        result = {
            'mean_dice': float(mean_dice),
            'dice_per_class': dice_per_class.tolist()
        }
        
        return result


class DiceCELoss(nn.Module):
    """
    Combined Dice Loss and Cross Entropy Loss.
    
    This is the gold standard for medical image segmentation.
    Combines the region-based Dice loss with the pixel-wise Cross Entropy.
    """
    
    def __init__(
        self, 
        num_classes: int = 4,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        smooth: float = 1e-5,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Args:
            num_classes: Number of segmentation classes
            dice_weight: Weight for Dice loss component
            ce_weight: Weight for Cross Entropy component
            smooth: Smoothing factor for Dice loss
            class_weights: Optional weights for each class in CE loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def dice_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss (1 - Dice score).
        
        Args:
            predictions: Predicted logits (B, C, D, H, W)
            targets: Ground truth one-hot encoded (B, C, D, H, W)
        
        Returns:
            Dice loss scalar
        """
        # Apply softmax to get probabilities
        preds_prob = torch.softmax(predictions, dim=1)
        
        # Compute Dice score
        dice_score = compute_dice_score(preds_prob, targets, smooth=self.smooth, per_class=False)
        
        # Return Dice loss
        return 1.0 - dice_score
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            predictions: Predicted logits (B, C, D, H, W)
            targets: Ground truth labels (B, D, H, W)
        
        Returns:
            Combined loss scalar
        """
        # Convert targets to one-hot for Dice loss
        targets_onehot = torch.nn.functional.one_hot(
            targets.long(), 
            num_classes=self.num_classes
        ).permute(0, 4, 1, 2, 3).float()
        
        # Compute losses
        dice_loss = self.dice_loss(predictions, targets_onehot)
        ce_loss = self.ce_loss(predictions, targets.long())
        
        # Combined loss
        total_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        
        return total_loss
