"""
SegMamba: Hybrid State-Space U-Net for 3D Medical Image Segmentation

A production-grade implementation for BraTS-style brain tumor segmentation.
Combines Conv3D efficiency with Mamba/Swin attention for global context.

Author: Elite Competitive Data Scientist
Date: December 2025
"""

__version__ = "1.0.0"
__author__ = "Competitive ML Engineer"

from .models import SegMamba
from .config import Config
from .utils import ExperimentManager

__all__ = ['SegMamba', 'Config', 'ExperimentManager']
