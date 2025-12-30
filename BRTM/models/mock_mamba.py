"""
Mock Mamba Implementation for Testing
Used when mamba-ssm is unavailable due to CUDA compatibility issues.
"""

import torch
import torch.nn as nn


class Mamba(nn.Module):
    """
    Mock Mamba implementation using standard attention mechanism.
    This is for TESTING ONLY when real mamba-ssm cannot be installed.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        d_inner = int(self.expand * self.d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        
        # Depthwise convolution (simulating state-space conv)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        )
        
        # State space parameters (simplified)
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_state, d_inner, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        
        print("⚠️  WARNING: Using MOCK Mamba implementation for testing!")
        print("   This is NOT the real state-space model.")
        print("   Install mamba-ssm for production use.")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, L, D) tensor
            
        Returns:
            (B, L, D) tensor
        """
        B, L, D = x.shape
        
        # Input projection
        x_and_res = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_gate, res = x_and_res.chunk(2, dim=-1)  # Each: (B, L, d_inner)
        
        # Conv1D (operates on sequence dimension)
        x_conv = x_gate.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[..., :L]  # Truncate extra padding
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        
        # Activation
        x_conv = nn.functional.silu(x_conv)
        
        # "State space" processing (simplified)
        ssm_input = self.x_proj(x_conv)  # (B, L, 2 * d_state)
        ssm_input = ssm_input.mean(dim=1, keepdim=True)  # Simple aggregation
        dt = self.dt_proj(ssm_input[:, :, :self.d_state])  # (B, 1, d_inner)
        
        # Apply gating
        y = x_conv * torch.sigmoid(dt)
        
        # Residual
        y = y * nn.functional.silu(res)
        
        # Output projection
        output = self.out_proj(y)
        
        return output


__all__ = ['Mamba']
