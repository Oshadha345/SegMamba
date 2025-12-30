"""
Test Script: Single Forward Pass with SegMamba
- Validates model architecture
- Computes parameters and FLOPs
- Tests forward pass
- Logs to TensorBoard
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import time
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "models"))

# Import SegMamba directly from segmamba module
from segmamba import SegMamba
from config import Config


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_flops_manual(model, input_shape):
    """
    Estimate FLOPs for 3D convolutions and linear layers.
    Note: This is a rough estimate for Conv3D operations.
    """
    total_flops = 0
    
    def conv3d_flops(module, input_size, output_size):
        """Estimate FLOPs for a 3D convolution."""
        batch_size, in_channels, in_d, in_h, in_w = input_size
        out_channels, _, out_d, out_h, out_w = output_size
        
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2]
        kernel_ops *= in_channels // module.groups
        
        # FLOPs = kernel_ops * output_elements
        output_elements = out_channels * out_d * out_h * out_w
        flops = kernel_ops * output_elements * batch_size
        return flops
    
    def linear_flops(module, input_size):
        """Estimate FLOPs for linear layer."""
        return module.in_features * module.out_features
    
    # Register hooks to track activations
    hooks = []
    flop_dict = {}
    
    def hook_fn(module, input, output):
        module_name = str(type(module).__name__)
        
        if isinstance(module, nn.Conv3d):
            input_size = input[0].shape
            output_size = output.shape
            flops = conv3d_flops(module, input_size, output_size)
            if module_name not in flop_dict:
                flop_dict[module_name] = 0
            flop_dict[module_name] += flops
        
        elif isinstance(module, nn.Linear):
            batch_size = input[0].shape[0]
            flops = linear_flops(module, input[0].shape) * batch_size
            if module_name not in flop_dict:
                flop_dict[module_name] = 0
            flop_dict[module_name] += flops
    
    # Register hooks
    for module in model.modules():
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Run forward pass
    with torch.no_grad():
        dummy_input = torch.randn(input_shape)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
            model = model.cuda()
        _ = model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Sum all FLOPs
    total_flops = sum(flop_dict.values())
    
    return total_flops, flop_dict


def format_number(num):
    """Format large numbers for readability."""
    if num >= 1e12:
        return f"{num/1e12:.2f}T"
    elif num >= 1e9:
        return f"{num/1e9:.2f}G"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"


def test_forward_pass():
    """Run a complete test of the SegMamba model."""
    
    print("=" * 80)
    print("SegMamba Forward Pass Test")
    print("=" * 80)
    print()
    
    # Setup TensorBoard
    log_dir = Path("/storage2/CV_Irradiance/VMamba/BRTM/runs/forward_pass_test")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"📊 TensorBoard logs: {log_dir}")
    print(f"   View with: tensorboard --logdir={log_dir}")
    print()
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Model configuration
    print("📐 Model Configuration:")
    print(f"   Input Channels: {Config.IN_CHANNELS}")
    print(f"   Output Classes: {Config.NUM_CLASSES}")
    print(f"   Base Channels: {Config.BASE_CHANNELS}")
    print(f"   Encoder Depths: {Config.ENCODER_DEPTHS}")
    print(f"   Patch Size: {Config.PATCH_SIZE}")
    print()
    
    # Initialize model
    print("🔨 Building SegMamba model...")
    try:
        model = SegMamba(
            in_channels=Config.IN_CHANNELS,
            num_classes=Config.NUM_CLASSES,
            base_channels=Config.BASE_CHANNELS,
            encoder_depths=Config.ENCODER_DEPTHS,
            use_checkpoint=Config.USE_CHECKPOINT
        )
        model = model.to(device)
        print("✅ Model built successfully!")
    except Exception as e:
        print(f"❌ Error building model: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        writer.close()
        return
    print()
    
    # Count parameters
    print("📊 Model Statistics:")
    total_params, trainable_params = count_parameters(model)
    print(f"   Total Parameters: {format_number(total_params)} ({total_params:,})")
    print(f"   Trainable Parameters: {format_number(trainable_params)} ({trainable_params:,})")
    
    # Log to TensorBoard
    writer.add_text("Model/Total_Parameters", f"{format_number(total_params)} ({total_params:,})")
    writer.add_text("Model/Trainable_Parameters", f"{format_number(trainable_params)} ({trainable_params:,})")
    print()
    
    # Estimate FLOPs
    print("⚙️  Computing FLOPs (this may take a moment)...")
    input_shape = (1, Config.IN_CHANNELS, *Config.PATCH_SIZE)
    try:
        model_cpu = SegMamba(
            in_channels=Config.IN_CHANNELS,
            num_classes=Config.NUM_CLASSES,
            base_channels=Config.BASE_CHANNELS,
            encoder_depths=Config.ENCODER_DEPTHS,
            use_checkpoint=False
        )
        total_flops, flop_dict = compute_flops_manual(model_cpu, input_shape)
        print(f"   Estimated FLOPs: {format_number(total_flops)} ({total_flops:,})")
        
        # Log FLOPs breakdown
        print(f"\n   FLOPs Breakdown:")
        for layer_type, flops in sorted(flop_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"     {layer_type}: {format_number(flops)}")
        
        writer.add_text("Model/Total_FLOPs", f"{format_number(total_flops)} ({total_flops:,})")
        del model_cpu
    except Exception as e:
        print(f"   ⚠️  Could not compute FLOPs: {e}")
    print()
    
    # Create dummy input
    print("🔄 Creating test input tensor...")
    batch_size = 1
    dummy_input = torch.randn(batch_size, Config.IN_CHANNELS, *Config.PATCH_SIZE).to(device)
    print(f"   Input shape: {list(dummy_input.shape)}")
    print(f"   Input memory: {dummy_input.element_size() * dummy_input.nelement() / 1e6:.2f} MB")
    print()
    
    # Forward pass
    print("▶️  Running forward pass...")
    model.eval()
    
    try:
        with torch.no_grad():
            # Warm-up
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            output = model(dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            inference_time = time.time() - start_time
        
        print("✅ Forward pass successful!")
        print(f"   Output shape: {list(output.shape)}")
        print(f"   Output memory: {output.element_size() * output.nelement() / 1e6:.2f} MB")
        print(f"   Inference time: {inference_time*1000:.2f} ms")
        
        # Log inference metrics
        writer.add_scalar("Inference/Time_ms", inference_time * 1000)
        writer.add_text("Model/Output_Shape", str(list(output.shape)))
        
        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(device) / 1e9
            memory_reserved = torch.cuda.memory_reserved(device) / 1e9
            max_memory = torch.cuda.max_memory_allocated(device) / 1e9
            
            print(f"\n💾 GPU Memory Usage:")
            print(f"   Allocated: {memory_allocated:.2f} GB")
            print(f"   Reserved: {memory_reserved:.2f} GB")
            print(f"   Peak: {max_memory:.2f} GB")
            
            writer.add_scalar("Memory/Allocated_GB", memory_allocated)
            writer.add_scalar("Memory/Reserved_GB", memory_reserved)
            writer.add_scalar("Memory/Peak_GB", max_memory)
        
        # Output statistics
        print(f"\n📈 Output Statistics:")
        print(f"   Min: {output.min().item():.4f}")
        print(f"   Max: {output.max().item():.4f}")
        print(f"   Mean: {output.mean().item():.4f}")
        print(f"   Std: {output.std().item():.4f}")
        
        writer.add_histogram("Output/Logits", output, 0)
        writer.add_scalar("Output/Min", output.min().item())
        writer.add_scalar("Output/Max", output.max().item())
        writer.add_scalar("Output/Mean", output.mean().item())
        writer.add_scalar("Output/Std", output.std().item())
        
        # Test with softmax
        probs = torch.softmax(output, dim=1)
        print(f"\n🎲 Probability Distribution (after softmax):")
        for i in range(Config.NUM_CLASSES):
            class_prob = probs[0, i].mean().item()
            print(f"   Class {i}: {class_prob:.4f}")
            writer.add_scalar(f"Output/Class_{i}_Prob", class_prob)
        
        print()
        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        writer.add_text("Error", str(e))
    
    finally:
        writer.close()
        print(f"\n📊 TensorBoard logs saved to: {log_dir}")
        print(f"   View with: tensorboard --logdir={log_dir}")


if __name__ == "__main__":
    test_forward_pass()
