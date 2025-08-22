#!/usr/bin/env python3
"""
Test GroupNorm behavior across backends.
Usage:
    python groupnorm_test.py                    # Run with default backend (Metal)
    WEBGPU=1 python groupnorm_test.py          # Run with WebGPU backend
    
This will save output to groupnorm_metal.safetensors or groupnorm_webgpu.safetensors
"""

import os
import numpy as np
from tinygrad import Tensor, nn, dtypes
from tinygrad.nn.state import safe_save

# Determine backend from environment
backend_name = "webgpu" if os.environ.get("WEBGPU") else "metal"
output_file = f"groupnorm_{backend_name}.safetensors"

print(f"Testing GroupNorm on {backend_name.upper()} backend")
print("=" * 60)

# Create test input with realistic values (similar to your model's intermediate activations)
np.random.seed(42)  # For reproducibility
test_shape = (1, 30, 64, 64, 64)  # Smaller than full size for faster testing
test_data = np.random.randn(*test_shape).astype(np.float32) * 10.0  # Similar magnitude to your data

print(f"Input shape: {test_shape}")
print(f"Input stats: min={test_data.min():.4f}, max={test_data.max():.4f}, mean={test_data.mean():.4f}")

# Create input tensor
input_tensor = Tensor(test_data, dtype=dtypes.float)

# Test GroupNorm
groupnorm = nn.GroupNorm(
    num_groups=30,  # Same as your model
    num_channels=30,
    affine=False    # Same as your model
)

print("\nApplying GroupNorm...")

# Test 1: Single GroupNorm operation (realized)
output_realized = groupnorm(input_tensor).realize()
output_np = output_realized.numpy()

print(f"\nGroupNorm output (realized):")
print(f"  Min: {output_np.min():.6f}")
print(f"  Max: {output_np.max():.6f}") 
print(f"  Mean: {output_np.mean():.9f}")  # Should be very close to 0
print(f"  Std: {output_np.std():.6f}")    # Should be close to 1
print(f"  NaN count: {np.isnan(output_np).sum()}")

# Test 2: GroupNorm + ReLU fusion (common pattern in your model)
print("\nTesting GroupNorm + ReLU fusion...")
fused_output = groupnorm(input_tensor).relu().realize()
fused_np = fused_output.numpy()

print(f"GroupNorm + ReLU output:")
print(f"  Min: {fused_np.min():.6f}")
print(f"  Max: {fused_np.max():.6f}")
print(f"  Mean: {fused_np.mean():.6f}")
print(f"  NaN count: {np.isnan(fused_np).sum()}")

# Test 3: Multiple operations fused (Conv + GroupNorm + ReLU pattern)
print("\nTesting Conv + GroupNorm + ReLU fusion...")
conv = nn.Conv2d(30, 30, kernel_size=[3, 3, 3], padding=1, bias=False)
# Initialize with small weights to avoid explosion
conv.weight.assign(Tensor.randn(30, 30, 3, 3, 3) * 0.1).realize()

# Without fusion (realize after each)
x = conv(input_tensor).realize()
x = groupnorm(x).realize()
x = x.relu().realize()
no_fusion_np = x.numpy()

print(f"Conv + GroupNorm + ReLU (no fusion):")
print(f"  Min: {no_fusion_np.min():.6f}")
print(f"  Max: {no_fusion_np.max():.6f}")
print(f"  Mean: {no_fusion_np.mean():.6f}")
print(f"  NaN count: {np.isnan(no_fusion_np).sum()}")

# With fusion (no intermediate realizes)
fused_x = conv(input_tensor)
fused_x = groupnorm(fused_x)
fused_x = fused_x.relu().realize()
fusion_np = fused_x.numpy()

print(f"Conv + GroupNorm + ReLU (with fusion):")
print(f"  Min: {fusion_np.min():.6f}")
print(f"  Max: {fusion_np.max():.6f}")
print(f"  Mean: {fusion_np.mean():.6f}")
print(f"  NaN count: {np.isnan(fusion_np).sum()}")

# Save outputs for comparison
save_dict = {
    "input": input_tensor,
    "groupnorm_only": output_realized,
    "groupnorm_relu": fused_output,
    "conv_groupnorm_relu_nofusion": x,
    "conv_groupnorm_relu_fusion": fused_x
}

print(f"\nSaving outputs to {output_file}...")
safe_save(save_dict, output_file)
print("Done!")

print("\n" + "=" * 60)
print(f"Results saved to {output_file}")
print("Run with both backends and then use compare_outputs.py to analyze differences")
