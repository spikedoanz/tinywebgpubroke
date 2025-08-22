#!/usr/bin/env python3
"""
Test Conv→GroupNorm→ReLU with parameters matching the actual model.
The bug might be specific to certain sizes, channel counts, or accumulation effects.
"""

import os
import numpy as np
from tinygrad import Tensor, nn, dtypes, Device

print("=" * 60)
print("TESTING WITH MODEL-SPECIFIC PARAMETERS")
print("=" * 60)

# Test different configurations
test_configs = [
    # (channels, size, description)
    (30, 64, "Model channels (30), smaller size"),
    (30, 128, "Model channels (30), medium size"),
    (30, 256, "Model channels (30), full size (will be slow!)"),
    (8, 64, "Fewer channels (8), smaller size"),
    (32, 64, "Power-of-2 channels (32), smaller size"),
]

def test_configuration(channels, size, description):
    """Test a specific configuration"""
    print(f"\nTesting: {description}")
    print(f"  Channels: {channels}, Size: {size}x{size}x{size}")
    print("-" * 50)
    
    # Create test data
    np.random.seed(42)
    input_shape = (1, channels, size, size, size)
    test_data = np.random.randn(*input_shape).astype(np.float32) * 10.0
    conv_weights = np.random.randn(channels, channels, 3, 3, 3).astype(np.float32) * 0.1
    
    results = {}
    
    for backend in ["METAL", "WEBGPU"]:
        if backend == "METAL":
            os.environ.pop('WEBGPU', None)
            Device.DEFAULT = "METAL"
        else:
            os.environ['WEBGPU'] = '1'
            Device.DEFAULT = "WEBGPU"
        
        # Create layers
        input_tensor = Tensor(test_data, dtype=dtypes.float)
        conv = nn.Conv2d(channels, channels, kernel_size=[3, 3, 3], padding=1, bias=False)
        conv.weight.assign(Tensor(conv_weights, dtype=dtypes.float)).realize()
        groupnorm = nn.GroupNorm(num_groups=channels, num_channels=channels, affine=False)
        
        # Run pipeline
        x = conv(input_tensor)
        x = groupnorm(x)
        x = x.relu()
        output = x.realize()
        
        results[backend] = output.numpy()
        
        print(f"  {backend:6s}: Min={results[backend].min():.3f}, "
              f"Max={results[backend].max():.3f}, "
              f"Mean={results[backend].mean():.3f}")
    
    # Compare
    diff = np.abs(results["METAL"] - results["WEBGPU"])
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    # Check for the specific pattern from original issue
    metal_positive = results["METAL"] > 0.5
    webgpu_zero_ish = np.abs(results["WEBGPU"]) < 0.1
    wrong_pattern = metal_positive & webgpu_zero_ish
    
    status = "🟢"
    if max_diff > 1.0:
        status = "🔴"
    elif max_diff > 0.1:
        status = "🟡"
    
    print(f"  {status} Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
    
    if wrong_pattern.any():
        print(f"  ⚠️  Found {wrong_pattern.sum()} locations where Metal>0.5 but WebGPU≈0")
    
    return max_diff

# Run tests
print("\n" + "=" * 60)
max_diffs = {}

for channels, size, description in test_configs:
    if size == 256:
        response = input(f"\nTest with size 256³ will be slow. Continue? (y/n): ")
        if response.lower() != 'y':
            print("  Skipped")
            continue
    
    max_diff = test_configuration(channels, size, description)
    max_diffs[description] = max_diff

# Test accumulation effect (multiple layers)
print("\n" + "=" * 60)
print("TESTING ACCUMULATION OVER MULTIPLE LAYERS")
print("=" * 60)

np.random.seed(42)
channels = 30
size = 64
input_shape = (1, channels, size, size, size)
test_data = np.random.randn(*input_shape).astype(np.float32) * 10.0

for backend in ["METAL", "WEBGPU"]:
    if backend == "METAL":
        os.environ.pop('WEBGPU', None)
        Device.DEFAULT = "METAL"
    else:
        os.environ['WEBGPU'] = '1'
        Device.DEFAULT = "WEBGPU"
    
    print(f"\n{backend} - Running 5 Conv→GN→ReLU blocks:")
    
    x = Tensor(test_data, dtype=dtypes.float)
    
    for i in range(5):
        # Create new conv for each layer
        conv = nn.Conv2d(channels, channels, kernel_size=[3, 3, 3], padding=1, bias=False)
        conv.weight.assign(Tensor.randn(channels, channels, 3, 3, 3) * 0.1).realize()
        groupnorm = nn.GroupNorm(num_groups=channels, num_channels=channels, affine=False)
        
        # Apply block
        x = conv(x)
        x = groupnorm(x)
        x = x.relu()
        x = x.realize()  # Force realization after each block
        
        x_np = x.numpy()
        print(f"  After block {i+1}: Min={x_np.min():.3f}, Max={x_np.max():.3f}, "
              f"Mean={x_np.mean():.3f}, NaN={np.isnan(x_np).sum()}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if any(d > 1.0 for d in max_diffs.values()):
    print("🔴 BUG DETECTED in certain configurations!")
    for desc, diff in max_diffs.items():
        if diff > 1.0:
            print(f"   - {desc}: max diff = {diff:.6f}")
else:
    print("🟡 No major differences found in these tests.")
    print("   The bug might require:")
    print("   - Even larger tensor sizes (256³)")
    print("   - More layers (your model has 17)")
    print("   - Specific weight values from your pretrained model")
    print("   - Different memory/cache conditions")
