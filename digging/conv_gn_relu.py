#!/usr/bin/env python3
"""
Debug the Conv -> GroupNorm interaction that's causing differences between backends.
"""

import os
import numpy as np
from tinygrad import Tensor, nn, dtypes
from tinygrad.nn.state import safe_save

backend_name = "webgpu" if os.environ.get("WEBGPU") else "metal"
output_file = f"conv_groupnorm_debug_{backend_name}.safetensors"

print(f"Debugging Conv→GroupNorm on {backend_name.upper()} backend")
print("=" * 60)

# Create test input
np.random.seed(42)
test_shape = (1, 30, 64, 64, 64)
test_data = np.random.randn(*test_shape).astype(np.float32) * 10.0

input_tensor = Tensor(test_data, dtype=dtypes.float)

# Create layers
conv = nn.Conv2d(30, 30, kernel_size=[3, 3, 3], padding=1, bias=False)
conv.weight.assign(Tensor.randn(30, 30, 3, 3, 3) * 0.1).realize()

groupnorm = nn.GroupNorm(
    num_groups=30,
    num_channels=30,
    affine=False
)

print("Testing Conv → GroupNorm pipeline...")
print("-" * 40)

# Step 1: Just Conv
conv_output = conv(input_tensor).realize()
conv_np = conv_output.numpy()

print(f"\n1. Conv output:")
print(f"   Shape: {conv_np.shape}")
print(f"   Min: {conv_np.min():.6f}, Max: {conv_np.max():.6f}")
print(f"   Mean: {conv_np.mean():.6f}, Std: {conv_np.std():.6f}")
print(f"   NaN count: {np.isnan(conv_np).sum()}")
print(f"   Zero count: {(conv_np == 0).sum()}")
print(f"   Near-zero count (|x| < 1e-6): {(np.abs(conv_np) < 1e-6).sum()}")

# Check for extreme values
extreme_mask = np.abs(conv_np) > 100
if extreme_mask.any():
    print(f"   ⚠️ Extreme values (|x| > 100): {extreme_mask.sum()} locations")
    max_loc = np.unravel_index(np.abs(conv_np).argmax(), conv_np.shape)
    print(f"   Max absolute value: {conv_np[max_loc]:.6f} at {max_loc}")

# Step 2: GroupNorm on Conv output
groupnorm_output = groupnorm(conv_output).realize()
gn_np = groupnorm_output.numpy()

print(f"\n2. GroupNorm(Conv) output:")
print(f"   Min: {gn_np.min():.6f}, Max: {gn_np.max():.6f}")
print(f"   Mean: {gn_np.mean():.9f}, Std: {gn_np.std():.6f}")
print(f"   NaN count: {np.isnan(gn_np).sum()}")
print(f"   Zero count: {(gn_np == 0).sum()}")

# Step 3: ReLU on top
relu_output = groupnorm_output.relu().realize()
relu_np = relu_output.numpy()

print(f"\n3. ReLU(GroupNorm(Conv)) output:")
print(f"   Min: {relu_np.min():.6f}, Max: {relu_np.max():.6f}")
print(f"   Mean: {relu_np.mean():.6f}")
print(f"   Zero count: {(relu_np == 0).sum()}")

# Step 4: Test with different conv weight scales
print("\n" + "=" * 60)
print("Testing with different Conv weight scales...")
print("-" * 40)

scales = [0.01, 0.1, 0.5, 1.0, 2.0]
results = {}

for scale in scales:
    # Create new conv with different weight scale
    conv_test = nn.Conv2d(30, 30, kernel_size=[3, 3, 3], padding=1, bias=False)
    conv_test.weight.assign(Tensor.randn(30, 30, 3, 3, 3) * scale).realize()
    
    # Run pipeline
    x = conv_test(input_tensor).realize()
    x = groupnorm(x).realize()
    x = x.relu().realize()
    
    x_np = x.numpy()
    
    # Check for issues
    zero_count = (x_np == 0).sum()
    total_elements = x_np.size
    zero_percent = 100 * zero_count / total_elements
    
    print(f"\nScale {scale:4.2f}:")
    print(f"  Output range: [{x_np.min():.3f}, {x_np.max():.3f}]")
    print(f"  Zeros: {zero_count}/{total_elements} ({zero_percent:.1f}%)")
    print(f"  NaN count: {np.isnan(x_np).sum()}")
    
    results[f"scale_{scale}"] = x

# Step 5: Test intermediate precision
print("\n" + "=" * 60)
print("Testing numerical precision at each step...")
print("-" * 40)

# Run the pipeline step by step, checking precision
conv_out_raw = conv(input_tensor)
conv_out_realized = conv_out_raw.realize()

# Check if realization changes values
print("\nComparing Conv output before/after realize:")
conv_raw_eval = conv_out_raw.numpy()  # This will realize internally
conv_real_eval = conv_out_realized.numpy()
diff = np.abs(conv_raw_eval - conv_real_eval)
print(f"  Max difference: {diff.max():.9f}")
if diff.max() > 1e-6:
    print(f"  ⚠️ Realization affects Conv output values!")

# Test GroupNorm with different input magnitudes
print("\nTesting GroupNorm with scaled inputs:")
for multiplier in [0.1, 1.0, 10.0, 100.0]:
    scaled_input = conv_output * multiplier
    gn_scaled = groupnorm(scaled_input).realize()
    gn_scaled_np = gn_scaled.numpy()
    
    print(f"  Input scale {multiplier:5.1f}x: ", end="")
    print(f"Mean={gn_scaled_np.mean():.9f}, Std={gn_scaled_np.std():.6f}")

# Save all outputs
save_dict = {
    "input": input_tensor,
    "conv_output": conv_output,
    "groupnorm_output": groupnorm_output,
    "relu_output": relu_output,
}

# Add scale test results
for key, tensor in results.items():
    save_dict[key] = tensor

print(f"\n" + "=" * 60)
print(f"Saving outputs to {output_file}...")
safe_save(save_dict, output_file)
print("Done!")
print(f"\nRun on both backends and compare to identify the issue.")
