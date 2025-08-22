from tinygrad import Tensor, nn, dtypes
from tinygrad.nn.state import torch_load, load_state_dict
import json
import nibabel as nib
import numpy as np
import time

print("=" * 80)
print("STARTING MESHNET DEBUG SCRIPT - LINEAR CONTROL FLOW")
print("=" * 80)

# ============================================================================
# STEP 1: Define all paths and load configuration
# ============================================================================
print("\n[STEP 1] Loading paths and configuration...")
nifti_path = "t1_crop.nii.gz"
model_path = "prime.pth"
config_path = "prime.json"
output_path = "segmentation_output.nii.gz"

with open(config_path, "r") as f:
    config = json.load(f)

in_channels = config["in_channels"]
channels = config["channels"]
n_classes = config["out_channels"]
dropout_p = config["dropout_p"]
bnorm = config["bnorm"]
gelu = config["gelu"]

print(f"  Config loaded: in_channels={in_channels}, channels={channels}, n_classes={n_classes}")
print(f"  Dropout: {dropout_p}, Batch norm: {bnorm}, GELU: {gelu}")

# ============================================================================
# STEP 2: Update config channels
# ============================================================================
print("\n[STEP 2] Updating config channels...")
# input layer
config["layers"][0]["in_channels"] = in_channels
config["layers"][0]["out_channels"] = channels
# output layer
config["layers"][-1]["in_channels"] = channels
config["layers"][-1]["out_channels"] = n_classes
# hidden layers
for layer in config["layers"][1:-1]:
    layer["in_channels"] = layer["out_channels"] = channels

print(f"  Updated {len(config['layers'])} layers")

# ============================================================================
# STEP 3: Build model layers manually (unrolled)
# ============================================================================
print("\n[STEP 3] Building model layers...")
model_layers = []

# Process all layers except the last one
for i, block_kwargs in enumerate(config["layers"][:-1]):
    print(f"  Building layer {i}: in={block_kwargs['in_channels']}, out={block_kwargs['out_channels']}")
    
    # Conv layer
    conv = nn.Conv2d(
        in_channels=block_kwargs["in_channels"],
        out_channels=block_kwargs["out_channels"],
        kernel_size=[block_kwargs["kernel_size"]] * 3,
        padding=block_kwargs["padding"],
        stride=block_kwargs["stride"],
        dilation=block_kwargs["dilation"],
        bias=False
    )
    model_layers.append(conv)
    
    # Batch norm if enabled
    if bnorm:
        bn = nn.GroupNorm(
            num_groups=block_kwargs["out_channels"],
            num_channels=block_kwargs["out_channels"],
            affine=False
        )
        model_layers.append(bn)
    
    # Activation (stored as string identifier)
    if gelu:
        model_layers.append("gelu")
    else:
        model_layers.append("relu")
    
    # Dropout if enabled
    if dropout_p > 0:
        model_layers.append(("dropout", dropout_p))

# Add final layer
last_config = config["layers"][-1]
print(f"  Building final layer: in={last_config['in_channels']}, out={last_config['out_channels']}")
final_conv = nn.Conv2d(
    last_config["in_channels"],
    last_config["out_channels"],
    kernel_size=[last_config["kernel_size"]] * 3,
    padding=last_config["padding"],
    stride=last_config["stride"],
    dilation=last_config["dilation"],
    bias=False
)
model_layers.append(final_conv)

print(f"  Total layers built: {len(model_layers)}")

# ============================================================================
# STEP 4: Load pretrained weights
# ============================================================================
print("\n[STEP 4] Loading pretrained weights...")
torch_state_dict = torch_load(model_path)
print(f"  Loaded {len(torch_state_dict)} weight tensors from {model_path}")

# Create a simple model object to hold the layers for state dict loading
class SimpleModel:
    def __init__(self, layers):
        self.model = [l for l in layers if not isinstance(l, str) and not isinstance(l, tuple)]

model_obj = SimpleModel(model_layers)
tiny_state_dict = nn.state.get_state_dict(model_obj)

# Convert keys
torch_keys = list(torch_state_dict.keys())
tiny_keys = list(tiny_state_dict.keys())
new_dict = {}
for (torch_key, tiny_key) in zip(torch_keys, tiny_keys):
    new_dict[tiny_key] = torch_state_dict[torch_key]
    print(f"  Mapping {torch_key} -> {tiny_key}")

load_state_dict(model_obj, new_dict, strict=True)
print("  Weights loaded successfully")

# ============================================================================
# STEP 5: Load NIfTI input data
# ============================================================================
print(f"\n[STEP 5] Loading NIfTI file: {nifti_path}")
img = nib.load(nifti_path)
volume_data = img.get_fdata().astype(np.float32)  # Use float32 instead of int32
affine = img.affine

print(f"  Volume shape: {volume_data.shape}")
print(f"  Data type: {volume_data.dtype}")
print(f"  Min value: {volume_data.min()}")
print(f"  Max value: {volume_data.max()}")
print(f"  Mean value: {volume_data.mean()}")
print(f"  NaN count: {np.isnan(volume_data).sum()}")

# ============================================================================
# STEP 6: Create input tensor
# ============================================================================
print("\n[STEP 6] Creating input tensor...")
input_tensor = Tensor(volume_data, dtype=dtypes.float).rearrange("... -> 1 1 ...")
print(f"  Input tensor shape: {input_tensor.shape}")
print(f"  Input tensor stats: min={input_tensor.min().item():.4f}, max={input_tensor.max().item():.4f}")

# Check for NaNs in input
input_check = (input_tensor != input_tensor).sum().item()
print(f"  NaN check in input tensor: {input_check} NaNs")

# ============================================================================
# STEP 7: Run inference layer by layer with debugging
# ============================================================================
print("\n[STEP 7] Running inference with layer-by-layer debugging...")
start_time = time.time()

x = input_tensor
for i, layer in enumerate(model_layers):
    print(f"\n  Layer {i}: ", end="")
    
    if isinstance(layer, nn.Conv2d):
        print(f"Conv2d")
        x = layer(x)
    elif isinstance(layer, nn.GroupNorm):
        print(f"GroupNorm")
        x = layer(x)
    elif layer == "relu":
        print(f"ReLU activation")
        x = x.relu()
    elif layer == "gelu":
        print(f"GELU activation")
        x = x.gelu()
    elif isinstance(layer, tuple) and layer[0] == "dropout":
        print(f"Dropout (p={layer[1]})")
        x = x.dropout(layer[1])
    else:
        print(f"Unknown layer type: {layer}")
    
    # Debug stats after each layer
    x_realized = x.realize()  # Force computation
    x_np = x_realized.numpy()
    
    print(f"    Shape: {x.shape}")
    print(f"    Min: {np.nanmin(x_np):.6f}, Max: {np.nanmax(x_np):.6f}, Mean: {np.nanmean(x_np):.6f}")
    print(f"    NaN count: {np.isnan(x_np).sum()}")
    print(f"    Inf count: {np.isinf(x_np).sum()}")
    print(f"    L2 norm: {np.sqrt(np.nansum(x_np**2)):.6f}")
    
    # Stop if NaNs detected
    if np.isnan(x_np).sum() > 0:
        print(f"\n  !!! WARNING: NaNs detected after layer {i} !!!")
        print(f"  First NaN location: {np.argwhere(np.isnan(x_np))[0]}")
        break

output = x
inference_time = time.time() - start_time
print(f"\n  Inference completed in {inference_time:.2f} seconds")

# ============================================================================
# STEP 8: Save output
# ============================================================================
print("\n[STEP 8] Saving segmentation...")
# Get class with highest probability
seg_class = output.argmax(1)[0].numpy().astype(np.int32)
print(f"  Segmentation shape: {seg_class.shape}")
print(f"  Unique classes: {np.unique(seg_class)}")

seg_img = nib.Nifti1Image(seg_class, affine)
nib.save(seg_img, output_path)
print(f"  Segmentation saved to {output_path}")

print("\n" + "=" * 80)
print("SCRIPT COMPLETED")
print("=" * 80)
