# Class Mappings Feature

## Overview

Class mappings are now automatically baked into model checkpoints during training, similar to YOLO. This allows the model to output actual class names (e.g., "black_bx_side") instead of numeric IDs (e.g., "2") during inference.

## How It Works

### During Training

1. **Dataset Configuration**: Class mappings are extracted from your dataset's COCO annotations file (`data/coco/train/coco_instances.json`)

2. **Automatic Extraction**: The training config (`configs/detrpose/include/dataset_custom.py`) automatically creates a `CLASS_MAPPINGS` dictionary:
   ```python
   CLASS_MAPPINGS = {c["id"]: c["name"] for c in categories}
   # Example: {2: "black_bx_side", 3: "black_bk_side", ...}
   ```

3. **Checkpoint Saving**: Class mappings are automatically saved in every checkpoint file:
   - `checkpoint.pth` (latest)
   - `checkpoint####.pth` (numbered checkpoints)
   - `checkpoint_best_regular.pth` (best model)

### During Inference

Class mappings are automatically loaded from checkpoints and used to display human-readable class names:

#### PyTorch Inference
```bash
visionhub-infer-detr \
  --checkpoint output/detrpose_hgnetv2_s_custom/checkpoint_best_regular.pth \
  --config configs.detrpose.detrpose_hgnetv2_s_custom \
  --input path/to/image.jpg
```

**Output:**
```
============================================================
Loaded class mappings from checkpoint:
  ID 2: black_bx_side
  ID 3: black_bk_side
  ID 4: black_bk_upright
  ...
============================================================

Detections in image.jpg:
  [1] black_bx_side (ID: 2, Score: 0.892)
  [2] green_ar_side (ID: 17, Score: 0.765)
  [3] white_ax_upright (ID: 19, Score: 0.543)
```

#### ONNX/TensorRT Inference

For ONNX and TensorRT models, class mappings are saved as a separate JSON file:

**Export to ONNX:**
```bash
visionhub-export-onnx \
  -c configs/detrpose/detrpose_hgnetv2_s_custom.py \
  -r output/detrpose_hgnetv2_s_custom/checkpoint_best_regular.pth
```

This creates:
- `onnx_engines/detrpose_hgnetv2_s_custom.onnx`
- `onnx_engines/detrpose_hgnetv2_s_custom_class_mappings.json` ✨ (NEW)

**ONNX Inference:**
```bash
visionhub-infer-onnx \
  --onnx onnx_engines/detrpose_hgnetv2_s_custom.onnx \
  --annotator coco \
  -i path/to/image.jpg
```

The inference script automatically finds and loads the `*_class_mappings.json` file.

**TensorRT Export:**
```bash
visionhub-export-tensorrt
```

This automatically copies class mappings to the TensorRT engines folder.

## File Structure

```
output/detrpose_hgnetv2_s_custom/
├── checkpoint.pth                      # Contains class_mappings
├── checkpoint_best_regular.pth         # Contains class_mappings
└── checkpoint0049.pth                  # Contains class_mappings

onnx_engines/
├── detrpose_hgnetv2_s_custom.onnx
└── detrpose_hgnetv2_s_custom_class_mappings.json  # Exported class mappings

trt_engines/
├── detrpose_hgnetv2_s_custom.engine
└── detrpose_hgnetv2_s_custom_class_mappings.json  # Copied class mappings
```

## API Usage

### Utility Functions

The `class_mapping_utils.py` module provides helper functions:

```python
from tools.inference.class_mapping_utils import (
    get_class_name,
    load_class_mappings_from_checkpoint,
    load_class_mappings_from_json,
    find_class_mappings_json,
    print_detections
)

# Load from PyTorch checkpoint
class_mappings = load_class_mappings_from_checkpoint('checkpoint.pth')

# Load from JSON file
class_mappings = load_class_mappings_from_json('model_class_mappings.json')

# Auto-find JSON near model file
class_mappings = find_class_mappings_json('model.onnx')

# Get class name from ID
class_name = get_class_name(2, class_mappings)  # Returns "black_bx_side"

# Print formatted detections
print_detections(labels, scores, class_mappings, max_display=10)
```

### Direct Checkpoint Access

```python
import torch

checkpoint = torch.load('checkpoint.pth', map_location='cpu', weights_only=False)
class_mappings = checkpoint.get('class_mappings', {})

# Example: {2: "black_bx_side", 3: "black_bk_side", ...}
```

## Backward Compatibility

- **Old checkpoints** without class mappings will still work - predictions will show numeric IDs
- **Warning messages** inform you when class mappings are missing
- No breaking changes to existing training or inference scripts

## Benefits

✅ **Consistent with YOLO**: Similar user experience to popular frameworks  
✅ **No manual mapping**: Class names travel with the model automatically  
✅ **Multi-format support**: Works with PyTorch, ONNX, and TensorRT  
✅ **Dataset-agnostic**: Adapts to any custom dataset automatically  
✅ **Easy debugging**: Immediately see what your model is detecting  

## Troubleshooting

### "No class mappings found in checkpoint"

**Cause**: Checkpoint was created before this feature was added.

**Solution**: Retrain your model, or add class mappings manually:
```python
import torch
checkpoint = torch.load('old_checkpoint.pth', map_location='cpu', weights_only=False)
checkpoint['class_mappings'] = {
    2: "black_bx_side",
    3: "black_bk_side",
    # ... your classes
}
torch.save(checkpoint, 'updated_checkpoint.pth')
```

### "Could not find class_mappings.json near model"

**Cause**: ONNX/TensorRT model exported before this feature was added.

**Solution**: Re-export your model using the updated export scripts, or create the JSON manually:
```json
{
  "2": "black_bx_side",
  "3": "black_bk_side"
}
```
Save as `model_name_class_mappings.json` next to your model file.

## Modified Files

**Training:**
- `configs/detrpose/include/dataset_custom.py` - Extracts class mappings
- `visionhub/solver/trainer.py` - Saves class mappings in checkpoints

**Inference:**
- `tools/inference/class_mapping_utils.py` - Utility functions (NEW)
- `tools/inference/torch_inf.py` - PyTorch inference with class names
- `tools/inference/onnx_inf.py` - ONNX inference with class names
- `tools/inference/trt_inf.py` - TensorRT inference with class names
- `tools/scripts/inference.py` - Scripts inference with class names

**Export:**
- `tools/deployment/export_onnx.py` - Exports class mappings JSON
- `tools/deployment/export_tensorrt.py` - Copies class mappings to TRT folder
