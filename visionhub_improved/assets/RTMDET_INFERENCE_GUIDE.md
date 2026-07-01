# RTMDetPose Inference & Export Guide

## PyTorch Inference

Run inference on images using a trained RTMDetPose checkpoint. **The script recursively searches all subdirectories** for images.

```bash
# Single image
visionhub-infer-rtmdetpose \
    --checkpoint output/rtmdetpose_hgnetv2_s_custom/checkpoint_best_regular.pth \
    --config configs.rtmdetpose.rtmdetpose_hgnetv2_s_custom \
    --source test_image.jpg \
    --device cuda

# Directory with nested folders (recursively searches all subdirectories)
visionhub-infer-rtmdetpose \
    --checkpoint output/rtmdetpose_hgnetv2_s_custom/checkpoint_best_regular.pth \
    --config configs.rtmdetpose.rtmdetpose_hgnetv2_s_custom \
    --source data/test_data/ \
    --device cuda \
    --conf_thresh 0.35 \
    --nms_thresh 0.45 \
    --output_dir inference_output

# Show results interactively
visionhub-infer-rtmdetpose \
    --checkpoint output/rtmdetpose_hgnetv2_s_custom/checkpoint_best_regular.pth \
    --config configs.rtmdetpose.rtmdetpose_hgnetv2_s_custom \
    --source test_image.jpg \
    --show
```

### Arguments:
- `--checkpoint`: Path to checkpoint file (.pth)
- `--config`: Config module (dot-separated path, e.g., `configs.rtmdetpose.rtmdetpose_hgnetv2_s_custom`)
- `--source`: Image file or directory (recursively searches all subdirectories for .jpg, .jpeg, .png, .bmp, .webp)
- `--output_dir`: Output directory for visualizations (default: `inference_output`)
- `--device`: Device to use (`cuda` or `cpu`)
- `--conf_thresh`: Confidence threshold (default: 0.35)
- `--nms_thresh`: NMS IoU threshold to filter overlapping predictions (default: uses config value, recommended: 0.4-0.5)
- `--image_size`: Input image size (default: 640)
- `--use_ema`: Use EMA weights from checkpoint (default: True)
- `--show`: Display results interactively

---

## ONNX Export

Export a trained RTMDetPose model to ONNX format:

```bash
visionhub-export-rtmdetpose-onnx \
    --config configs/rtmdetpose/rtmdetpose_hgnetv2_s_custom.py \
    --resume output/rtmdetpose_hgnetv2_s_custom/checkpoint_best_regular.pth \
    --check \
    --simplify
```

### Arguments:
- `--config`: Path to config file (.py)
- `--resume`: Path to checkpoint file (.pth)
- `--check`: Run ONNX model checker (default: True)
- `--simplify`: Simplify ONNX model using onnxsim (default: True)

### Output:
- ONNX model saved to: `onnx_engines/rtmdetpose_hgnetv2_s_custom.onnx`
- Class mappings saved to: `onnx_engines/rtmdetpose_hgnetv2_s_custom_class_mappings.json`

### ONNX Model Interface:
**Inputs:**
- `images`: `[N, 3, 640, 640]` - Batch of images (RGB, normalized 0-1)
- `orig_target_sizes`: `[N, 2]` - Original image sizes (H, W)

**Outputs:**
- `scores`: `[N, 300]` - Detection confidence scores
- `labels`: `[N, 300]` - Class labels (int64)
- `keypoints`: `[N, 300, K*3]` - Keypoints (x, y, visibility)
- `boxes`: `[N, 300, 4]` - Bounding boxes (x1, y1, x2, y2)

All outputs are padded to max_detections=300.

---

## Example Workflow

1. **Train RTMDetPose:**
```bash
visionhub-train \
    --config_file configs/rtmdetpose/rtmdetpose_hgnetv2_s_custom.py \
    --amp
```

2. **Run Inference:**
```bash
# Recursively processes all images in data/test_data/ and its subdirectories
visionhub-infer-rtmdetpose \
    --checkpoint output/rtmdetpose_hgnetv2_s_custom/checkpoint_best_regular.pth \
    --config configs.rtmdetpose.rtmdetpose_hgnetv2_s_custom \
    --source data/test_data/ \
    --output_dir results/
```

3. **Export to ONNX:**
```bash
visionhub-export-rtmdetpose-onnx \
    --config configs/rtmdetpose/rtmdetpose_hgnetv2_s_custom.py \
    --resume output/rtmdetpose_hgnetv2_s_custom/checkpoint_best_regular.pth
```

4. **Use ONNX for deployment:**
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("onnx_engines/rtmdetpose_hgnetv2_s_custom.onnx")
outputs = session.run(
    None,
    {
        "images": np.random.rand(1, 3, 640, 640).astype(np.float32),
        "orig_target_sizes": np.array([[640, 640]]).astype(np.int64)
    }
)
scores, labels, keypoints, boxes = outputs
```

---

## Key Differences from RTMOPose

**RTMDetPose:**
- ✅ No DCC component (simpler, faster)
- ✅ LTRB distance-based bbox regression
- ✅ Direct keypoint offset predictions
- ✅ Shared regression + keypoint branch

**RTMOPose:**
- Uses DCC (Dynamic Coordinate Classification) for final keypoint coordinates
- Requires pose vectors and dual-head architecture
- Slightly more accurate but slower

Choose RTMDetPose for faster inference, RTMOPose for maximum accuracy.

---

## TensorRT Export (Optional)

For TensorRT deployment, first export to ONNX, then use `trtexec`:

```bash
# Export to ONNX first
visionhub-export-rtmdetpose-onnx \
    --config configs/rtmdetpose/rtmdetpose_hgnetv2_s_custom.py \
    --resume output/rtmdetpose_hgnetv2_s_custom/checkpoint_best_regular.pth

# Convert to TensorRT
trtexec \
    --onnx=onnx_engines/rtmdetpose_hgnetv2_s_custom.onnx \
    --saveEngine=trt_engines/rtmdetpose_hgnetv2_s_custom.engine \
    --fp16 \
    --workspace=4096
```

Or use the existing export script:
```bash
visionhub-export-tensorrt \
    --onnx onnx_engines/rtmdetpose_hgnetv2_s_custom.onnx \
    --output trt_engines/rtmdetpose_hgnetv2_s_custom.engine \
    --fp16
```

---

## Troubleshooting

### Multiple Overlapping Predictions

**Problem:** Seeing multiple bounding boxes and keypoints stacked on top of each other for the same object.

**Cause:** NMS (Non-Maximum Suppression) threshold is too high, allowing overlapping predictions to survive.

**Solution:**
1. **Lower the NMS threshold at runtime:**
   ```bash
   visionhub-infer-rtmdetpose \
       --checkpoint output/rtmdetpose_hgnetv2_s_custom/checkpoint0130.pth \
       --config configs.rtmdetpose.rtmdetpose_hgnetv2_s_custom \
       --source data/test_data/ \
       --nms_thresh 0.45  # Try values between 0.4-0.5
   ```

2. **Or change it permanently in your config:**
   ```python
   # In configs/rtmdetpose/rtmdetpose_hgnetv2_s_custom.py
   postprocessor.nms_threshold = 0.45
   ```

**Explanation:** 
- NMS threshold controls when overlapping boxes are considered duplicates
- Higher values (0.6-0.7) = more lenient, keeps overlapping boxes
- Lower values (0.4-0.5) = stricter, removes more duplicates
- For pose estimation, use 0.4-0.5 to ensure one detection per object
