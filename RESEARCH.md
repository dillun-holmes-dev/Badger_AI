# Research Sources & Citation Rules

> **Rule**: Every architectural claim in Badger's improvement backlog must link a
> specific paper or repo with the exact dataset, hardware, and benchmark conditions
> the published number came from. No "research suggests" without a DOI or repo link.

---

## 1. Paper Discovery & Leaderboards

### Primary Leaderboard (Object Detection)
| Source | URL | What It Provides |
|--------|-----|------------------|
| **Papers With Code — Object Detection** | https://paperswithcode.com/task/object-detection | SOTA table with linked code, exact benchmark conditions (COCO val2017/test-dev, mAP metric) |
| **Vision Analysis** | https://www.visionanalysis.org | Independent re-benchmarking of detectors on identical hardware (Tesla T4, FP32 PyTorch) — removes hardware variance from comparisons |

### Daily Paper Feeds
| Source | URL | Use For |
|--------|-----|---------|
| **arXiv cs.CV (recent)** | https://arxiv.org/list/cs.CV/recent | New detection architectures, backbones, augmentation techniques |
| **arXiv cs.LG (recent)** | https://arxiv.org/list/cs.LG/recent | Training techniques, optimizers, loss functions, SSL methods |
| **HuggingFace Daily Papers** | https://huggingface.co/papers | Curated papers with linked model weights and demos — good for finding immediately reproducible work |
| **OpenReview** | https://openreview.net/ | CVPR/ICCV/ECCV/NeurIPS/ICLR papers with reviewer discussion — **read the reviews** to spot known weaknesses before adopting a method |

---

## 2. Reference Implementations (Model Zoos)

For comparing Badger's implementation choices against established baselines:

| Source | URL | What It Provides | Hardware Note |
|--------|-----|------------------|---------------|
| **timm (PyTorch Image Models)** | https://github.com/huggingface/pytorch-image-models | 300+ vision backbones with consistent eval harness, pretrained weights | Reported on V100/A100, ImageNet-1k |
| **torchvision** | https://github.com/pytorch/vision | Reference ResNet/EfficientNet/ViT implementations | Standard PyTorch benchmarking |
| **MMDetection** | https://github.com/open-mmlab/mmdetection | Detection architectures with standardized COCO benchmarks, training recipes | Reported on 8×V100, COCO val2017 |
| **MMSegmentation** | https://github.com/open-mmlab/mmsegmentation | Segmentation equivalent (useful for backbone quality assessment) | Same as MMDetection |
| **Ultralytics** | https://github.com/ultralytics/ultralytics | Current YOLO family, latency/accuracy on edge hardware (Jetson, Raspberry Pi) | Reported on Tesla T4, COCO val2017 |
| **LibreYOLO** | https://github.com/LibreYOLO/libreyolo | Unified open-source detection with reproducible configs | Reported on RTX 3090, COCO val2017 |

---

## 3. Conference Proceedings (Primary Sources)

**Always cite the proceedings version**, not just the arXiv preprint, when available.
Proceedings versions have passed peer review.

| Venue | URL | Years | Notes |
|-------|-----|-------|-------|
| **CVF Open Access** | https://openaccess.thecvf.com/ | CVPR, ICCV, WACV | Full papers, free access |
| **NeurIPS Proceedings** | https://proceedings.neurips.cc/ | All years | Includes supplemental material |
| **PMLR** | https://proceedings.mlr.press/ | ICML, AISTATS, CoRL | Open access proceedings |

---

## 4. Robustness & Corruption Benchmarks

Relevant because Badger targets real-world deployment conditions (lighting,
occlusion, sensor noise) — not just clean COCO images.

| Source | URL | What It Provides |
|--------|-----|------------------|
| **Hendrycks Robustness** | https://github.com/hendrycks/robustness | ImageNet-C (15 corruption types × 5 severities), methodology for building custom corruption suites |
| **Paper** | https://arxiv.org/abs/1903.12261 | Hendrycks & Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations" — the corruption taxonomy: noise, blur, weather, digital |
| **COCO-C** | (derived) | Apply same corruption taxonomy to COCO val2017 for detection-specific robustness measurement |

### Corruption Taxonomy (from Hendrycks & Dietterich 2019)
```
Noise:    Gaussian, Shot, Impulse
Blur:     Defocus, Glass, Motion, Zoom
Weather:  Snow, Frost, Fog, Brightness
Digital:  Contrast, Elastic, Pixelate, JPEG
```
Each at 5 severity levels. Badger should report mAP degradation per category.

---

## 5. Edge / Latency Benchmarking

Relevant if Badger targets deployment outside raw PyTorch:

| Source | URL | Use For |
|--------|-----|---------|
| **PyTorch Benchmark** | https://github.com/pytorch/benchmark | Pattern for building BASELINE.json with reproducible latency numbers |
| **ONNX Runtime** | https://onnxruntime.ai/ | Cross-checking inference latency after quantization/export |
| **TensorRT** | https://developer.nvidia.com/tensorrt | NVIDIA-specific deployment benchmarking |
| **OpenVINO** | https://docs.openvino.ai/ | Intel CPU/GPU deployment benchmarking |

---

## 6. Citation Format (REQUIRED)

Every entry in `IMPROVEMENT_BACKLOG.md` must follow this format:

```markdown
### [Technique Name]

- **Paper**: Author et al., "Title" (Venue Year) — [arXiv:XXXX.XXXXX](link)
- **Repo**: [github.com/owner/repo](link) (if available)
- **Claimed improvement**: +X.X mAP on [dataset] ([hardware], [precision])
- **Our hypothesis**: [why this should work for Badger specifically]
- **Validation plan**: [how we'll measure it — which config, which dataset]
- **Status**: [not-tested | testing | adopted | rejected]
- **Our result**: [mAP before/after, hardware, notes] (filled after testing)
```

### Example (Correct):
```markdown
### SimOTA Label Assignment

- **Paper**: Ge et al., "YOLOX: Exceeding YOLO Series in 2021" (arXiv 2021) — [arXiv:2107.08430](https://arxiv.org/abs/2107.08430)
- **Repo**: [github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- **Claimed improvement**: +0.8 AP on COCO val2017 (YOLOX-S vs YOLOv5-S, 640×640, Tesla V100, FP32)
- **Our hypothesis**: Dynamic-k assignment adapts per-object, helping Badger's small-object detection
- **Validation plan**: Train Badger-S 300 epochs on COCO, compare TAL vs SimOTA, measure mAP@[0.5:0.95]
- **Status**: implemented (not yet trained)
- **Our result**: TBD
```

### Anti-pattern (Wrong — DO NOT USE):
```markdown
- Research suggests SimOTA is better for detection accuracy
```
This is banned. No claim without a specific paper/repo and exact benchmark conditions.

---

## 7. Quick Reference — Key Papers by Category

### Backbone Architecture
| Paper | Venue | Code | Key Metric |
|-------|-------|------|------------|
| ResNet (He et al., 2015) | CVPR 2016 | torchvision | 76.1% top-1 ImageNet (ResNet-50) |
| EfficientNet (Tan & Le, 2019) | ICML 2019 | timm | 84.3% top-1 ImageNet (EfficientNet-B7) |
| ConvNeXt (Liu et al., 2022) | CVPR 2022 | timm | 87.8% top-1 ImageNet (ConvNeXt-XL) |
| Swin Transformer (Liu et al., 2021) | ICCV 2021 | timm | 87.3% top-1 ImageNet (Swin-L) |
| ViT (Dosovitskiy et al., 2020) | ICLR 2021 | timm | 88.6% top-1 ImageNet (ViT-H/14) |

### Detection Architecture
| Paper | Venue | Code | Key Metric (COCO val2017) |
|-------|-------|------|---------------------------|
| YOLOX (Ge et al., 2021) | arXiv | Megvii-BaseDetection/YOLOX | 51.5 mAP (YOLOX-X, 640) |
| D-FINE (Peng et al., 2024) | arXiv | Peterande/D-FINE | 61.4 mAP (D-FINE-X, 640) |
| DEIM (Huang et al., 2024) | arXiv | ShihuaHuang/DEIM | 61.3 mAP (DEIMv2-X, 640) |
| RT-DETR (Zhao et al., 2023) | arXiv | lyuwenyu/RT-DETR | 54.8 mAP (RT-DETR-R101, 640) |
| YOLOv9 (Wang et al., 2024) | arXiv | WongKinYiu/yolov9 | 55.6 mAP (YOLOv9-X, 640) |

### Loss Functions
| Paper | Venue | Key Contribution |
|-------|-------|------------------|
| Focal Loss (Lin et al., 2017) | ICCV 2017 | Downweight easy negatives: `FL(p) = -(1-p)ᵞlog(p)` |
| Generalized Focal Loss (Li et al., 2020) | NeurIPS 2020 | DFL for bounding box distributions |
| VarifocalNet (Zhang et al., 2020) | CVPR 2021 | IoU-weighted classification loss |
| SIoU (Gevorgyan, 2022) | arXiv 2022 | Angle-aware box regression loss |

### Label Assignment
| Paper | Key Innovation | Where Used |
|-------|---------------|------------|
| SimOTA (Ge et al., 2021) | Dynamic-k via IoU distribution | YOLOX |
| TAL (Feng et al., 2022) | Task alignment score `clsᵅ × iouᵝ` | YOLOv8 |
| D-DETR (Zhu et al., 2020) | Deformable attention for assignment | D-FINE, DEIM |

---

## 8. How to Add a New Source

When you find a promising paper:

1. **Check reproducibility first**: Does it have code? Are weights released?
2. **Read OpenReview reviews**: Are there known failure modes? Does it only work on specific datasets?
3. **Note exact benchmark conditions**: Dataset split, image size, hardware, precision (FP32/FP16/INT8), number of training epochs
4. **Add to this file** under the appropriate category
5. **Create an entry in `IMPROVEMENT_BACKLOG.md`** using the required citation format
6. **Create an experiment** in `experiments/improvements.py` with a unique key
