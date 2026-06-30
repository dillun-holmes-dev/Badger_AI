# 🦡 Badger AI

> **Beat every object detector. Lighter. Faster. More accurate.**

Badger is an object detection research framework that systematically combines
the best ideas from YOLOX, YOLOv8, YOLOv9, D-FINE, DEIMv2, RT-DETR, and RF-DETR
into a single, MIT-licensed codebase — then pushes beyond them.

---

## 🎯 The Goal

Build a family of models that beats **every** publicly benchmarked detector on
[visionanalysis.org](https://www.visionanalysis.org) at every size tier:

| Size Tier | Current SOTA | mAP | Our Target |
|-----------|-------------|-----|------------|
| Nano (<1M) | DEIMv2-Atto | 27.5% | **>30%** |
| Tiny (~2M) | YOLOv9-T | 41.8% | **>44%** |
| Small (~10M) | D-FINE-S | 53.4% | **>55%** |
| Medium (~20M) | EC-M | 58.4% | **>59%** |
| Large (~30M) | D-FINE-L | 60.0% | **>61%** |
| X-Large (~50M+) | D-FINE-X | 61.4% | **>62%** |

---

## 🧬 Architecture — The Best of Everything

Badger combines proven innovations from multiple SOTA detectors:

| Innovation | Source | What It Does |
|-----------|--------|--------------|
| **CSPDarknet Backbone** | YOLOv5/v8 | Efficient cross-stage feature extraction |
| **PA-FPN Neck** | YOLOv8 | Path aggregation for multi-scale fusion |
| **Decoupled Head** | YOLOX | Separate cls/reg branches (better than coupled) |
| **Anchor-Free Detection** | YOLOX/FCOS | No anchor boxes = simpler + better for small objects |
| **DFL Regression** | YOLOv8 | Distribution focal loss for precise localization |
| **CIoU Box Loss** | YOLOv8 | Complete IoU with aspect ratio penalty |
| **SimOTA Assigner** | YOLOX | Dynamic-k optimal transport label assignment |
| **TAL Assigner** | YOLOv8 | Task-aligned label assignment (alternative) |
| **Mosaic + MixUp** | YOLOv5 | Strong data augmentation |

### Pipeline

```
Image [B, 3, 640, 640]
    │
    ▼
┌─────────────────────┐
│  CSPDarknet Backbone │  ← Feature extraction at 3 scales
│  P3(80×80) P4(40×40) │
│  P5(20×20)           │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  PA-FPN Neck         │  ← Multi-scale feature fusion
│  N3 ← N4 ← N5       │    (top-down + bottom-up paths)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Decoupled Head      │  ← Detection predictions
│  cls_branch: scores  │
│  reg_branch: boxes   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Post-processing     │  ← NMS + thresholding
│  [boxes, scores, cls]│
└─────────────────────┘
```

---

## 📦 Project Structure

```
Badger_Ai/
├── config/
│   ├── default.yaml          # Training & model configuration
│   └── coco.yaml             # COCO dataset paths & classes
│
├── src/
│   ├── models/
│   │   ├── blocks.py         # Conv, Bottleneck, C2f, SPPF, DFL
│   │   ├── backbone.py       # CSPDarknet (EfficientNet/ResNet coming)
│   │   ├── neck.py           # PA-FPN (BiFPN coming)
│   │   ├── head.py           # DecoupledHead, CoupledHead
│   │   └── badger.py         # Full Badger model + create_model()
│   │
│   ├── losses/
│   │   └── badger_loss.py    # BadgerLoss + TAL + SimOTA assigners
│   │
│   ├── data/
│   │   └── dataset.py        # COCO loader, mosaic, collate
│   │
│   └── utils/
│       ├── box_ops.py        # NMS, box format conversions
│       ├── metrics.py        # mAP, AP, IoU computation
│       ├── logging.py        # TensorBoard, checkpoint save/load
│       └── benchmark.py      # Standardized SOTA comparison
│
├── experiments/
│   └── improvements.py       # CBAM, ECA, SimOTA, EMA, multi-scale...
│
├── scripts/
│   ├── train.py              # Training entry point
│   └── benchmark.py          # Benchmark entry point
│
├── notebooks/                # Jupyter notebooks for exploration
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install

```bash
# Clone
git clone <your-repo> Badger_Ai
cd Badger_Ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify It Works

```python
import torch
from src.models import create_model

# Create a Badger-Small model
model = create_model('badger-s', num_classes=80)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Forward pass
dummy = torch.randn(1, 3, 640, 640)
cls_scores, bbox_preds = model(dummy)
print(f"P3 cls: {cls_scores[0].shape}")  # [1, 80, 80, 80]
print(f"P4 cls: {cls_scores[1].shape}")  # [1, 80, 40, 40]
print(f"P5 cls: {cls_scores[2].shape}")  # [1, 80, 20, 20]
```

### 3. Quick Benchmark

```bash
# Compare params, GFLOPs, and latency against SOTA
python scripts/benchmark.py --model badger-s
```

### 4. Train

```bash
# Train on COCO (download dataset first!)
python scripts/train.py --config config/default.yaml

# Train with experiments
python scripts/train.py --config config/default.yaml \
    --experiment simota_assigner,ema_weights,cbam_attention
```

---

## 🧪 Experiment System

Badger has a built-in experiment framework. Each experiment is an isolated
improvement that you can toggle on/off and measure:

```bash
# List all available experiments
python -c "from experiments import list_experiments; list_experiments()"

# Run a single experiment
python scripts/train.py --experiment simota_assigner

# Stack multiple experiments (combos!)
python scripts/train.py --experiment simota_assigner,ema_weights,cbam_attention
```

### Available Experiments

| Experiment | Expected Gain | Param Cost | Speed Cost |
|-----------|--------------|------------|------------|
| **CBAM Attention** | +0.5-1.5% mAP | ~1% | ~5% slower |
| **ECA Attention** | +0.3-1.0% mAP | <0.1% | ~2% slower |
| **SE Attention** | +0.5-1.0% mAP | ~1-2% | ~3% slower |
| **SimOTA Assigner** | +0.2-0.8% mAP | 0% | Free |
| **EMA Weights** | +0.3-0.8% mAP | 0%¹ | Free |
| **Multi-Scale Train** | +0.5-1.0% mAP | 0% | Variable |
| **BiFPN Neck** | +0.3-1.0% mAP | Minimal | ~10% slower |
| **Mish Activation** | +0.1-0.5% mAP | 0% | ~5% slower |
| **Label Smoothing** | +0.1-0.5% mAP | 0% | Free |
| **Attention Neck** | +1.0-3.0% mAP | ~10-15% | ~20% slower |
| **Gaussian DFL** | +0.2-0.5% mAP | 0% | ~2% slower |
| **Varifocal Loss** | +0.5-1.0% mAP | 0% | ~1% slower |
| **SIoU Box Loss** | +0.3-0.5% mAP | 0% | Same |
| **Close Mosaic** | +0.5-1.5% mAP | 0% | Faster late |
| **1280 Resolution** | +1.0-3.0% mAP | 0% | ~4× compute |
| **Badger-XL Scale** | +3.0-5.0% mAP | ~500% | ~6× compute |

¹ EMA uses 2x memory during training only

---

## 📊 Benchmarking

Badger uses the same protocol as [visionanalysis.org](https://www.visionanalysis.org)
for apples-to-apples comparison:

```bash
# Quick: params + GFLOPs + latency (no dataset needed)
python scripts/benchmark.py --model badger-s

# Full: quick + mAP on COCO val2017
python scripts/benchmark.py --model badger-s --full --data config/coco.yaml --compare

# Save results
python scripts/benchmark.py --model badger-s --save results.json
```

### Current SOTA Reference (June 2026, COCO val2017, PyTorch FP32)

| Rank | Model | mAP | Params | GFLOPs |
|------|-------|-----|--------|--------|
| 1 | D-FINE-X | 61.4% | 62.0M | 202 |
| 2 | DEIMv2-X | 61.3% | 51.2M | 152 |
| 3 | EC-X | 61.1% | 49.9M | 151 |
| 4 | RT-DETRv4-X | 60.0% | 62.6M | 202 |
| 5 | D-FINE-L | 60.0% | 31.2M | 91 |
| 10 | EC-M | 58.4% | 19.4M | 53 |
| 16 | YOLOv9-C | 57.1% | 25.5M | 52 |
| 34 | D-FINE-S | 53.4% | 10.3M | 25 |
| 44 | YOLOv9-S | 50.5% | 7.2M | 14 |
| 52 | YOLOv9-T | 41.8% | 2.0M | 4 |

---

## 🔬 Research Roadmap — How to Beat SOTA

### Phase 1: Solid Baseline (Week 1-2)
- [x] CSPDarknet backbone + PA-FPN neck + Decoupled head
- [x] TAL label assigner (YOLOv8-style)
- [x] CIoU + BCE + DFL loss
- [x] Mosaic + HSV augmentation
- [ ] Train on COCO, get baseline mAP

### Phase 2: Quick Wins (Week 3-4)
- [ ] **SimOTA assigner** — YOLOX's dynamic-k (free accuracy)
- [ ] **EMA weights** — smoother convergence (free at inference)
- [ ] **Multi-scale training** — better size generalization
- [ ] **Label smoothing** — regularization
- [ ] **Cosine LR schedule** — better convergence

### Phase 3: Attention (Week 5-6)
- [ ] **CBAM** — channel + spatial attention in backbone
- [ ] **ECA** — ultra-light channel attention
- [ ] Try attention in neck too

### Phase 4: Architecture Upgrades (Week 7-10)
- [ ] **Attention neck** — Swin window attention + global token mixing
- [ ] **Gaussian DFL** — continuous (μ,σ) instead of discrete bins
- [ ] **Varifocal loss** — IoU-weighted classification
- [ ] **SIoU loss** — angle-aware box regression
- [ ] **BiFPN neck** — weighted feature fusion (EfficientDet-style)
- [ ] **DEIM-style improvements** — explore distillation + matching
- [ ] **GELAN backbone** — YOLOv9's generalized efficient layer aggregation
- [ ] **RepConv** — reparameterizable convolutions

### Phase 5: Advanced Techniques (Week 11+)
- [ ] **Knowledge distillation** — train small from large Badger
- [ ] **Test-time augmentation** — free +0.5-1% mAP
- [ ] **Model soup** — average multiple checkpoints
- [ ] **NAS** — architecture search for the best combo
- [ ] **Custom augmentations** — beyond mosaic/mixup

---

## 🛠 Model Variants

```python
from src.models import create_model

# Nano:  ~0.9M params (phone/edge)
model = create_model('badger-n', num_classes=80)

# Small: ~9M params (lightweight desktop)
model = create_model('badger-s', num_classes=80)

# Medium: ~25M params (balanced)
model = create_model('badger-m', num_classes=80)

# Large:  ~44M params (high accuracy)
model = create_model('badger-l', num_classes=80)

# XL:     ~68M params (maximum accuracy)
model = create_model('badger-x', num_classes=80)
```

---

## 📝 Configuration

Edit `config/default.yaml` to customize:

- **Model**: backbone type, depth/width scaling, head type
- **Training**: batch size, LR, optimizer, loss weights
- **Label assignment**: `tal` (YOLOv8) or `simota` (YOLOX)
- **Augmentation**: mosaic, mixup, HSV, geometric
- **Regularization**: dropout, label smoothing
- **Logging**: TensorBoard, wandb, checkpoint frequency

---

## 🔗 References

### Foundational CNN Architecture
- LeNet-5 (LeCun et al., 1998) — http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
- AlexNet (Krizhevsky, Sutskever, Hinton, 2012) — https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
- VGGNet (Simonyan & Zisserman, 2014) — https://arxiv.org/abs/1409.1556
- GoogLeNet / Inception (Szegedy et al., 2014) — https://arxiv.org/abs/1409.4842
- ResNet (He et al., 2015) — https://arxiv.org/abs/1512.03385
- EfficientNet (Tan & Le, 2019) — https://arxiv.org/abs/1905.11946

### Training Mechanics
- Convolution Arithmetic (Dumoulin & Visin, 2016) — https://arxiv.org/abs/1603.07285
- Im2col/GEMM on GPU (Chellapilla, Puri & Simard, 2006) — https://hal.science/inria-00112631/document
- Batch Normalization (Ioffe & Szegedy, 2015) — https://arxiv.org/abs/1502.03167
- BN Optimization (Santurkar et al., 2018) — https://arxiv.org/abs/1805.11604
- Kaiming Init / PReLU (He et al., 2015) — https://arxiv.org/abs/1502.01852
- GELU (Hendrycks & Gimpel, 2016) — https://arxiv.org/abs/1606.08415

### Attention Mechanisms
- Self-Attention (Vaswani et al., 2017) — https://arxiv.org/abs/1706.03762
- ViT (Dosovitskiy et al., 2020) — https://arxiv.org/abs/2010.11929
- Swin Transformer (Liu et al., 2021) — https://arxiv.org/abs/2103.14030
- CBAM (Woo et al., 2018) — https://arxiv.org/abs/1807.06521
- ECA-Net (Wang et al., 2020) — https://arxiv.org/abs/1910.03151
- SE-Net (Hu et al., 2018) — https://arxiv.org/abs/1709.01507

### Object Detection
- Faster R-CNN (Ren et al., 2015) — https://arxiv.org/abs/1506.01497
- YOLO (Redmon et al., 2015) — https://arxiv.org/abs/1506.02640
- YOLOv3 (Redmon & Farhadi, 2018) — https://arxiv.org/abs/1804.02767
- Mask R-CNN (He et al., 2017) — https://arxiv.org/abs/1703.06870
- [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430) — Anchor-free + SimOTA + decoupled head
- Deformable DETR (Zhu et al., 2020) — https://arxiv.org/abs/2010.04159
- RT-DETR (Zhao et al., 2023) — https://arxiv.org/abs/2304.08069
- D-FINE (Peng et al., 2024) — https://arxiv.org/abs/2410.13842
- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics) — C2f, DFL, PA-FPN
- [YOLOv9: Learning What You Want to Learn](https://arxiv.org/abs/2402.13616) — GELAN, PGI
- [LibreYOLO](https://github.com/LibreYOLO/libreyolo) — Unified open-source detection library

### Loss Functions
- Focal Loss / RetinaNet (Lin et al., 2017) — https://arxiv.org/abs/1708.02002
- GIoU Loss (Rezatofighi et al., 2019) — https://arxiv.org/abs/1902.09630
- Generalized Focal Loss (Li et al., 2020) — https://arxiv.org/abs/2006.04388
- VarifocalNet (Zhang et al., 2020) — https://arxiv.org/abs/2008.13367
- SIoU Loss (Gevorgyan, 2022) — https://arxiv.org/abs/2205.12740

### Multimodal / Self-Supervised
- CLIP (Radford et al., 2021) — https://arxiv.org/abs/2103.00020
- MAE (He et al., 2021) — https://arxiv.org/abs/2111.06377

### Benchmarks & Leaderboards
- [Vision Analysis Leaderboard](https://www.visionanalysis.org) — SOTA detector comparisons
- [Papers With Code — Object Detection on COCO](https://paperswithcode.com/sota/object-detection-on-coco)

---

## 📄 License

MIT — use it, modify it, ship it.

---

<p align="center">
  <b>🦡 Built to beat everything. Systematically.</b>
</p>
