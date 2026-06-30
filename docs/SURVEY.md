# SURVEY — Individual Techniques Across SOTA Object Detectors

> **Methodology**: Each row is one INDIVIDUAL technique, not a whole architecture.
> Numbers are from visionanalysis.org (COCO val2017, PyTorch FP32 unless noted).
> "Cost" columns are estimates relative to a basic anchor-based YOLOv5 baseline.

---

## 1. Detection Head Design

| Technique | Paper/Source | What It Improves | Accuracy Gain | Param Cost | Speed Cost | Used By |
|-----------|-------------|------------------|---------------|------------|------------|---------|
| **Decoupled head** (separate cls/reg branches) | YOLOX (Ge et al. 2021) | Classification and regression benefit from different features | +1.0-2.0 AP | +5-10% params | ~2% slower | YOLOX, YOLOv8+, D-FINE |
| **Anchor-free detection** (predict l,t,r,b from grid) | FCOS (Tian et al. 2019), YOLOX | Eliminates anchor hyperparameters; better for small objects | +1.5 AP (vs anchor-based, same backbone) | -30% params (no anchor computation) | ~5% faster (no anchor matching) | YOLOX, YOLOv8, D-FINE, DEIM |
| **Distribution Focal Loss (DFL)** for box regression | Generalized Focal Loss (Li et al. 2020) | Predicts edge distribution instead of scalar → finer localization | +0.3-0.8 AP | +2% params (reg_max=16) | ~1% slower | YOLOv8, D-FINE, DEIMv2 |
| **IoU-aware classification** (Varifocal Loss) | VarifocalNet (Zhang et al. 2020) | High-IoU positives get higher classification weight | +0.5-1.5 AP | 0% | Free | D-FINE, EC |
| **Task-aligned label assignment (TAL)** | TOOD (Feng et al. 2021), YOLOv8 | Matches predictions to GT based on both cls AND reg quality | +0.5-1.0 AP | 0% | Free (assigner runs @no_grad) | YOLOv8, YOLOv9 |
| **SimOTA dynamic-k assignment** | YOLOX (Ge et al. 2021) | Dynamic k per GT based on IoU distribution | +0.3-0.8 AP (over fixed top-k) | 0% | ~1% slower | YOLOX |
| **Hybrid matching** (CNN anchor + DETR query) | D-FINE (Peng et al. 2024), DEIM | Combines dense prediction efficiency with DETR-style sparse matching | +2.0-3.0 AP | +10% params | ~10% slower | D-FINE, DEIM, DEIMv2, EC |

## 2. Backbone Architecture

| Technique | Paper/Source | What It Improves | Accuracy Gain | Param Cost | Speed Cost | Used By |
|-----------|-------------|------------------|---------------|------------|------------|---------|
| **CSPDarknet** (cross-stage partial) | YOLOv5, YOLOX | Better gradient flow via split-merge paths | +1.0 AP (over plain Darknet) | -10% params (vs plain) | ~5% faster | YOLOv5, YOLOX, YOLOv9 |
| **C2f module** (faster CSP variant) | YOLOv8 (Ultralytics 2023) | More bottlenecks with fewer parameters | +0.5 AP (over C3) | -15% params | ~5% faster | YOLOv8, YOLOv10 |
| **EfficientRep** (reparameterized backbone) | YOLOv6 (Li et al. 2022) | Multi-branch training → single-branch deployment | +1.0-2.0 AP | 3× training, 1× deployment | 0% inference cost | YOLOv6, YOLOv9 |
| **GELAN** (generalized ELAN) | YOLOv9 (Wang et al. 2024) | Generalized efficient layer aggregation with programmable gradient paths | +2.0 AP (over CSPDarknet) | Similar | Similar | YOLOv9 |
| **ResNet with deformable convs (DCNv2)** | DCNv2 (Zhu et al. 2019) | Learns where to sample → better for non-rigid objects | +1.0-3.0 AP | +10% params | ~15% slower | EC, some RT-DETR variants |
| **Transformer backbone** (ViT/Swin) | ViT (Dosovitskiy et al. 2020), Swin (Liu et al. 2021) | Global receptive field from layer 1 | +3.0-5.0 AP (large models) | +50-100% params | 30-100% slower | RT-DETR, D-FINE (encoder), DEIM |
| **EfficientNet-style compound scaling** | EfficientNet (Tan & Le 2019) | Scales width, depth, resolution jointly under fixed budget | +1.0-2.0 AP (per scale-up) | Proportional | Proportional | YOLOv8, YOLOv9 scaling |

## 3. Neck / Feature Fusion

| Technique | Paper/Source | What It Improves | Accuracy Gain | Param Cost | Speed Cost | Used By |
|-----------|-------------|------------------|---------------|------------|------------|---------|
| **PA-FPN** (path aggregation) | PANet (Liu et al. 2018), YOLOv8 | Bottom-up path augmentation on top of FPN | +1.0-2.0 AP (over plain FPN) | +20% params | ~10% slower | YOLOv5+, YOLOX, YOLOv8+ |
| **BiFPN** (weighted bidirectional FPN) | EfficientDet (Tan et al. 2020) | Learned per-input fusion weights; removes single-input nodes | +0.5-1.5 AP (over PA-FPN) | -20% params (fewer nodes) | ~5% faster | EfficientDet, YOLOv9 variant |
| **ASFF** (adaptive spatial fusion) | ASFF (Liu et al. 2019) | Spatially-varying attention weights for multi-scale fusion | +0.5-1.0 AP | +5% params | ~10% slower | Some YOLO variants |
| **Transformer encoder neck** (cross-scale attention) | DETR, RT-DETR, D-FINE | Global cross-scale reasoning via self/cross-attention | +2.0-4.0 AP | +30-50% params | 20-50% slower | RT-DETR, D-FINE, DEIM |
| **RepNCSP** (reparameterized neck) | YOLOv6, YOLOv9 | Multi-branch neck → single-branch at deploy | +0.5-1.0 AP | 3× training | 0% inference | YOLOv6, YOLOv9 |

## 4. Training Strategy

| Technique | Paper/Source | What It Improves | Accuracy Gain | Param Cost | Speed Cost | Training Time Cost | Used By |
|-----------|-------------|------------------|---------------|------------|------------|-------------------|---------|
| **Mosaic augmentation** (4-image stitch) | YOLOv5 (Ultralytics 2020) | Increases object diversity per image → strong regularization | +1.0-2.0 AP | 0% | 0% (only training) | 5% slower per epoch | YOLOv5+, YOLOX, YOLOv8+ |
| **MixUp augmentation** | Zhang et al. 2017 | Blends two images → even stronger regularization | +0.3-0.5 AP | 0% | 0% | 2% slower | YOLOv5+, YOLOX |
| **Close mosaic late training** | YOLOX practice | Fine-tune on real image statistics for final epochs | +0.5-1.5 AP | 0% | 0% | None (epochs stay same) | YOLOX, YOLOv8, YOLOv9 |
| **Multi-scale training** (random resize) | YOLOv3+ practice | Robustness to object size variation | +0.5-1.5 AP | 0% | 0% | Variable (depends on size) | All modern detectors |
| **Cosine LR schedule** | Standard practice | Smoother convergence vs step decay | +0.2-0.5 AP | 0% | 0% | None | Most detectors |
| **EMA weight averaging** | Izmailov et al. 2018 | Smoother final weights → better generalization | +0.3-1.0 AP | 2× memory (training only) | 0% inference | 2% slower (update step) | YOLOX, D-FINE, DEIM |
| **Knowledge distillation** (teacher→student) | Hinton et al. 2015 | Train small model to mimic large model | +1.0-4.0 AP (student vs training from scratch) | 0% (student unchanged) | 0% inference | +100% (teacher forward pass) | D-FINE, DEIM, YOLO-NAS |
| **Test-time augmentation (TTA)** | Standard practice | Multi-scale + flip inference merging | +0.5-1.5 AP | 0% | 4-8× slower inference | N/A (inference only) | Competition entries |

## 5. Loss Functions

| Technique | Paper/Source | What It Improves | Accuracy Gain | Param Cost | Speed Cost | Used By |
|-----------|-------------|------------------|---------------|------------|------------|---------|
| **CIoU loss** (box regression) | Zheng et al. 2020 | Adds center distance + aspect ratio to IoU | +0.5-1.0 AP (over IoU) | 0% | Free | YOLOv5+, YOLOX, YOLOv8+ |
| **SIoU loss** (angle-aware) | Gevorgyan 2022 | Adds angle cost → better for rotated objects | +0.2-0.5 AP (over CIoU) | 0% | Free | Some YOLO variants |
| **Focal Loss** (classification) | Lin et al. 2017 | Downweights easy negatives → handles class imbalance | +1.0-2.0 AP (vs plain CE in dense detection) | 0% | Free | RetinaNet; implicitly used in BCE variants |
| **Quality Focal Loss (QFL)** | GFL (Li et al. 2020) | Jointly represents classification score + IoU quality | +0.5-1.0 AP | 0% | Free | GFL, D-FINE |

## 6. Post-processing

| Technique | Paper/Source | What It Improves | Accuracy Gain | Param Cost | Speed Cost | Used By |
|-----------|-------------|------------------|---------------|------------|------------|---------|
| **Weighted Box Fusion (WBF)** | Solovyev et al. 2019 | Merges overlapping boxes instead of discarding | +0.3-0.5 AP (over NMS) | 0% | 2× slower than NMS | Ensembles, TTA |
| **Soft-NMS** | Bodla et al. 2017 | Decays scores instead of zeroing → better for crowded scenes | +0.2-0.5 AP | 0% | 5% slower | Some detectors |

## 7. Deployment Optimization

| Technique | Paper/Source | What It Improves | Speed Gain | Accuracy Cost | Used By |
|-----------|-------------|------------------|------------|---------------|---------|
| **Conv+BN fusion** | Standard practice | Removes BN computation at inference | 5-15% faster | 0% | All production detectors |
| **FP16 inference** | Standard practice | Half precision on Tensor Core GPUs | 1.5-2× faster | <0.1% AP | All modern detectors |
| **INT8 quantization** | Standard practice | 8-bit weights + activations | 2-4× faster | 0.1-0.5% AP | Edge deployment |
| **Reparameterization** (RepConv) | RepVGG (Ding et al. 2021) | Multi-branch training → single-branch deploy | 10-20% faster at inference | 0% | YOLOv6, YOLOv7, YOLOv9 |
| **TensorRT export** | NVIDIA | Optimized for NVIDIA GPUs | 2-5× faster (varies by hardware) | <0.1% AP | Production NVIDIA deployments |

---

## Key Patterns Across the Leaderboard

### What Separates 61+ mAP from 55-57 mAP (~6 AP gap)?

Looking at D-FINE/DEIMv2/EC (61.4/61.3/61.1) vs YOLOv9-C/YOLOX-X (57.1/56.3):

| Factor | High-SOTA (61+) | Mid-SOTA (55-57) | Gap |
|--------|-----------------|-------------------|-----|
| **Matching strategy** | DETR-style hybrid matching with Hungarian algorithm | CNN-style anchor-free with SimOTA/TAL | +2-3 AP |
| **Neck** | Transformer encoder with cross-scale attention | PA-FPN (pure CNN) | +1-2 AP |
| **Loss design** | IoU-aware + distribution-based + quality focal | CIoU + BCE + DFL | +0.5-1 AP |
| **Training recipe** | EMA + distillation + close mosaic + multi-scale | Basic mosaic + standard schedule | +1-2 AP |
| **Backbone** | Transformer-enhanced (D-FINE) or DCNv2 (EC) | CSPDarknet (pure CNN) | +1-2 AP |

### The "DETR gap" is real

The top 4 models (D-FINE-X, DEIMv2-X, EC-X, RT-DETRv4-X) ALL use some form of transformer/DETR-style architecture. The best pure-CNN model is YOLOv9-C at 57.1 mAP — a 4.3 AP gap to D-FINE-X.

### Efficiency outliers

- **EC-M**: 58.4 mAP from only 19.4M params — the most efficient model above 58 mAP
- **D-FINE-S**: 53.4 mAP from 10.3M params — at the Small tier
- **YOLOv9-T**: 41.8 mAP from 2.0M params — strongest ultra-light model
