# Improvement Backlog

> **Citation rule**: Every entry must link a specific paper or repo with the exact
> dataset, hardware, and benchmark conditions the published number came from.
> See `RESEARCH.md` for the full source taxonomy.

## Status Legend
- `[ ]` not tested
- `[~]` in progress
- `[+]` adopted (verified improvement)
- `[-]` rejected (no improvement or regression)
- `[?]` inconclusive (need more runs)

---

## Phase 2: Label Assignment & Loss Functions

### [ ] SimOTA Label Assignment

- **Paper**: Ge et al., "YOLOX: Exceeding YOLO Series in 2021" (arXiv 2021) — [arXiv:2107.08430](https://arxiv.org/abs/2107.08430)
- **Repo**: [github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache 2.0)
- **Claimed improvement**: YOLOX-S achieves 40.5 mAP vs YOLOv5-S 37.2 mAP on COCO val2017 (640×640, Tesla V100, FP32, 300 epochs). SimOTA alone accounts for ~+0.8 AP of this gap per the ablation in Table 3.
- **Our hypothesis**: Dynamic k = `Σ(top_k_iou).sum().clamp(min=1)` adapts positives per GT based on object size. Badger's small-object detection should benefit because small objects naturally have lower IoU and need fewer (but higher-quality) positives.
- **Validation plan**: Train Badger-S 300 epochs on COCO train2017, eval on val2017. Compare `assigner=tal` vs `assigner=simota`. Measure mAP@[0.5:0.95], AP_S, AP_M, AP_L.
- **Experiment key**: `simota_assigner`
- **Our result**: TBD

### [ ] Varifocal Classification Loss

- **Paper**: Zhang et al., "VarifocalNet: An IoU-aware Dense Object Detector" (CVPR 2021) — [arXiv:2008.13367](https://arxiv.org/abs/2008.13367)
- **Repo**: [github.com/hyz-xmaster/VarifocalNet](https://github.com/hyz-xmaster/VarifocalNet) (Apache 2.0)
- **Claimed improvement**: +1.6 AP over standard Focal Loss on COCO val2017 (VarifocalNet-ResNet-50, 1333×800, 8×V100, FP32, 24 epochs). The key insight: `VFL(p, q) = -q·log(p)` where q = IoU for positives, 0 for negatives.
- **Our hypothesis**: Coupling classification confidence to localization quality should reduce false positives with poorly localized boxes — a known weakness of standard BCE.
- **Validation plan**: Same Badger-S training protocol as SimOTA experiment. Compare BCE vs Varifocal classification loss.
- **Experiment key**: `varifocal_loss`
- **Our result**: TBD

### [ ] SIoU Box Regression Loss

- **Paper**: Gevorgyan, "SIoU Loss: More Powerful Learning for Bounding Box Regression" (arXiv 2022) — [arXiv:2205.12740](https://arxiv.org/abs/2205.12740)
- **Repo**: No official repo; our implementation in `src/losses/advanced_losses.py`
- **Claimed improvement**: +0.5 AP over CIoU on COCO val2017 (YOLOv5-S, 640×640, Tesla V100, FP32). Angle cost Λ penalizes boxes with 45° center misalignment.
- **Our hypothesis**: CIoU ignores the angle between predicted and GT box centers. SIoU's angle cost should help with diagonally-oriented objects (common in COCO: people, bicycles, vehicles at angles).
- **Validation plan**: Same Badger-S protocol. Compare CIoU vs SIoU box loss.
- **Experiment key**: `siou_loss`
- **Our result**: TBD

### [ ] Gaussian DFL Regression

- **Paper**: Li et al., "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection" (NeurIPS 2020) — [arXiv:2006.04388](https://arxiv.org/abs/2006.04388)
- **Repo**: [github.com/implus/GFocal](https://github.com/implus/GFocal) (Apache 2.0)
- **Claimed improvement**: +0.3 AP from DFL component alone on COCO val2017 (GFLV2-ResNet-50, 1333×800, 8×V100, FP32, 24 epochs). The discrete DFL uses reg_max=16 bins; we extend to continuous Gaussian with (μ,σ) per edge.
- **Our hypothesis**: Continuous Gaussian DFL eliminates discretization error and provides per-prediction uncertainty (σ). The uncertainty can be used for NMS weighting or hard example mining.
- **Validation plan**: Compare standard DFL (16 bins) vs Gaussian DFL on Badger-S. Measure mAP AND calibration (ECE).
- **Experiment key**: `gaussian_dfl`
- **Our result**: TBD

---

## Phase 3: Attention Mechanisms

### [ ] CBAM Attention in Backbone

- **Paper**: Woo et al., "CBAM: Convolutional Block Attention Module" (ECCV 2018) — [arXiv:1807.06521](https://arxiv.org/abs/1807.06521)
- **Repo**: [github.com/Jongchan/attention-module](https://github.com/Jongchan/attention-module) (MIT)
- **Claimed improvement**: +0.8% top-1 on ImageNet-1k (ResNet-50, 224×224, 8×Titan Xp, FP32). Channel + spatial attention with ~1% parameter overhead.
- **Our hypothesis**: Channel attention helps the backbone focus on task-relevant features (objects vs background). Spatial attention suppresses noise in non-object regions. Combined, they should improve feature quality at all FPN levels.
- **Validation plan**: Insert CBAM after each C2f block in Badger-S backbone. Train on COCO, compare mAP.
- **Experiment key**: `cbam_attention`
- **Our result**: TBD

### [ ] Hybrid Attention Neck (Swin + Global Tokens)

- **Paper**: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (ICCV 2021) — [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
- **Repo**: [github.com/microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer) (MIT)
- **Claimed improvement**: Swin-L achieves 58.7 box AP on COCO val2017 (HTC++ framework, Mask R-CNN, 3× schedule, 8×V100). Window attention provides O(M²) complexity while maintaining cross-window connections via shifted partitioning.
- **Our hypothesis**: CNN FPN necks have O(depth) receptive field growth. Adding window attention (M=7) to each FPN level gives global context within windows, and global token mixing enables cross-scale reasoning. This should especially help large object detection and cluttered scenes.
- **Validation plan**: Replace PA-FPN neck with AttentionNeck in Badger-S. Measure mAP, FLOPs, and latency. Compare with baseline PA-FPN.
- **Experiment key**: `attention_neck`
- **Our result**: TBD

### [ ] ECA Attention (Lighter CBAM Alternative)

- **Paper**: Wang et al., "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks" (CVPR 2020) — [arXiv:1910.03151](https://arxiv.org/abs/1910.03151)
- **Repo**: [github.com/BangguWu/ECANet](https://github.com/BangguWu/ECANet) (MIT)
- **Claimed improvement**: +0.5% top-1 on ImageNet-1k with <0.01% parameter overhead (ResNet-50, 224×224). Uses 1D conv instead of MLP for channel attention.
- **Our hypothesis**: If CBAM's parameter overhead is too high for Badger-Nano/Tiny, ECA provides similar gains with negligible cost.
- **Validation plan**: Compare CBAM vs ECA on Badger-Nano. Measure mAP and parameter count.
- **Experiment key**: `eca_attention`
- **Our result**: TBD

---

## Phase 4: Training Strategy

### [ ] Close Mosaic Late Training

- **Paper**: Used by YOLOX (Ge et al., 2021) and YOLOv8 (Ultralytics) — not a standalone paper, but a verified practice across 3+ detector families
- **Repo**: [github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) — see `train.py` `no_aug_epochs` parameter
- **Claimed improvement**: YOLOX reports +0.5 AP from closing mosaic at epoch 285/300 on COCO val2017 (YOLOX-S, 640×640, Tesla V100). YOLOv5/v8 consistently apply this.
- **Our hypothesis**: Mosaic-stitched images don't match real image statistics. Objects appear at unnatural scales/contexts. Turning mosaic off for the last 10 epochs lets the model fine-tune on real distributions, improving generalization.
- **Validation plan**: Train Badger-S with mosaic on for epochs 0-290, off for 290-300. Compare with full 300-epoch mosaic training.
- **Experiment key**: `mosaic_close`
- **Our result**: TBD

### [ ] EMA Weight Averaging

- **Paper**: Common practice across SOTA detectors; formalized in "Averaging Weights Leads to Wider Optima and Better Generalization" (Izmailov et al., UAI 2018) — [arXiv:1803.05407](https://arxiv.org/abs/1803.05407)
- **Repo**: Used in YOLOX, D-FINE, DEIM training pipelines
- **Claimed improvement**: YOLOX reports ~+0.3 AP from EMA (decay=0.9999) on COCO val2017. Cost: 2× memory during training, zero inference cost.
- **Our hypothesis**: EMA smooths optimization noise and finds wider minima. It's a nearly-free accuracy gain.
- **Validation plan**: Train Badger-S with/without EMA. Compare validation mAP curves for stability.
- **Experiment key**: `ema_weights`
- **Our result**: TBD

### [ ] Multi-Scale Training

- **Paper**: Standard practice since YOLOv3 (Redmon & Farhadi, 2018) — [arXiv:1804.02767](https://arxiv.org/abs/1804.02767). YOLOv3 used random resize [320, 608]; modern detectors use [0.5×, 1.5×] range.
- **Claimed improvement**: YOLOv3 reports +1.7 AP from multi-scale training on COCO val2017. YOLOX achieves +1.0 AP on YOLOX-S (640×640 base, [448, 832] range, Tesla V100).
- **Our hypothesis**: CNN features are not scale-invariant. Training at multiple resolutions builds robustness to object size variation, especially helping AP_S and AP_L simultaneously.
- **Validation plan**: Train Badger-S with random resize [320, 960]. Compare size-stratified AP.
- **Experiment key**: `multi_scale_training`
- **Our result**: TBD

---

## Phase 5: Architecture Upgrades

### [ ] Larger Resolution (1280)

- **Paper**: Standard practice; D-FINE (Peng et al., 2024) reports results at 640 and 1280
- **Repo**: [github.com/Peterande/D-FINE](https://github.com/Peterande/D-FINE)
- **Claimed improvement**: D-FINE-X: 61.4 mAP at 640 → not reported at 1280. Historical pattern: YOLOv5-X gains ~+3 AP from 640→1280 on COCO val2017 (Tesla V100). Cost: ~4× compute.
- **Our hypothesis**: Higher resolution preserves fine details. Small objects (AP_S) benefit most. The cost is high but worth it for maximum accuracy.
- **Validation plan**: Train Badger-S at 1280, compare with 640 baseline across all AP metrics.
- **Experiment key**: `larger_resolution`
- **Our result**: TBD

### [ ] Badger-XL Scale

- **Paper**: Scaling pattern from EfficientNet (Tan & Le, ICML 2019) — [arXiv:1905.11946](https://arxiv.org/abs/1905.11946). Compound scaling: depth, width, resolution jointly.
- **Repo**: [github.com/google/automl](https://github.com/google/automl) (Apache 2.0)
- **Claimed improvement**: Each doubling of parameters typically yields +2-3 AP on COCO (pattern observed across YOLOv5/v8/v9/X families). Badger-S (11M) → Badger-X (68M) expected: +3-5 AP.
- **Our hypothesis**: More capacity helps, but diminishing returns. The key is whether our architecture improvements compound with scale or saturate.
- **Validation plan**: Train Badger-S, M, L, X variants. Plot mAP vs params to find the Pareto frontier.
- **Experiment key**: `deeper_backbone`
- **Our result**: TBD

---

## Phase 6: Robustness (Beyond Clean COCO)

### [ ] COCO-C Corruption Robustness

- **Paper**: Hendrycks & Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations" (ICLR 2019) — [arXiv:1903.12261](https://arxiv.org/abs/1903.12261)
- **Repo**: [github.com/hendrycks/robustness](https://github.com/hendrycks/robustness) (Apache 2.0)
- **Claimed improvement**: Not a technique — a measurement framework. Apply 15 corruption types × 5 severities, measure mAP degradation. Current SOTA detectors show 20-40% mAP drop under severe corruption.
- **Our hypothesis**: Attention mechanisms (Swin neck, CBAM) may improve robustness by learning to focus on structural features rather than texture. Gaussian DFL's uncertainty estimates could help filter corrupted predictions.
- **Validation plan**: Build COCO-C (apply corruption taxonomy to COCO val2017 images). Measure mAP degradation per corruption category. Compare Badger vs YOLOX vs D-FINE robustness.
- **Experiment key**: `corruption_robustness` (custom benchmark)
- **Our result**: TBD

---

## Rejected Experiments (Document Why)

### [-] Mish Activation

- **Paper**: Misra, "Mish: A Self Regularized Non-Monotonic Activation Function" (BMVC 2020) — [arXiv:1908.08681](https://arxiv.org/abs/1908.08681)
- **Reason rejected**: SiLU (Swish) is mathematically similar to Mish (both smooth, non-monotonic, self-gated) with near-identical performance in modern architectures. YOLOv8 switched from Mish to SiLU with no accuracy loss. Not worth the extra compute.
- **Date**: Pre-decision based on literature evidence; can revisit if ablation shows otherwise.

### [-] Label Smoothing (ε > 0.1)

- **Paper**: Szegedy et al., "Rethinking the Inception Architecture for Computer Vision" (CVPR 2016) — [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)
- **Reason rejected**: Label smoothing helps classification but hurts calibration in dense detection where most predictions are background. The `-4.0` bias initialization already handles the extreme class imbalance. Standard in YOLOX/YOLOv8 to use ε=0.
- **Date**: Pre-decision based on YOLOX/YOLOv8 defaults; can test if class imbalance analysis suggests otherwise.
