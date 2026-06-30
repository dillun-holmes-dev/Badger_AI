# Badger Technical Reference — Grounding Document

> **Purpose**: This document grounds every architectural decision in Badger.
> Every layer, every loss function, every design choice must trace back to
> either the math below or a specific paper. No "research suggests" without a DOI.

---

## 1. Convolution — The Atom

For input $X \in \mathbb{R}^{H \times W \times C_{in}}$, kernel $K \in \mathbb{R}^{k \times k \times C_{in} \times C_{out}}$:

$$Y_{i,j,o} = \sum_{c} \sum_{m} \sum_{n} X_{i \cdot s + m - p,\, j \cdot s + n - p,\, c} \cdot K_{m,n,c,o} + b_o$$

**Badger implementation**: `src/models/blocks.py:Conv` — uses `nn.Conv2d` (im2col+GEMM via cuDNN).  
**Paper**: Chellapilla, Puri & Simard (2006) — https://hal.science/inria-00112631/document

### Backprop: $\frac{\partial L}{\partial K}$ = conv(input, upstream_grad)  
$\frac{\partial L}{\partial X}$ = full-conv(upstream_grad, rot180(K))

**Badger verification**: `scripts/test_all.py:test_conv` — gradient flow confirmed.  
**Paper**: Dumoulin & Visin (2016) — https://arxiv.org/abs/1603.07285

---

## 2. Badger's Layer Stack (Section-by-Section Audit)

| Layer Type | Badger Module | Standard? | Verified? |
|-----------|--------------|-----------|-----------|
| **Stem** (early convs) | `backbone.py:stem` — 3×3 convs, stride=2 | ✓ VGG principle (small kernels, stacking) | ✓ test_backbone |
| **Activation** | SiLU (Swish) — `blocks.py:Conv.act` | ✓ YOLOv8 default, smooth non-monotonic | ✓ test_conv |
| **Normalization** | BatchNorm2d — fused into Conv block | ✓ Ioffe & Szegedy 2015 | ✓ test_conv |
| **Residual** | Bottleneck: `y = x + F(x)` | ✓ He et al. 2015 | ✓ test_bottleneck |
| **CSP connection** | C2f: split→process→concat | ✓ Cross-stage partial (gradient highway) | ✓ test_c2f |
| **Pooling** | SPPF: sequential 5×5 max pools | ✓ Spatial pyramid (fast variant) | ✓ test_sppf |
| **Receptive field** | P3=8×, P4=16×, P5=32× stride | ✓ YOLO standard 3-scale pyramid | ✓ test_backbone |
| **Neck** | PA-FPN: top-down + bottom-up paths | ✓ Path aggregation (YOLOv8) | ✓ test_neck |
| **Head** | Decoupled: separate cls/reg branches | ✓ YOLOX innovation | ✓ test_head |
| **DFL** | Softmax-weighted sum over 16 bins | ✓ Li et al. GFL 2020 | ✓ test_dfl |
| **Attention** | Swin window attn + global tokens | ✓ Liu et al. 2021 | ✓ test_attention |

---

## 3. Loss Functions — Exact Math

### Cross-Entropy (classification)
$L_{CE} = -\sum_c y_c \log(\hat{y}_c)$  
**Badger**: `badger_loss.py` — BCEWithLogitsLoss

### Focal Loss (handles class imbalance)
$L_{FL} = -\alpha_t (1-p_t)^\gamma \log(p_t)$  
**Badger**: Used implicitly via BCE + label weighting  
**Paper**: Lin et al. 2017 — https://arxiv.org/abs/1708.02002

### CIoU Loss (box regression)
$L_{CIoU} = 1 - IoU + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$  
**Badger**: `badger_loss.py:ciou_loss`  
**Verified**: Perfect match → loss ≈ 0 ✓

### DFL (distribution focal loss)
$L_{DFL} = -\sum_n [(y_{n+1}-y)\log(S_n) + (y-y_n)\log(S_{n+1})]$  
**Badger**: `badger_loss.py:dfl_loss`  
**Paper**: Li et al. GFL 2020 — https://arxiv.org/abs/2006.04388

### Varifocal Loss (IoU-weighted classification)
$L_{VFL} = -q \cdot \log(p)$ where $q$ = IoU for positives  
**Badger**: `advanced_losses.py:VarifocalLoss`  
**Paper**: Zhang et al. 2020 — https://arxiv.org/abs/2008.13367

### SIoU Loss (angle-aware)
$\Lambda = 1 - 2\sin^2(2\alpha)$ angle cost  
**Badger**: `advanced_losses.py:siou_loss`  
**Paper**: Gevorgyan 2022 — https://arxiv.org/abs/2205.12740

---

## 4. Label Assignment — The Critical Step

### TAL (YOLOv8)
Alignment = $cls^\alpha \times IoU^\beta$, fixed top-k per GT  
**Badger**: `badger_loss.py:TaskAlignedAssigner`

### SimOTA (YOLOX)
Dynamic k = $\lceil \sum \text{top-k IoU} \rceil$, cost = $cls\_cost + \lambda \cdot reg\_cost$  
**Badger**: `badger_loss.py:SimOTAAssigner`  
**Paper**: Ge et al. 2021 — https://arxiv.org/abs/2107.08430

---

## 5. Attention Math

### Self-Attention (Vaswani et al. 2017)
$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

### Window Attention (Swin, Liu et al. 2021)
Complexity: $O(M^2 \cdot \frac{HW}{M^2} \cdot C) = O(M^2 HW C)$ vs global $O(H^2 W^2 C)$  
**Badger**: `attention.py:WindowAttention` — M=7 windows, shifted alternating blocks

### Global Token Mixer
Cross-attention between learnable tokens and spatial features  
**Badger**: `attention.py:GlobalTokenMixer`

---

## 6. Deployment Optimizations

| Technique | Badger Module | Expected Gain |
|-----------|--------------|---------------|
| FP16 inference | `quantize.py:convert_to_fp16` | 2× speed, 50% memory |
| Conv+BN fusion | `quantize.py:fuse_all_conv_bn` | 5-15% speedup |
| ONNX export | `quantize.py:export_onnx` | Cross-platform |
| INT8 quantization | `quantize.py:quantize_int8` | 2-4× speed, 4× smaller |
| TensorRT | `quantize.py:export_tensorrt` | Max NVIDIA perf |
| Channel pruning | `quantize.py:prune_channels` | ~30% smaller |

---

## 7. Initialization Standards

- **Kaiming/He**: $W \sim \mathcal{N}(0, 2/n_{in})$ — correct for SiLU (similar to ReLU in variance behavior)
- **Bias**: Classification heads initialized to $-4.0$ (extreme class imbalance in dense detection)
- **DFL**: Uniform softmax (equal bin probabilities at initialization)
- **Paper**: He et al. 2015 — https://arxiv.org/abs/1502.01852

---

## 8. Robustness (Hendrycks & Dietterich 2019 Taxonomy)

Badger's COCO-C corruption test covers:
- **Noise**: Gaussian, shot, impulse
- **Blur**: defocus, motion
- **Weather**: brightness, fog
- **Digital**: contrast, JPEG

**Paper**: https://arxiv.org/abs/1903.12261  
**Repo**: https://github.com/hendrycks/robustness

---

## 9. Full Paper Index

Reference `RESEARCH.md` and `IMPROVEMENT_BACKLOG.md` for the complete annotated bibliography with exact benchmark conditions per claim.
