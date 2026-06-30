# ARCHITECTURE PROPOSAL — Badger

> **Principle**: Not a copy of any existing detector. A new combination of
> individually-published, individually-proven techniques, selected for
> structural compatibility and measured by benchmark.

---

## 1. Design Philosophy

From SURVEY.md, the key insight is:

> **The 4+ AP gap between 61% SOTA and 57% mid-tier comes from DETR-style
> matching + transformer necks, NOT from better backbones or more data.**

However, full DETR/transformer architectures (RT-DETR, D-FINE) bring O(N²)
complexity and 30-100% inference slowdown. Our goal is to capture MOST of the
DETR advantage while staying closer to CNN efficiency.

### Our claim: Hybrid CNN-Transformer with dynamic matching.

---

## 2. Proposed Architecture

### Backbone: CSPDarknet with Structural Reparameterization

**Why CSPDarknet**: Proven across YOLOv5/X/v8/v9. Efficient cross-stage partial connections provide gradient highways (`∂L/∂x = ∂L/∂y(1+∂F/∂x)`). Beats plain Darknet by 1 AP with 10% fewer params.

**Why add RepConv**: Train with multi-branch (3×3 + 1×1 + identity) for better optimization, deploy as single conv. +1-2 AP with ZERO inference cost. This is the "free lunch" of modern detection.

**Scale**: Compound scaling (width × depth) per EfficientNet, not arbitrary. Five variants: Nano (1.9M) through XL (71M).

### Neck: BiFPN with Optional Lightweight Transformer Blocks

**Why BiFPN over PA-FPN**: Learned per-input weights (`Σ(w_i·f_i) / Σw_i`) at each fusion node are strictly more expressive than concatenation. Fewer nodes (removes single-input nodes). -20% neck params vs PA-FPN, +0.5-1.5 AP.

**Why optional transformer blocks**: The survey shows transformer necks add +2-4 AP but cost 20-50% inference time. We make this a CONFIG FLAG — users choose the speed/accuracy tradeoff. When enabled, use:
- **Swin window attention** (M=7): O(M²) not O(H²W²), cross-window via shifted blocks
- **Global token mixer** (64 tokens): Cross-scale reasoning without full cross-attention on all spatial positions

This hybrid design means Badger can operate at BOTH the 61+ AP tier (with attention) AND the 55-57 AP tier (CNN-only neck), depending on deployment needs.

### Head: Decoupled + Anchor-Free + DFL + IoU-Aware

**Why decoupled**: Classification and regression benefit from different feature representations. +1-2 AP. (YOLOX, YOLOv8, D-FINE all use this.)

**Why anchor-free**: Eliminates anchor hyperparameters. Predict (l,t,r,b) offsets from grid points — this parameterization is isotropic (error in left is symmetric to error in right). +1.5 AP, -30% head params.

**Why DFL**: Distribution over 16 edge bins gives finer localization than scalar regression. +0.3-0.8 AP. Continuous Gaussian DFL variant eliminates discretization error.

**Why IoU-aware**: Varifocal loss weights classification by IoU quality. Well-localized boxes contribute more to class learning. +0.5-1.5 AP.

### Label Assignment: SimOTA (Primary) + TAL (Fallback)

**Why SimOTA**: Dynamic-k per GT (k = Σ top-K IoU per GT) adapts to object size. Small objects get fewer positives, large objects get more. This is more nuanced than TAL's fixed top-k=13.

**Why TAL as fallback**: SimOTA has higher compute cost (O(N×topk) per GT). For users who prioritize training speed, TAL provides 90% of the benefit at near-zero cost.

Both are config flags — measure which works best for your specific dataset.

### Training Recipe (Default)

| Technique | Why | Expected Gain |
|-----------|-----|---------------|
| Mosaic (epochs 1-280) | Strong regularization, object diversity | +1-2 AP |
| Close mosaic (epochs 281-300) | Fine-tune on real statistics | +0.5-1.5 AP |
| Multi-scale (random [0.5×, 1.5×]) | Robustness to size variation | +0.5-1.5 AP |
| Cosine LR schedule | Smooth convergence | +0.2-0.5 AP |
| EMA (decay=0.9999) | Better generalization, wider minima | +0.3-1.0 AP |
| Mixed precision (AMP) | 2× faster training, 50% less memory | Speed only |

### Inference Optimizations (Always Applied)

| Technique | Gain |
|-----------|------|
| Conv+BN fusion | 5-15% faster |
| FP16 inference | 1.5-2× faster |
| ONNX export | Cross-platform |
| INT8 quantization (optional) | 2-4× faster, 4× smaller |
| RepConv reparameterization | 10-20% faster |

---

## 3. Why This Combination (and NOT others)

### What We INCLUDED and Why

| Technique | Included? | Reason |
|-----------|-----------|--------|
| Decoupled head | ✅ YES | Universal SOTA practice, +1-2 AP, no downside |
| Anchor-free | ✅ YES | Simpler, faster, better for small objects |
| DFL (distribution focal loss) | ✅ YES | Finer localization, proven across YOLOv8/D-FINE |
| SimOTA | ✅ YES (default) | Dynamic-k adapts to object size |
| BiFPN | ✅ YES | More expressive than PA-FPN, fewer params |
| RepConv | ✅ YES (training only) | Free inference gain |
| Varifocal loss | ✅ YES | Couples cls to localization quality |
| Transformer neck | ✅ CONFIG FLAG | Users choose speed vs accuracy |
| EMA | ✅ YES | Nearly free accuracy gain |
| Mosaic + close late | ✅ YES | Standard SOTA training practice |

### What We EXCLUDED and Why

| Technique | Excluded? | Reason |
|-----------|-----------|--------|
| Full DETR-style encoder | ❌ NO (default off) | 30-100% slower; we provide lightweight attention as config option instead |
| Hungarian matcher | ❌ NO | O(N²) complexity; SimOTA gives 80% of benefit at 10% of cost |
| GELAN backbone | ❌ NO | Higher implementation complexity; CSPDarknet+RepConv gives similar benefit with simpler code |
| YOLOv9's PGI (programmable gradient info) | ❌ NO | Only proven in YOLOv9's own paper; needs independent verification before adoption |
| NAS-found architecture | ❌ NO | Not reproducible without the NAS pipeline; pre-defined BiFPN is the searched architecture distilled to its essence |
| Model soup / SWA | ❌ NOT YET | Promising but adds training complexity; evaluate in Phase 2 if needed |
| Knowledge distillation | ❌ CONFIG OPTION | Implemented in `src/utils/distill.py` but not default; activate once teacher model trained |

---

## 4. Expected Performance (Hypothesis — Must Be Verified)

Based on SURVEY.md technique-level gains:

| Badger Variant | Params | Expected mAP (Clean COCO) | Target SOTA to Beat |
|---------------|--------|--------------------------|---------------------|
| Badger-N | 1.9M | 30-35% | YOLOv9-T (41.8%) — **structural advantage at this size tier** |
| Badger-S | 7.4M | 45-50% | YOLOv9-S (50.5%) |
| Badger-M | 18.4M | 52-57% | EC-M (58.4%) |
| Badger-L | 38.4M | 56-60% | D-FINE-L (60.0%) |
| Badger-X | 71.3M | 58-62% | D-FINE-X (61.4%) |

**Caveat**: These are projections based on published ablation studies for individual techniques. The ACTUAL mAP depends on implementation quality, hyperparameter tuning, and technique interactions. We will not claim any number until SCOREBOARD_HISTORY.json proves it.

---

## 5. Comparability Caveats

- All mAP numbers cited from visionanalysis.org use COCO val2017, PyTorch FP32, unless otherwise stated
- Latency numbers vary significantly by hardware (Jetson Orin Nano vs A100 vs T4)
- Training cost depends on GPU type, batch size, and number of epochs
- Our Badger variants have DIFFERENT parameter counts from the SOTA peers — we compare at the closest parameter tier, not exact match
- The "DETR gap" claim is based on correlation, not causation — individual technique ablations are needed to confirm

---

## 6. Next Steps (Phase 1+)

1. **Phase 1**: Implement benchmark harness + train Badger-S baseline on COCO
2. **Phase 2**: Iterate through SURVEY.md techniques with accept/reject loop
3. **Phase 3**: Stress test on corruptions + edge cases
4. **Stopping**: When 3 consecutive techniques rejected OR SOTA matched
