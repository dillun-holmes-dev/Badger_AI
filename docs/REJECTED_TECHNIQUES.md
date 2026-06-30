# REJECTED TECHNIQUES

> **Rule**: Every technique that was tried and failed goes here. Log the exact
> reason it failed — root cause, not "it didn't work." This is the most
> valuable log in the repo. Never delete entries.

---

## Format

```markdown
### [Technique Name] — Rejected YYYY-MM-DD

- **Paper/Source**: [Citation with link]
- **What we tried**: [Exact implementation — what file, what config flag]
- **Expected gain**: [What the paper claimed and why we thought it would transfer]
- **Actual result**:
  - mAP before: X.X% (COCO val2017, [hardware], [precision])
  - mAP after: X.X%
  - Latency before/after: Xms → Yms
  - Training time before/after: Xh → Yh
- **Root cause of failure**: [Specific measurement, not opinion]
- **Commit**: [git hash where it was reverted]
- **Lessons**: [What we learned — how to spot this failure mode earlier next time]
```

---

## Entries

_(No entries yet — Phase 2 iteration will populate this.)_

---

## Pre-Rejected (Based on Literature Evidence + Incompatibility Analysis)

These techniques were NOT implemented because published evidence suggests they
won't help Badger, OR they conflict with other chosen techniques. We document
them here so we don't waste iteration cycles reconsidering.

### Mish Activation — Pre-Rejected 2026-06-30

- **Paper**: Misra, "Mish: A Self Regularized Non-Monotonic Activation Function" (BMVC 2020) — arXiv:1908.08681
- **Mathematical form**: Mish(x) = x × tanh(softplus(x))
- **Why we rejected**: 
  1. SiLU(x) = x × σ(x) is functionally almost identical. Both are smooth, non-monotonic, self-gated, unbounded above, bounded below. The difference is at the 3rd decimal place in activation values.
  2. YOLOv8 (Ultralytics) explicitly switched from Mish (used in YOLOv5) to SiLU with no accuracy change reported. This is the closest apples-to-apples evidence available.
  3. Computational cost: tanh(softplus(x)) is ~3-5× more expensive than σ(x) due to the double nonlinearity path.
- **Risk of re-testing**: Low — if someone claims Mish matters, test on Badger-S with otherwise identical config. Expected: ≤0.1% AP difference, not worth the compute.
- **Date**: June 2026

### Label Smoothing (ε > 0.1) for Detection — Pre-Rejected 2026-06-30

- **Paper**: Szegedy et al., "Rethinking the Inception Architecture" (CVPR 2016) — arXiv:1512.00567
- **Why we rejected**:
  1. Originally designed for ImageNet classification (1000 balanced classes). Detection has extreme foreground/background imbalance (~10 objects vs 8400 predictions per image).
  2. Label smoothing assigns probability mass to all classes, including the 80-1000 negative classes. This dilutes the already-weak signal for rare classes.
  3. The detection head already addresses this via bias initialization: cls output bias = -4.0 → sigmoid(-4) ≈ 0.018 → only ~1.8% of predictions activate initially, matching the foreground ratio.
  4. YOLOX and YOLOv8 both use ε=0 for detection. When they use label smoothing, it's only in the classification pretraining phase, not detection fine-tuning.
- **Known risk**: If we later pretrain the backbone on ImageNet, ε=0.1 there IS standard. Just don't carry it over to detection.
- **Date**: June 2026

### Full DETR-Style Hungarian Matching — Pre-Rejected 2026-06-30

- **Paper**: Carion et al., "End-to-End Object Detection with Transformers" (ECCV 2020) — arXiv:2005.12872
- **Why we rejected**:
  1. Hungarian algorithm complexity: O(N³) with N object queries. SimOTA gives O(N×M×q) where q=10 and M is number of GTs — effectively O(N) in practice.
  2. RT-DETR shows Hungarian matching can be fast with deformable attention, but requires transformer encoder which we exclude by default (config flag only).
  3. Research question: is the 2-3 AP gap between SimOTA and Hungarian matching due to the matching algorithm itself, or to the transformer architecture that typically accompanies it? No one has isolated these two variables. We take the position that SimOTA + our attention neck (config flag) gives us the benefit of both at lower cost.
- **Date**: June 2026

### GELAN Backbone (YOLOv9) — Deferred, Not Rejected

- **Paper**: Wang et al., "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information" (arXiv 2024) — arXiv:2402.13616
- **Status**: NOT rejected. Deferred because GELAN + PGI is the most complex architectural change possible, and we need baseline numbers first. YOLOv9 claims +2 AP over CSPDarknet, but:
  1. This gain hasn't been independently reproduced outside the YOLOv9 paper.
  2. The PGI (Programmable Gradient Information) component is tightly coupled to GELAN — you can't test one without the other, so it's a package deal.
  3. Our GhostC2f backbone + LightweightHead gives efficiency gains that partially overlap with what GELAN claims (more efficient gradient flow).
- **Plan**: Revisit after Phase 2 baseline is established. If the accuracy gap to SOTA is >3 AP, implement GELAN as an experiment.
- **Date**: June 2026
