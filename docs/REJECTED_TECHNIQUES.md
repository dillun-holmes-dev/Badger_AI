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

## Pre-Rejected (Based on Literature Evidence)

These techniques were NOT implemented because published evidence suggests they
won't help Badger specifically, OR they've been superseded by better alternatives.
We document them here so we don't waste time reconsidering.

### Mish Activation — Pre-Rejected

- **Paper**: Misra, "Mish: A Self Regularized Non-Monotonic Activation Function" (BMVC 2020) — arXiv:1908.08681
- **Reason**: SiLU (Swish) is mathematically nearly identical to Mish (both smooth, non-monotonic, self-gated). YOLOv8 explicitly switched from Mish to SiLU with no accuracy loss. Not worth the 5% compute overhead.
- **Date**: June 2026

### Label Smoothing (ε > 0.1) — Pre-Rejected

- **Paper**: Szegedy et al., "Rethinking the Inception Architecture" (CVPR 2016) — arXiv:1512.00567
- **Reason**: Helps classification but hurts calibration in dense detection where >99% of predictions are background. The -4.0 bias initialization already handles class imbalance. YOLOX and YOLOv8 both use ε=0 by default. We can test if ablation shows benefit, but it's low priority.
- **Date**: June 2026
