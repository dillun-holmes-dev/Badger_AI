# CHANGELOG

> Every accepted change must show full before/after on ALL three axes:
> accuracy (mAP), speed (latency), and training time.
> No claim without a SCOREBOARD_HISTORY.json entry backing it.

---

## Format

```markdown
## [YYYY-MM-DD] Technique Name — ACCEPTED/REJECTED

**Commit**: `abc123`

**Benchmark**:
| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| mAP@0.5:0.95 | X.X% | Y.Y% | +Z.Z |
| AP50 | X.X% | Y.Y% | +Z.Z |
| AP75 | X.X% | Y.Y% | +Z.Z |
| AP_S (small) | X.X% | Y.Y% | +Z.Z |
| AP_M (medium) | X.X% | Y.Y% | +Z.Z |
| AP_L (large) | X.X% | Y.Y% | +Z.Z |
| Latency p50 | Xms | Yms | +Zms |
| Latency p95 | Xms | Yms | +Zms |
| Training time | Xh | Yh | +Zh |
| Params | XM | YM | +ZM |
| GFLOPs | X | Y | +Z |

**Hardware**: [GPU model, precision, batch size]
**Dataset**: COCO val2017 / train2017
**Run command**: `python scripts/train.py --config ...`

---

## Entries

_(No entries yet — to be populated during Phase 2 iteration.)_
