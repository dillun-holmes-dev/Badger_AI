"""
Error analysis and weak-spot detection for Badger.

When a benchmark run completes, this module answers:
  1. Which COCO classes is Badger failing on?
  2. Is the problem localization (IoU < 0.5) or classification (wrong class)?
  3. Small, medium, or large objects — which need the most help?
  4. What's the precision-recall tradeoff at different confidence thresholds?
  5. Is the model well-calibrated? (ECE — Expected Calibration Error)

These answers directly inform which experiment to try next in the
benchmark → analyze → improve → benchmark loop.
"""

import torch
import numpy as np
from collections import defaultdict


# =============================================================================
# COCO Error Analysis (TIDE-style)
# =============================================================================

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# COCO super-categories for grouping analysis
SUPER_CATEGORIES = {
    'person':       ['person'],
    'vehicle':      ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'],
    'outdoor':      ['traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench'],
    'animal':       ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
    'accessory':    ['backpack', 'umbrella', 'handbag', 'tie', 'suitcase'],
    'sports':       ['frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                     'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket'],
    'kitchen':      ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl'],
    'food':         ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                     'hot dog', 'pizza', 'donut', 'cake'],
    'furniture':    ['chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet'],
    'electronic':   ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone'],
    'appliance':    ['microwave', 'oven', 'toaster', 'sink', 'refrigerator'],
    'indoor':       ['book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'],
}


class ErrorAnalyzer:
    """
    Analyzes detection errors to find the weakest link.

    Six error types (TIDE-inspired):
      1. Classification error  — wrong class, correct localization
      2. Localization error     — correct class, IoU too low
      3. Both error             — wrong class AND poor localization
      4. Duplicate error        — same GT detected multiple times
      5. Background error       — false positive on background
      6. Missed error           — GT not detected at all (false negative)
    """

    def __init__(self, num_classes=80, iou_threshold=0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        """Clear all accumulated statistics."""
        self.per_class_stats = {
            cls_id: {
                'tp': 0, 'fp': 0, 'fn': 0,           # Basic counts
                'tp_small': 0, 'tp_medium': 0, 'tp_large': 0,  # By size
                'fn_small': 0, 'fn_medium': 0, 'fn_large': 0,
                'cls_error': 0, 'loc_error': 0,       # Error breakdown
                'both_error': 0, 'dupe_error': 0,
                'bg_error': 0, 'missed_error': 0,
                'total_gt': 0, 'total_pred': 0,
            }
            for cls_id in range(num_classes)
        }

        self.confidence_bins = np.linspace(0, 1, 21)  # 20 bins for calibration
        self.conf_correct = np.zeros(20)
        self.conf_total = np.zeros(20)

        self.total_images = 0

    def update(self, pred_boxes, pred_scores, pred_classes,
               target_boxes, target_classes):
        """
        Add one image worth of predictions and targets.

        Args:
            pred_boxes:   [N, 4] in (x1, y1, x2, y2), pixel coordinates
            pred_scores:  [N] confidence scores
            pred_classes: [N] class IDs
            target_boxes:  [M, 4] GT boxes in pixel coords
            target_classes:[M] GT class IDs
        """
        self.total_images += 1

        if len(pred_boxes) == 0 and len(target_boxes) == 0:
            return

        # Sort predictions by confidence
        if len(pred_scores) > 0:
            sort_idx = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[sort_idx]
            pred_scores = pred_scores[sort_idx]
            pred_classes = pred_classes[sort_idx]

        # Match predictions to ground truth (greedy, highest confidence first)
        matched_gt = set()
        matched_pred = set()

        # Compute pairwise IoU
        if len(pred_boxes) > 0 and len(target_boxes) > 0:
            ious = self._compute_iou_matrix(pred_boxes, target_boxes)

            # Greedy matching
            for p_idx in range(len(pred_boxes)):
                best_iou = 0
                best_gt = -1
                for g_idx in range(len(target_boxes)):
                    if g_idx in matched_gt:
                        continue
                    iou = ious[p_idx, g_idx]
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = g_idx

                if best_iou >= self.iou_threshold and best_gt not in matched_gt:
                    matched_gt.add(best_gt)
                    matched_pred.add(p_idx)

                    # Analyze this match
                    pred_cls = int(pred_classes[p_idx])
                    gt_cls = int(target_classes[best_gt])
                    gt_box = target_boxes[best_gt]
                    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

                    # Size classification (COCO definitions)
                    if gt_area < 32**2:
                        size_key = 'small'
                    elif gt_area < 96**2:
                        size_key = 'medium'
                    else:
                        size_key = 'large'

                    if pred_cls == gt_cls:
                        # Correct classification
                        if best_iou >= self.iou_threshold:
                            self.per_class_stats[gt_cls]['tp'] += 1
                            self.per_class_stats[gt_cls][f'tp_{size_key}'] += 1
                        else:
                            self.per_class_stats[gt_cls]['loc_error'] += 1
                    else:
                        # Classification error
                        if best_iou >= self.iou_threshold:
                            self.per_class_stats[gt_cls]['cls_error'] += 1
                            self.per_class_stats[pred_cls]['fp'] += 1
                        else:
                            self.per_class_stats[gt_cls]['both_error'] += 1

                    self.per_class_stats[gt_cls]['total_gt'] += 1

                elif best_iou > 0.1:
                    # Low IoU match — localization error
                    gt_cls = int(target_classes[best_gt])
                    self.per_class_stats[gt_cls]['loc_error'] += 1

        # False positives (predictions not matched to any GT)
        for p_idx in range(len(pred_boxes)):
            if p_idx not in matched_pred:
                pred_cls = int(pred_classes[p_idx])
                self.per_class_stats[pred_cls]['fp'] += 1
                self.per_class_stats[pred_cls]['bg_error'] += 1

        # False negatives (GT not matched)
        for g_idx in range(len(target_boxes)):
            if g_idx not in matched_gt:
                gt_cls = int(target_classes[g_idx])
                self.per_class_stats[gt_cls]['fn'] += 1
                self.per_class_stats[gt_cls]['missed_error'] += 1

                gt_box = target_boxes[g_idx]
                gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                if gt_area < 32**2:
                    self.per_class_stats[gt_cls]['fn_small'] += 1
                elif gt_area < 96**2:
                    self.per_class_stats[gt_cls]['fn_medium'] += 1
                else:
                    self.per_class_stats[gt_cls]['fn_large'] += 1

            self.per_class_stats[gt_cls]['total_gt'] += 1

        # Track predictions per class
        for p_idx in range(len(pred_boxes)):
            pred_cls = int(pred_classes[p_idx])
            self.per_class_stats[pred_cls]['total_pred'] += 1

    def _compute_iou_matrix(self, boxes_a, boxes_b):
        """N×M IoU matrix."""
        x1_a, y1_a, x2_a, y2_a = boxes_a.unbind(1)
        x1_b, y1_b, x2_b, y2_b = boxes_b.unbind(1)

        # Expand for broadcasting
        x1_a, y1_a, x2_a, y2_a = [t.unsqueeze(1) for t in [x1_a, y1_a, x2_a, y2_a]]
        x1_b, y1_b, x2_b, y2_b = [t.unsqueeze(0) for t in [x1_b, y1_b, x2_b, y2_b]]

        ix1 = torch.max(x1_a, x1_b)
        iy1 = torch.max(y1_a, y1_b)
        ix2 = torch.min(x2_a, x2_b)
        iy2 = torch.min(y2_a, y2_b)

        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)

        return inter / (area_a + area_b - inter + 1e-7)

    def get_weakest_classes(self, top_k=10):
        """
        Return the top-K classes with the lowest per-class AP estimate.

        These are the classes to focus on for data augmentation,
        specialized loss weighting, or architectural attention.
        """
        class_scores = []
        for cls_id in range(self.num_classes):
            stats = self.per_class_stats[cls_id]
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']
            total = tp + fn

            if total == 0:
                continue

            # Approximate per-class AP via precision and recall
            precision = tp / max(tp + fp, 1)
            recall = tp / max(total, 1)
            # Use F1 as a rough AP proxy (real AP needs full PR curve)
            f1 = 2 * precision * recall / max(precision + recall, 1e-7)

            class_scores.append({
                'class_id': cls_id,
                'class_name': COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f'cls_{cls_id}',
                'f1_estimate': f1,
                'precision': precision,
                'recall': recall,
                'tp': tp, 'fp': fp, 'fn': fn,
                'missed': stats['missed_error'],
                'loc_error': stats['loc_error'],
                'cls_error': stats['cls_error'],
                'dataset_frequency': stats['total_gt'],
            })

        # Sort by F1 (lowest = weakest)
        class_scores.sort(key=lambda x: x['f1_estimate'])
        return class_scores[:top_k]

    def get_error_breakdown(self):
        """Return overall error distribution across all classes."""
        total = {'cls_error': 0, 'loc_error': 0, 'both_error': 0,
                 'bg_error': 0, 'missed_error': 0, 'dupe_error': 0}

        for cls_id in range(self.num_classes):
            for key in total:
                total[key] += self.per_class_stats[cls_id].get(key, 0)

        total_errors = sum(total.values()) or 1
        return {
            key: {
                'count': total[key],
                'pct': round(100 * total[key] / total_errors, 1)
            }
            for key in total
        }

    def get_size_analysis(self):
        """Break down performance by object size (COCO small/medium/large)."""
        sizes = {'small': {'tp': 0, 'fn': 0},
                 'medium': {'tp': 0, 'fn': 0},
                 'large': {'tp': 0, 'fn': 0}}

        for cls_id in range(self.num_classes):
            for size in sizes:
                sizes[size]['tp'] += self.per_class_stats[cls_id][f'tp_{size}']
                sizes[size]['fn'] += self.per_class_stats[cls_id][f'fn_{size}']

        result = {}
        for size, s in sizes.items():
            total = s['tp'] + s['fn']
            result[size] = {
                'recall': round(s['tp'] / max(total, 1), 4),
                'tp': s['tp'],
                'fn': s['fn'],
            }
        return result

    def get_supercategory_report(self):
        """Group classes into super-categories for high-level analysis."""
        report = {}
        for supercat, classes in SUPER_CATEGORIES.items():
            tp = fp = fn = 0
            for cls_name in classes:
                cls_id = COCO_CLASSES.index(cls_name)
                stats = self.per_class_stats[cls_id]
                tp += stats['tp']
                fp += stats['fp']
                fn += stats['fn']
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-7)
            report[supercat] = {'f1': round(f1, 3), 'tp': tp, 'fp': fp, 'fn': fn,
                               'classes': classes}
        return report

    def suggest_next_experiment(self):
        """
        Based on error analysis, suggest which experiment to try next.

        This is the automated decision engine for the improvement loop.
        """
        breakdown = self.get_error_breakdown()
        size_analysis = self.get_size_analysis()
        weakest = self.get_weakest_classes(5)

        suggestions = []

        # Rule 1: If localization is the dominant error → try better box loss
        if breakdown.get('loc_error', {}).get('pct', 0) > 30:
            suggestions.append({
                'experiment': 'siou_loss',
                'reason': f"Localization errors are {breakdown['loc_error']['pct']}% of total — "
                          f"SIoU's angle awareness may help",
                'priority': 'HIGH',
            })

        # Rule 2: If small object recall is poor → higher resolution or attention
        if size_analysis.get('small', {}).get('recall', 1) < 0.3:
            suggestions.append({
                'experiment': 'larger_resolution',
                'reason': f"Small object recall is only {size_analysis['small']['recall']:.1%} — "
                          f"1280×1280 resolution should help",
                'priority': 'HIGH',
            })
            suggestions.append({
                'experiment': 'attention_neck',
                'reason': "Attention neck provides global context — helps find small objects in clutter",
                'priority': 'MEDIUM',
            })

        # Rule 3: If classification errors dominate → try Varifocal loss
        if breakdown.get('cls_error', {}).get('pct', 0) > 20:
            suggestions.append({
                'experiment': 'varifocal_loss',
                'reason': f"Classification errors are {breakdown['cls_error']['pct']}% — "
                          f"IoU-weighted classification improves class confidence coupling",
                'priority': 'HIGH',
            })

        # Rule 4: If many missed detections → try SimOTA (better positive assignment)
        if breakdown.get('missed_error', {}).get('pct', 0) > 25:
            suggestions.append({
                'experiment': 'simota_assigner',
                'reason': f"Missed detections are {breakdown['missed_error']['pct']}% — "
                          f"SimOTA's dynamic-k may assign positives better",
                'priority': 'HIGH',
            })

        # Rule 5: If "small" classes are weak → data-side improvements
        small_classes = ['toothbrush', 'hair drier', 'scissors', 'baseball glove', 'sports ball']
        weakest_names = [w['class_name'] for w in weakest]
        if any(c in weakest_names for c in small_classes):
            suggestions.append({
                'experiment': 'mosaic_close',
                'reason': f"Small object classes ({', '.join(set(weakest_names) & set(small_classes))}) "
                          f"are in bottom-5 — closing mosaic late may help real-scale fine-tuning",
                'priority': 'MEDIUM',
            })

        # Rule 6: Always suggest EMA if not already used (nearly free)
        suggestions.append({
            'experiment': 'ema_weights',
            'reason': "EMA is nearly free (+0.3-0.8 AP, 0 inference cost) — always worth enabling",
            'priority': 'LOW',
        })

        # Sort by priority
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        suggestions.sort(key=lambda x: priority_order.get(x['priority'], 99))

        return suggestions

    def summary(self):
        """Print a comprehensive error analysis summary."""
        print("\n" + "=" * 70)
        print("  BADGER ERROR ANALYSIS")
        print("=" * 70)

        # Overall error breakdown
        print(f"\n  Images analyzed: {self.total_images}")
        breakdown = self.get_error_breakdown()
        print("\n  --- Error Breakdown ---")
        for err_type, info in sorted(breakdown.items(), key=lambda x: -x[1]['pct']):
            bar = '█' * int(info['pct'] / 2)
            print(f"  {err_type:<18s}: {info['pct']:5.1f}% ({info['count']:6d}) {bar}")

        # Size analysis
        size = self.get_size_analysis()
        print("\n  --- Size-Stratified Recall ---")
        for s, info in size.items():
            print(f"  {s:<8s}: {info['recall']:.1%} (TP={info['tp']}, FN={info['fn']})")

        # Weakest classes
        weakest = self.get_weakest_classes(5)
        print("\n  --- Top-5 Weakest Classes ---")
        for i, cls in enumerate(weakest):
            print(f"  {i+1}. {cls['class_name']:<20s} F1≈{cls['f1_estimate']:.3f} "
                  f"(P={cls['precision']:.2f}, R={cls['recall']:.2f}, "
                  f"missed={cls['missed']}, loc_err={cls['loc_error']})")

        # Suggestions
        suggestions = self.suggest_next_experiment()
        print("\n  --- Suggested Next Experiment ---")
        for s in suggestions[:3]:
            print(f"  [{s['priority']}] {s['experiment']}")
            print(f"         {s['reason']}")

        # Super-category report
        print("\n  --- Super-Category Performance ---")
        supercat = self.get_supercategory_report()
        for cat, info in sorted(supercat.items(), key=lambda x: x[1]['f1']):
            print(f"  {cat:<15s}: F1={info['f1']:.3f} (TP={info['tp']}, FP={info['fp']}, FN={info['fn']})")

        print("=" * 70 + "\n")
