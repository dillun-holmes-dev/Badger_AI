#!/usr/bin/env python3
"""
Train Badger on synthetic shapes, then run real inference on unseen images.

This is stricter than checking loss: predictions are decoded, filtered with NMS,
matched to held-out ground truth by class and IoU, then scored.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from scripts.test_learning import SyntheticShapesDataset, collate_fn
from src.models import create_model
from src.training.supermind import SuperMind
from src.utils.metrics import compute_iou


def xywh_to_xyxy(boxes):
    out = boxes.clone()
    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return out.clamp(0, 1)


def nms(boxes, scores, iou_threshold=0.5):
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    keep = []
    order = scores.argsort(descending=True)
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break

        ious = compute_iou(boxes[i].unsqueeze(0), boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]

    return torch.stack(keep)


def decode_predictions(model, images, conf_threshold=0.03, nms_iou=0.5,
                       max_det=50):
    model.eval()
    cls_scores, bbox_preds, raw_reg = model(images, return_raw_reg=True)
    quality_scores = getattr(model, '_last_quality_scores', None)
    device = images.device
    img_h, img_w = images.shape[-2:]
    strides = [8, 16, 32]

    all_scores = []
    all_boxes = []
    all_quality = [] if quality_scores is not None else None
    for scale_idx, (cls, bbox) in enumerate(zip(cls_scores, bbox_preds)):
        b, c, h, w = cls.shape
        stride = strides[scale_idx]

        yy, xx = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij'
        )
        anchors = torch.stack([xx, yy], dim=-1).float().view(1, -1, 2)
        anchors = anchors * stride + stride / 2

        scores = cls.permute(0, 2, 3, 1).reshape(b, -1, c).sigmoid()
        offsets = bbox.permute(0, 2, 3, 1).reshape(b, -1, 4) * stride

        lt = offsets[..., :2]
        rb = offsets[..., 2:]
        xy1 = anchors - lt
        xy2 = anchors + rb
        cxcy = (xy1 + xy2) / 2
        wh = xy2 - xy1
        boxes = torch.cat([cxcy, wh], dim=-1)
        boxes[..., 0] /= img_w
        boxes[..., 1] /= img_h
        boxes[..., 2] /= img_w
        boxes[..., 3] /= img_h
        boxes = xywh_to_xyxy(boxes)

        all_scores.append(scores)
        all_boxes.append(boxes)
        if quality_scores is not None:
            q = quality_scores[scale_idx]
            all_quality.append(q.permute(0, 2, 3, 1).reshape(b, -1, 1).sigmoid())

    scores = torch.cat(all_scores, dim=1)
    boxes = torch.cat(all_boxes, dim=1)

    # Apply quality gating if available
    if all_quality is not None:
        quality = torch.cat(all_quality, dim=1)  # [B, N, 1]
        quality_exp = getattr(model.head, 'quality_exp', 1.0)
        scores = scores * quality.pow(quality_exp)

    batch_results = []
    for b in range(images.shape[0]):
        det_scores, det_classes = scores[b].max(dim=-1)
        keep = det_scores >= conf_threshold
        det_boxes = boxes[b][keep]
        det_scores = det_scores[keep]
        det_classes = det_classes[keep]

        final_boxes = []
        final_scores = []
        final_classes = []
        for cls_id in det_classes.unique():
            cls_mask = det_classes == cls_id
            cls_boxes = det_boxes[cls_mask]
            cls_scores = det_scores[cls_mask]
            cls_keep = nms(cls_boxes, cls_scores, iou_threshold=nms_iou)
            final_boxes.append(cls_boxes[cls_keep])
            final_scores.append(cls_scores[cls_keep])
            final_classes.append(torch.full(
                (len(cls_keep),), int(cls_id), dtype=torch.long, device=device
            ))

        if final_boxes:
            final_boxes = torch.cat(final_boxes)
            final_scores = torch.cat(final_scores)
            final_classes = torch.cat(final_classes)
            if len(final_scores) > max_det:
                top = final_scores.topk(max_det).indices
                final_boxes = final_boxes[top]
                final_scores = final_scores[top]
                final_classes = final_classes[top]
        else:
            final_boxes = torch.zeros((0, 4), device=device)
            final_scores = torch.zeros((0,), device=device)
            final_classes = torch.zeros((0,), dtype=torch.long, device=device)

        batch_results.append((final_boxes, final_scores, final_classes))

    return batch_results


@torch.no_grad()
def evaluate_inference(model, dataloader, device, conf_threshold=0.03,
                       iou_threshold=0.5, nms_iou=0.5, max_det=50):
    tp = fp = fn = 0
    total_iou = 0.0
    matched_count = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        predictions = decode_predictions(
            model, images, conf_threshold=conf_threshold, nms_iou=nms_iou,
            max_det=max_det
        )

        for batch_idx, (pred_boxes, pred_scores, pred_classes) in enumerate(predictions):
            gt = targets[targets[:, 0] == batch_idx]
            gt_classes = gt[:, 1].long()
            gt_boxes = xywh_to_xyxy(gt[:, 2:6])
            matched_gt = set()

            order = pred_scores.argsort(descending=True)
            for pred_idx in order:
                if len(gt_boxes) == 0:
                    fp += 1
                    continue

                same_class = gt_classes == pred_classes[pred_idx]
                candidate_idx = same_class.nonzero(as_tuple=False).flatten()
                candidate_idx = torch.tensor(
                    [int(i) for i in candidate_idx if int(i) not in matched_gt],
                    device=device,
                    dtype=torch.long,
                )
                if candidate_idx.numel() == 0:
                    fp += 1
                    continue

                ious = compute_iou(
                    pred_boxes[pred_idx].unsqueeze(0), gt_boxes[candidate_idx]
                )[0]
                best_iou, rel_idx = ious.max(dim=0)
                if best_iou >= iou_threshold:
                    gt_idx = int(candidate_idx[rel_idx])
                    matched_gt.add(gt_idx)
                    tp += 1
                    total_iou += float(best_iou)
                    matched_count += 1
                else:
                    fp += 1

            fn += len(gt_boxes) - len(matched_gt)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    avg_iou = total_iou / max(1, matched_count)

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_iou': avg_iou,
    }


def parse_args():
    p = argparse.ArgumentParser(description='Synthetic inference verification')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--model', default='badger-n')
    p.add_argument('--device', default='cuda')
    p.add_argument('--head-type', default='quality_decoupled',
                   choices=['decoupled', 'quality_decoupled'])
    p.add_argument('--quality-exp', type=float, default=1.0)
    p.add_argument('--quality-weight', type=float, default=1.0)
    p.add_argument('--num-classes', type=int, default=3)
    p.add_argument('--img-size', type=int, default=128)
    p.add_argument('--train-samples', type=int, default=512)
    p.add_argument('--val-samples', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--conf', type=float, default=0.03)
    p.add_argument('--iou', type=float, default=0.5)
    p.add_argument('--nms-iou', type=float, default=0.5)
    p.add_argument('--max-det', type=int, default=10)
    p.add_argument('--target-f1', type=float, default=0.99)
    p.add_argument('--box-weight', type=float, default=7.5)
    p.add_argument('--cls-weight', type=float, default=2.0)
    p.add_argument('--dfl-weight', type=float, default=1.5)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train_ds = SyntheticShapesDataset(
        num_samples=args.train_samples,
        img_size=args.img_size,
        num_classes=args.num_classes,
        seed=42,
    )
    val_ds = SyntheticShapesDataset(
        num_samples=args.val_samples,
        img_size=args.img_size,
        num_classes=args.num_classes,
        seed=999,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    model = create_model(variant=args.model, num_classes=args.num_classes,
                          head_type=args.head_type, quality_exp=args.quality_exp)
    total, _ = model.count_parameters()
    print(f"Model: {args.model} | head={args.head_type} | params={total:,} | device={device}")

    trainer = SuperMind(
        model,
        train_loader,
        val_loader=val_loader,
        device=str(device),
        project_dir='runs/synthetic_inference',
        use_amp=('cuda' in str(device)),
        use_ema=True,
        use_compile=False,
    )
    history = trainer.fit(
        epochs=args.epochs,
        lr=args.lr,
        num_classes=args.num_classes,
        box_weight=args.box_weight,
        cls_weight=args.cls_weight,
        dfl_weight=args.dfl_weight,
        quality_weight=args.quality_weight,
        preset='fast',
        stability_patience=3,
        stability_reduce_factor=0.75,
    )

    metrics = evaluate_inference(
        trainer.model,
        val_loader,
        device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
    )

    print("\n=== Unseen Inference Result ===")
    print(f"Train loss: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")
    print(f"Val loss:   {history['val_loss'][0]:.4f} -> {history['val_loss'][-1]:.4f}")
    print(f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"F1:        {metrics['f1'] * 100:.2f}%")
    print(f"Avg IoU:   {metrics['avg_iou'] * 100:.2f}%")

    if metrics['f1'] >= args.target_f1:
        print(f"PASS: inference F1 >= {args.target_f1 * 100:.1f}%")
        return 0

    print(f"FAIL: inference F1 < {args.target_f1 * 100:.1f}%")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
