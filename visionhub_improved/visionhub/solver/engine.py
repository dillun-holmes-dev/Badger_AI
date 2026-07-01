"""
DETRPose: Real-time end-to-end transformer model for multi-person pose estimation
Copyright (c) 2025 The DETRPose Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM (https://github.com/Intellindust-AI-Lab/DEIM/)
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import math
import sys
import time
from typing import Iterable

import torch
from ..misc import logger as utils
from ..misc import dist_utils

GIGABYTE = 1024 ** 3

def _sync_if_cuda(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _map_prediction_label(coco_evaluator, label):
    if coco_evaluator is not None and hasattr(coco_evaluator, "map_label_to_category_id"):
        return coco_evaluator.map_label_to_category_id(label)
    return int(label)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    batch_size:int, grad_accum_steps:int, 
                    device: torch.device, epoch: int, max_norm: float = 0, writer=None,
                    lr_scheduler=None, warmup_scheduler=None, ema=None, args=None):
    device_type = device.type
    amp_enabled = bool(getattr(args, 'amp', False)) and device_type == 'cuda'
    grad_accum_steps = max(1, int(getattr(args, 'grad_accum_steps', grad_accum_steps)))
    scaler = torch.amp.GradScaler(device_type, enabled=amp_enabled)
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch + 1)  # Display as 1-indexed
    print_freq = args.print_freq
    debug_first_batches = max(0, int(getattr(args, 'debug_first_batches', 1)))
    
    print("Grad accum steps: ", grad_accum_steps, flush=True)
    print("Batch size/GPU: ", batch_size, flush=True)
    print("Total batch size: ", batch_size * dist_utils.get_world_size(), flush=True)
    if debug_first_batches > 0:
        print("[train info] Waiting for first batch from DataLoader...", flush=True)

    optimizer.zero_grad(set_to_none=True)

    data_wait_start = time.perf_counter()
    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        debug_this_batch = i < debug_first_batches
        iter_start = time.perf_counter()
        if debug_this_batch:
            target_counts = [int(t.get("labels", torch.empty(0)).numel()) for t in targets]
            print(
                "[train info] "
                f"batch {i}: DataLoader yielded in {iter_start - data_wait_start:.2f}s; "
                f"samples={tuple(samples.shape)}; "
                f"targets total={sum(target_counts)}, max/image={max(target_counts, default=0)}",
                flush=True,
            )

        stage_start = time.perf_counter()
        samples = samples.to(device, non_blocking=True)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        if debug_this_batch:
            _sync_if_cuda(device)
            print(f"[train info] batch {i}: host->device {time.perf_counter() - stage_start:.2f}s", flush=True)

        global_step = epoch * len(data_loader) + i
        actual_batch_size = samples.shape[0]
        micro_batch_size = math.ceil(actual_batch_size / grad_accum_steps)
        num_micro_batches = math.ceil(actual_batch_size / micro_batch_size)

        for j, start_idx in enumerate(range(0, actual_batch_size, micro_batch_size)):
            final_idx = min(start_idx + micro_batch_size, actual_batch_size)
            new_samples = samples[start_idx:final_idx]
            new_targets = targets[start_idx:final_idx]

            stage_start = time.perf_counter()
            if debug_this_batch:
                print(f"[train info] batch {i}.{j}: starting forward", flush=True)
            with torch.amp.autocast(device_type, enabled=amp_enabled):
                outputs = model(new_samples, new_targets)
            if debug_this_batch:
                _sync_if_cuda(device)
                print(
                    f"[train info] batch {i}.{j}: forward {time.perf_counter() - stage_start:.2f}s",
                    flush=True,
                )
            
            stage_start = time.perf_counter()
            if debug_this_batch:
                print(f"[train info] batch {i}.{j}: starting criterion/loss", flush=True)
            with torch.amp.autocast(device_type, enabled=False):
                loss_dict = criterion(outputs, new_targets)
                losses = sum(loss_dict.values()) / num_micro_batches
            if debug_this_batch:
                _sync_if_cuda(device)
                loss_preview = ", ".join(f"{k}={float(v.detach().cpu()):.4f}" for k, v in loss_dict.items())
                print(
                    f"[train info] batch {i}.{j}: criterion {time.perf_counter() - stage_start:.2f}s "
                    f"({loss_preview})",
                    flush=True,
                )

            stage_start = time.perf_counter()
            if debug_this_batch:
                print(f"[train info] batch {i}.{j}: starting backward", flush=True)
            if amp_enabled:
                scaler.scale(losses).backward()
            else:
                losses.backward()
            if debug_this_batch:
                _sync_if_cuda(device)
                print(
                    f"[train info] batch {i}.{j}: backward {time.perf_counter() - stage_start:.2f}s",
                    flush=True,
                )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced_scaled = sum(loss_dict_reduced.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        if debug_this_batch:
            print(f"[train info] batch {i}: starting optimizer step", flush=True)

        # amp backward function
        if amp_enabled:
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        if debug_this_batch:
            _sync_if_cuda(device)
            print(f"[train info] batch {i}: optimizer+ema total {time.perf_counter() - iter_start:.2f}s", flush=True)
                    
        # ema
        if ema is not None:
            ema.update(model)
            
        if warmup_scheduler is not None:
            warmup_scheduler.step() 


        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])     


        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value, global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)
            free, total = torch.cuda.mem_get_info(device)
            mem_used_MB = (total - free) / GIGABYTE
            writer.add_scalar('Info/memory',  mem_used_MB, global_step)

        optimizer.zero_grad(set_to_none=True)
        data_wait_start = time.perf_counter()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}




@torch.no_grad()
def evaluate(model, postprocessors, coco_evaluator, data_loader, device, writer=None, save_results=False):
    model.eval()
    model_ref = dist_utils.de_parallel(model)
    if hasattr(postprocessors, "set_dcc") and hasattr(model_ref, "dcc"):
        postprocessors.set_dcc(model_ref.dcc)
    if coco_evaluator is not None:
        coco_evaluator.cleanup()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    res_json = [] 

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, targets)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors(outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if save_results:
            for k, v in res.items():
                scores = v['scores']
                labels = v['labels']
                if 'keypoints' in v:
                    keypoints = v['keypoints']
                    for s, l, kpt in zip(scores, labels, keypoints):
                        res_json.append(
                            {
                                "image_id": k,
                                "category_id": _map_prediction_label(coco_evaluator, l.item()),
                                "keypoints": kpt.round(decimals=4).tolist(),
                                "score": s.item()
                            }
                        )
                elif 'boxes' in v:
                    boxes = v['boxes']
                    for s, l, box in zip(scores, labels, boxes):
                        xyxy = box.round(decimals=4)
                        x1, y1, x2, y2 = xyxy.tolist()
                        res_json.append(
                            {
                                "image_id": k,
                                "category_id": _map_prediction_label(coco_evaluator, l.item()),
                                "bbox": [x1, y1, round(x2 - x1, 4), round(y2 - y1, 4)],
                                "score": s.item(),
                            }
                        )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    if save_results:
        return res_json

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in coco_evaluator.coco_eval:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'keypoints' in coco_evaluator.coco_eval:
            stats['coco_eval_keypoints'] = coco_evaluator.coco_eval['keypoints'].stats.tolist()
    return stats
