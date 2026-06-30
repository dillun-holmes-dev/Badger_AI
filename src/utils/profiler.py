#!/usr/bin/env python3
"""
Model Profiler — FLOPs, parameter counts, and latency measurement.

Rule 3 of Maximum Technical Rigor: "Every model variant must be
profiled — FLOPs, params, latency. No unmeasured claims."

This profiler:
  1. Counts parameters (total, trainable, by module)
  2. Estimates FLOPs (multiply-add operations)
  3. Measures GPU inference latency (mean, p50, p95)
  4. Measures CPU inference latency
  5. Exports results as JSON for leaderboard tracking

Usage:
    python src/utils/profiler.py --variant badger-s --input-size 640
    python src/utils/profiler.py --all  # Profile all variants

    # In code:
    from src.utils.profiler import profile_model
    results = profile_model(model, input_size=640)
    print(f"FLOPs: {results['gflops']:.1f}G, Params: {results['params_M']:.1f}M")
"""

import time
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# =============================================================================
# Parameter Counting
# =============================================================================

def count_parameters(model, detailed=False):
    """
    Count parameters in a PyTorch model.

    Args:
        model: nn.Module
        detailed: if True, return per-module breakdown

    Returns:
        dict with 'total', 'trainable', and optionally 'by_module'
    """
    total = 0
    trainable = 0
    by_module = defaultdict(lambda: {'total': 0, 'trainable': 0})

    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n

        if detailed:
            # Group by top-level module
            module_name = name.split('.')[0]
            by_module[module_name]['total'] += n
            if param.requires_grad:
                by_module[module_name]['trainable'] += n

    result = {
        'total': total,
        'trainable': trainable,
        'total_M': total / 1e6,
        'trainable_M': trainable / 1e6,
    }

    if detailed:
        result['by_module'] = dict(by_module)

    return result


# =============================================================================
# FLOP Estimation
# =============================================================================

def estimate_flops(model, input_size=640, channels=3, device='cpu'):
    """
    Estimate FLOPs (floating-point multiply-add operations) for a model.

    Uses PyTorch's built-in profiler when possible, falls back to
    manual counting for standard layer types.

    For Conv2d: FLOPs = 2 × k_h × k_w × C_in × C_out × H_out × W_out / groups
    For Linear: FLOPs = 2 × in_features × out_features
    For BatchNorm2d: FLOPs = 2 × num_features × H × W (affine)
    For SiLU: FLOPs = 4 × num_elements (approx: sigmoid + multiply)

    Args:
        model: nn.Module
        input_size: int (square) or (H, W) tuple
        channels: input channels (3 for RGB)

    Returns:
        dict with 'total_flops', 'gflops', and per-op breakdown
    """
    if isinstance(input_size, int):
        H, W = input_size, input_size
    else:
        H, W = input_size

    dummy = torch.randn(1, channels, H, W).to(device)
    model = model.to(device).eval()

    total_flops = 0
    op_counts = defaultdict(int)

    def conv2d_flops(m, x, y):
        """FLOPs for Conv2d layer."""
        _, c_in, h_in, w_in = x[0].shape
        c_out, h_out, w_out = y[0].shape[1:]
        k_h, k_w = m.kernel_size
        groups = m.groups

        # MACs = k_h * k_w * c_in/groups * c_out * h_out * w_out
        # FLOPs = 2 * MACs (multiply + add)
        macs = k_h * k_w * (c_in // groups) * c_out * h_out * w_out
        flops = 2 * macs
        return flops

    def bn_flops(m, x, y):
        """FLOPs for BatchNorm2d (affine)."""
        _, c, h, w = y[0].shape
        # 2 ops per element: multiply by gamma, add beta
        return 2 * c * h * w

    def activation_flops(m, x, y):
        """FLOPs for activation functions."""
        # Approximate: SiLU = sigmoid + multiply ≈ 4 ops per element
        _, c, h, w = y[0].shape
        if isinstance(m, (nn.SiLU, nn.Sigmoid)):
            return 4 * c * h * w
        elif isinstance(m, nn.ReLU):
            return 1 * c * h * w  # comparison only
        return 1 * c * h * w

    def maxpool_flops(m, x, y):
        """FLOPs for MaxPool2d."""
        _, c, h_out, w_out = y[0].shape
        k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
        # k² comparisons per output element
        return (k ** 2) * c * h_out * w_out

    def upsample_flops(m, x, y):
        return 0  # Nearest-neighbor upsample has 0 FLOPs

    # Map module types to FLOP counters
    handlers = {
        nn.Conv2d: conv2d_flops,
        nn.BatchNorm2d: bn_flops,
        nn.SiLU: activation_flops,
        nn.ReLU: activation_flops,
        nn.Sigmoid: activation_flops,
        nn.MaxPool2d: maxpool_flops,
        nn.Upsample: upsample_flops,
    }

    # Track hooks
    hooks = []

    def make_hook(layer_type, name):
        def hook(m, x, y):
            if not isinstance(y, (list, tuple)):
                y = (y,)
            flops = handlers[layer_type](m, x, y)
            nonlocal total_flops
            total_flops += flops
            op_counts[name] += flops
        return hook

    for name, module in model.named_modules():
        for layer_type, handler in handlers.items():
            if isinstance(module, layer_type):
                h = module.register_forward_hook(make_hook(layer_type, name))
                hooks.append(h)

    # Forward pass
    with torch.no_grad():
        try:
            model(dummy)
        except Exception as e:
            print(f"  ⚠ Forward pass failed during FLOP counting: {e}")
            print(f"  Falling back to param-based estimate...")
            # Cleanup hooks
            for h in hooks:
                h.remove()
            return _fallback_flops(model, input_size)

    # Cleanup hooks
    for h in hooks:
        h.remove()

    result = {
        'total_flops': total_flops,
        'gflops': total_flops / 1e9,
        'mflops': total_flops / 1e6,
    }

    return result


def _fallback_flops(model, input_size):
    """Fallback FLOP estimation based on parameter count * spatial resolution."""
    params = count_parameters(model)
    # Rough heuristic: each param participates in ~H*W/64 multiplications
    H, W = (input_size, input_size) if isinstance(input_size, int) else input_size
    spatial_factor = (H // 8) * (W // 8)  # Based on P3 feature map
    approx_flops = 2 * params['total'] * spatial_factor * 0.15  # Heuristic
    return {
        'total_flops': approx_flops,
        'gflops': approx_flops / 1e9,
        'mflops': approx_flops / 1e6,
        'estimate': 'fallback (param-based)',
    }


# =============================================================================
# Latency Measurement
# =============================================================================

def measure_latency(model, input_size=640, batch_size=1, num_warmup=50,
                    num_runs=200, device='cuda'):
    """
    Measure inference latency on GPU or CPU.

    Args:
        model: nn.Module
        input_size: int (square) or (H, W) tuple
        batch_size: batch size for measurement
        num_warmup: number of warmup iterations (excluded from stats)
        num_runs: number of measurement iterations

    Returns:
        dict with mean, median (p50), p95, p99 latency in ms,
        and FPS (frames per second)
    """
    if isinstance(input_size, int):
        H, W = input_size, input_size
    else:
        H, W = input_size

    model = model.to(device).eval()
    dummy = torch.randn(batch_size, 3, H, W).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            model(dummy)

    # Synchronize GPU before timing
    if device == 'cuda':
        torch.cuda.synchronize()

    # Measure
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(dummy)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
            else:
                t0 = time.perf_counter()
                model(dummy)
                t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000)  # ms

    timings = np.array(timings)

    ms_per_image = timings / batch_size

    result = {
        'mean_ms': float(np.mean(ms_per_image)),
        'p50_ms': float(np.median(ms_per_image)),
        'p95_ms': float(np.percentile(ms_per_image, 95)),
        'p99_ms': float(np.percentile(ms_per_image, 99)),
        'min_ms': float(np.min(ms_per_image)),
        'max_ms': float(np.max(ms_per_image)),
        'fps': float(1000.0 / np.mean(ms_per_image)),
        'batch_size': batch_size,
        'num_runs': num_runs,
        'device': device,
    }

    return result


# =============================================================================
# Full Profile
# =============================================================================

def profile_model(model, input_size=640, batch_size=1, device='cuda',
                  detailed_params=False, measure_cpu=False):
    """
    Complete model profiling: params, FLOPs, GPU latency, CPU latency.

    Args:
        model: nn.Module
        input_size: input image size
        batch_size: batch size
        device: 'cuda' or 'cpu'
        detailed_params: include per-module parameter breakdown
        measure_cpu: also measure CPU latency (slower)

    Returns:
        dict with all profiling metrics
    """
    print(f"\n{'='*60}")
    print(f"  PROFILING MODEL")
    print(f"{'='*60}")

    # 1. Parameters
    print("\n  [1/4] Counting parameters...")
    params = count_parameters(model, detailed=detailed_params)
    print(f"    Total:     {params['total_M']:.2f}M")
    print(f"    Trainable: {params['trainable_M']:.2f}M")

    # 2. FLOPs
    print("\n  [2/4] Estimating FLOPs...")
    flops = estimate_flops(model, input_size, device=device)
    print(f"    FLOPs: {flops['gflops']:.2f}G")

    # 3. GPU latency
    print(f"\n  [3/4] Measuring {device.upper()} latency...")
    latency = measure_latency(model, input_size, batch_size,
                              num_warmup=30, num_runs=100, device=device)
    print(f"    Mean: {latency['mean_ms']:.1f}ms  "
          f"P50: {latency['p50_ms']:.1f}ms  "
          f"P95: {latency['p95_ms']:.1f}ms")
    print(f"    FPS:  {latency['fps']:.1f}")

    # 4. CPU latency (optional)
    cpu_latency = None
    if measure_cpu and device != 'cpu':
        print(f"\n  [4/4] Measuring CPU latency...")
        cpu_latency = measure_latency(model, input_size, batch_size,
                                      num_warmup=10, num_runs=50, device='cpu')
        print(f"    Mean: {cpu_latency['mean_ms']:.1f}ms  "
              f"FPS:  {cpu_latency['fps']:.1f}")

    result = {
        'params': params,
        'flops': flops,
        'latency': latency,
        'input_size': input_size,
        'batch_size': batch_size,
    }
    if cpu_latency:
        result['cpu_latency'] = cpu_latency

    return result


# =============================================================================
# Batch Profiling (all variants)
# =============================================================================

def profile_all_variants(input_size=640, device='cuda', output_path=None):
    """
    Profile all Badger model variants and save comparison.

    Args:
        input_size: input image size
        device: 'cuda' or 'cpu'
        output_path: path to save JSON results (default: runs/profiles.json)

    Returns:
        list of result dicts for all variants
    """
    from src.models import create_model

    variants = [
        ('badger-n',  0.25, 0.33),
        ('badger-s',  0.50, 0.33),
        ('badger-m',  0.75, 0.67),
        ('badger-l',  1.00, 1.00),
        ('badger-x',  1.25, 1.00),
    ]

    all_results = []

    print(f"\n{'='*60}")
    print(f"  BADGER MODEL COMPARISON")
    print(f"  Input: {input_size}×{input_size}, Device: {device}")
    print(f"{'='*60}")

    for name, w, d in variants:
        print(f"\n{'─'*60}")
        print(f"  {name.upper()} (w={w}, d={d})")
        print(f"{'─'*60}")

        model = create_model(name, num_classes=80)
        result = profile_model(model, input_size, device=device)
        result['variant'] = name
        result['width'] = w
        result['depth'] = d
        all_results.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Variant':<12s} {'Params':>8s} {'FLOPs':>8s} "
          f"{'Latency':>10s} {'FPS':>8s}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")

    for r in all_results:
        name = r['variant']
        params = f"{r['params']['total_M']:.1f}M"
        flops = f"{r['flops']['gflops']:.1f}G"
        lat = f"{r['latency']['p50_ms']:.1f}ms"
        fps = f"{r['latency']['fps']:.0f}"
        print(f"  {name:<12s} {params:>8s} {flops:>8s} {lat:>10s} {fps:>8s}")

    # Save to JSON
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  ✓ Saved to {output_path}")

    return all_results


# =============================================================================
# Export helper
# =============================================================================

def export_profile_summary(results, format='markdown'):
    """
    Export profiling results as a formatted table.

    Args:
        results: list of result dicts from profile_all_variants
        format: 'markdown' or 'csv'

    Returns:
        formatted string
    """
    if format == 'markdown':
        lines = [
            "| Variant | Params (M) | GFLOPs | Latency P50 (ms) | FPS |",
            "|---------|-----------|--------|-----------------|-----|",
        ]
        for r in results:
            lines.append(
                f"| {r['variant']} | "
                f"{r['params']['total_M']:.1f} | "
                f"{r['flops']['gflops']:.1f} | "
                f"{r['latency']['p50_ms']:.1f} | "
                f"{r['latency']['fps']:.0f} |"
            )
        return '\n'.join(lines)

    elif format == 'csv':
        lines = ["variant,params_M,gflops,latency_p50_ms,fps"]
        for r in results:
            lines.append(
                f"{r['variant']},"
                f"{r['params']['total_M']:.1f},"
                f"{r['flops']['gflops']:.1f},"
                f"{r['latency']['p50_ms']:.1f},"
                f"{r['latency']['fps']:.0f}"
            )
        return '\n'.join(lines)

    return str(results)


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Badger Model Profiler')
    parser.add_argument('--variant', type=str, default='badger-s',
                        help='Model variant to profile')
    parser.add_argument('--all', action='store_true',
                        help='Profile all variants')
    parser.add_argument('--input-size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--cpu', action='store_true',
                        help='Also measure CPU latency')
    parser.add_argument('--detailed', action='store_true',
                        help='Show per-module parameter breakdown')
    parser.add_argument('--output', type=str, default='runs/profiles.json',
                        help='Output JSON path')
    parser.add_argument('--format', type=str, choices=['json', 'markdown', 'csv'],
                        default='json', help='Output format')

    args = parser.parse_args()

    if args.all:
        results = profile_all_variants(args.input_size, args.device, args.output)
        if args.format != 'json':
            print('\n' + export_profile_summary(results, args.format))
    else:
        device = args.device if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠ CUDA not available, falling back to CPU")
            device = 'cpu'

        from src.models import create_model
        model = create_model(args.variant, num_classes=80)
        result = profile_model(
            model, args.input_size, args.batch_size, device,
            detailed_params=args.detailed, measure_cpu=args.cpu
        )

        if args.format == 'json':
            print('\n' + json.dumps(result, indent=2, default=str))
        elif args.format == 'markdown':
            print('\n' + export_profile_summary([result], 'markdown'))
