#!/usr/bin/env python3
"""
Benchmark runner script.

Usage:
    # Quick benchmark (params, GFLOPs, latency — no dataset needed)
    python scripts/benchmark.py --model badger-s

    # Full benchmark on COCO (requires dataset)
    python scripts/benchmark.py --model badger-s --full --data config/coco.yaml

    # Compare against SOTA
    python scripts/benchmark.py --model runs/exp/best.pth --compare
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models import create_model
from src.utils.benchmark import BenchmarkRunner, quick_benchmark


def parse_args():
    parser = argparse.ArgumentParser(description='Badger Benchmark Suite')
    parser.add_argument('--model', type=str, default='badger-s',
                        help='Model variant or checkpoint path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--full', action='store_true',
                        help='Run full benchmark (requires dataset)')
    parser.add_argument('--data', type=str, default='config/coco.yaml',
                        help='Dataset config for full benchmark')
    parser.add_argument('--compare', action='store_true',
                        help='Compare against SOTA baselines')
    parser.add_argument('--save', type=str, default=None,
                        help='Save report to JSON file')
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    if args.model.endswith('.pth'):
        print(f"Loading checkpoint: {args.model}")
        from src.models import Badger
        model = Badger(num_classes=80)
        checkpoint = torch.load(args.model, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Creating model: {args.model}")
        model = create_model(variant=args.model, num_classes=80)

    model = model.to(device)
    model.eval()

    if args.full:
        # Full benchmark
        print("\nRunning full benchmark...")
        from src.data import create_dataloader
        dataloader = create_dataloader(
            args.data, img_size=args.img_size, batch_size=1,
            augment=False, mosaic=False, shuffle=False
        )
        runner = BenchmarkRunner(model, dataloader, device=str(device),
                                 img_size=args.img_size)
        results = runner.run_full()

        if args.compare:
            runner.print_report(results)

        if args.save:
            runner.save_report(results, args.save)
    else:
        # Quick benchmark
        print("\nRunning quick benchmark...")
        results = quick_benchmark(model, device=str(device), img_size=args.img_size)

        # Show comparison
        print(f"\n{'='*60}")
        print("  QUICK COMPARISON")
        print(f"{'='*60}")
        print(f"  {'Model':<20} {'mAP':>6} {'Params':>8} {'GFLOPs':>8}")
        print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*8}")
        baselines = [
            ('D-FINE-X', 61.4, 62.0, 202.0),
            ('DEIMv2-X', 61.3, 51.2, 151.6),
            ('EC-X', 61.1, 49.9, 151.0),
            ('YOLOv9-C', 57.1, 25.5, 51.8),
            ('D-FINE-S', 53.4, 10.3, 25.0),
            ('YOLOX-S', 44.3, 9.0, 13.5),
            ('YOLOv9-T', 41.8, 2.0, 4.0),
            ('YOLOX-Nano', 28.8, 0.9, 1.3),
        ]
        for name, mAP, params, gflops in baselines:
            print(f"  {name:<20} {mAP:>4.1f}%  {params:>6.1f}M  {gflops:>6.0f}")

        print(f"\n  {'─' * 50}")
        our_params = results['params_M']
        print(f"  {'Badger (ours)':<20} {'-':>6}  {our_params:>6.1f}M  "
              f"{results.get('gflops', '-'):>6}")
        print(f"  {'FPS:':<20} {results['fps']:.1f} | {'Latency:':>12} {results['latency_ms']:.1f}ms")


if __name__ == '__main__':
    main()
