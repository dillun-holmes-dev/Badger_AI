"""
VisionHub Unified CLI — single entry point for all workflows.

Replaces 20+ separate CLI scripts with clean subcommands:

    visionhub train --family detrpose --variant s --epochs 100
    visionhub infer --family rtmdetdet --weights output/best.pth --image cat.jpg
    visionhub export --family detrpose --weights output/best.pth --format onnx
    visionhub benchmark --family rtmdetdet --variant s
    visionhub classify train --data /path/to/images --epochs 50
    visionhub classify infer --weights classifier.pth --image dog.jpg
    visionhub info --family detrpose --variant s

Colab quickstart:
    !visionhub train --family rtmdetdet --variant n --epochs 50 \\
        --data_root /content/coco_data --output_dir /content/output
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


# =============================================================================
# Shared argument builders
# =============================================================================

def _add_family_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--family', '-f', type=str, default='rtmdetdet',
        choices=['detrpose', 'rtmopose', 'rtmdetpose', 'detrdet', 'rtmodet', 'rtmdetdet'],
        help='Model family (default: rtmdetdet for detection)'
    )
    parser.add_argument(
        '--variant', '-v', type=str, default='s',
        choices=['n', 's', 'm', 'l', 'x'],
        help='Model size variant (default: s)'
    )


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--image_size', type=int, default=640, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--data_root', type=str, default=None, help='Dataset root (contains train/val/)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--pretrain', type=str, default=None, help='Pretrained weights path')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True,
                        help='Mixed precision training')
    parser.add_argument('--eval_interval', type=int, default=5, help='Eval every N epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--compile_model', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable torch.compile')


# =============================================================================
# Subcommand: train
# =============================================================================

def _cmd_train(args: argparse.Namespace) -> None:
    """Train a detection/pose model."""
    from visionhub.detection_variants import resolve_detection_config_file

    config_file = resolve_detection_config_file(args.family, args.variant)
    print(f'🚀 Training {args.family}/{args.variant}')
    print(f'   Config: {config_file}')
    print(f'   Epochs: {args.epochs} | Image size: {args.image_size}')

    # Build argv for the existing train.py
    train_argv = ['train.py', '--config_file', config_file]
    if args.epochs:
        train_argv += ['--epochs', str(args.epochs)]
    if args.batch_size:
        train_argv += ['--batch_size', str(args.batch_size)]
    if args.image_size:
        train_argv += ['--image_size', str(args.image_size)]
    if args.num_workers is not None:
        train_argv += ['--num_workers', str(args.num_workers)]
    if args.output_dir:
        train_argv += ['--output_dir', str(args.output_dir)]
    if args.data_root:
        train_argv += ['--data_root', str(args.data_root)]
    if args.resume:
        train_argv += ['--resume', str(args.resume)]
    if args.pretrain:
        train_argv += ['--pretrain', str(args.pretrain)]
    if args.amp:
        train_argv += ['--amp']
    else:
        train_argv += ['--no-amp']
    if args.eval_interval:
        train_argv += ['--eval_interval', str(args.eval_interval)]
    if args.device:
        train_argv += ['--device', str(args.device)]
    if args.seed is not None:
        train_argv += ['--seed', str(args.seed)]
    if args.compile_model:
        train_argv += ['--compile_model']

    sys.argv = train_argv
    import train as train_module
    parsed = train_module.get_args_parser().parse_args(train_argv[1:])
    train_module.main(parsed)


# =============================================================================
# Subcommand: infer
# =============================================================================

def _cmd_infer(args: argparse.Namespace) -> None:
    """Run inference with a trained model."""
    print(f'🔍 Inference with {args.family}/{args.variant}')
    print(f'   Weights: {args.weights}')
    print(f'   Image: {args.image}')

    # Map family → inference module
    infer_modules = {
        'detrpose': 'visionhub.cli.infer_pose',
        'rtmopose': 'visionhub.cli.infer_rtmo',
        'rtmdetpose': 'visionhub.cli.infer_rtmdetpose',
        'detrdet': 'visionhub.cli.infer_detr_detect',
        'rtmodet': 'visionhub.cli.infer_rtmo_detect',
        'rtmdetdet': 'visionhub.cli.infer_rtmdet_detect',
    }
    mod_name = infer_modules.get(args.family)
    if not mod_name:
        print(f'❌ Inference not supported for family: {args.family}')
        sys.exit(1)

    from visionhub.cli._run import run_module
    # Build argv with the key inference args
    infer_argv = [
        '--weights', args.weights,
        '--image', args.image,
    ]
    if args.output:
        infer_argv += ['--output', args.output]
    if args.conf:
        infer_argv += ['--conf', str(args.conf)]
    sys.argv = [mod_name] + infer_argv
    run_module(mod_name.replace('.', '/'))


# =============================================================================
# Subcommand: export
# =============================================================================

def _cmd_export(args: argparse.Namespace) -> None:
    """Export model to ONNX or TensorRT."""
    print(f'📦 Exporting {args.family}/{args.variant} to {args.format}')
    print(f'   Weights: {args.weights}')

    if args.format == 'onnx':
        export_modules = {
            'detrpose': 'visionhub.cli.export_onnx',
            'rtmopose': 'visionhub.cli.export_rtmo_onnx',
            'rtmdetpose': 'visionhub.cli.export_rtmdetpose_onnx',
            'detrdet': 'visionhub.cli.export_detr_detect_onnx',
            'rtmodet': 'visionhub.cli.export_rtmo_detect_onnx',
            'rtmdetdet': 'visionhub.cli.export_rtmdet_detect_onnx',
        }
    elif args.format == 'tensorrt':
        export_modules = {
            'detrpose': 'visionhub.cli.export_tensorrt',
            'rtmopose': 'visionhub.cli.export_tensorrt',
            'rtmdetpose': 'visionhub.cli.export_tensorrt',
            'detrdet': 'visionhub.cli.export_tensorrt',
            'rtmodet': 'visionhub.cli.export_tensorrt',
            'rtmdetdet': 'visionhub.cli.export_tensorrt',
        }
    else:
        print(f'❌ Unsupported export format: {args.format}')
        sys.exit(1)

    mod_name = export_modules.get(args.family)
    if not mod_name:
        print(f'❌ Export not supported for family: {args.family}')
        sys.exit(1)

    from visionhub.cli._run import run_module
    export_argv = ['--weights', args.weights]
    if args.output:
        export_argv += ['--output', args.output]
    if args.opset:
        export_argv += ['--opset', str(args.opset)]
    sys.argv = [mod_name] + export_argv
    run_module(mod_name.replace('.', '/'))


# =============================================================================
# Subcommand: benchmark
# =============================================================================

def _cmd_benchmark(args: argparse.Namespace) -> None:
    """Benchmark model latency and FLOPs."""
    print(f'⏱️  Benchmarking {args.family}/{args.variant}')

    if args.family.startswith('rtmdet'):
        from visionhub.cli._run import run_module
        sys.argv = ['benchmark_torch', '--family', args.family, '--variant', args.variant]
        run_module('visionhub.cli.benchmark_torch')
    else:
        from visionhub.cli._run import run_module
        sys.argv = ['benchmark_torch', '--family', args.family, '--variant', args.variant]
        run_module('visionhub.cli.benchmark_torch')


# =============================================================================
# Subcommand: classify
# =============================================================================

def _cmd_classify(args: argparse.Namespace) -> None:
    """Train or infer with a native PyTorch classifier."""
    if args.classify_cmd == 'train':
        print(f'🏷️  Training classifier on {args.data}')
        from visionhub.cli._run import run_module
        sys.argv = ['train_classifier', '--data', args.data, '--epochs', str(args.epochs),
                     '--batch_size', str(args.batch_size), '--output', args.output]
        run_module('visionhub.cli.train_classifier')
    elif args.classify_cmd == 'infer':
        print(f'🏷️  Classifier inference: {args.image}')
        from visionhub.cli._run import run_module
        sys.argv = ['infer_classifier', '--weights', args.weights, '--image', args.image]
        run_module('visionhub.cli.infer_classifier')


# =============================================================================
# Subcommand: info
# =============================================================================

def _cmd_info(args: argparse.Namespace) -> None:
    """Show model information."""
    from visionhub.detection_variants import DETECTION_VARIANTS, normalize_detection_family, normalize_detection_variant

    family = normalize_detection_family(args.family)
    variant = normalize_detection_variant(args.variant)
    config = DETECTION_VARIANTS[family][variant]

    print(f'📋 Model Info: {family}/{variant}')
    print(f'   Config: {config["config_file"]}')
    print(f'   Module: {config["config_module"]}')
    print()
    print('Available families:', ', '.join(DETECTION_VARIANTS.keys()))
    print('Available variants: n, s, m, l, x')


# =============================================================================
# Main CLI
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description='🦾 VisionHub — Unified CV Training & Deployment CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # --- train ---
    p_train = subparsers.add_parser('train', help='Train a detection/pose model')
    _add_family_args(p_train)
    _add_training_args(p_train)

    # --- infer ---
    p_infer = subparsers.add_parser('infer', help='Run inference')
    _add_family_args(p_infer)
    p_infer.add_argument('--weights', '-w', type=str, required=True, help='Model weights path')
    p_infer.add_argument('--image', '-i', type=str, required=True, help='Input image path')
    p_infer.add_argument('--output', '-o', type=str, default=None, help='Output image path')
    p_infer.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')

    # --- export ---
    p_export = subparsers.add_parser('export', help='Export model to ONNX/TensorRT')
    _add_family_args(p_export)
    p_export.add_argument('--weights', '-w', type=str, required=True, help='Model weights path')
    p_export.add_argument('--format', type=str, default='onnx', choices=['onnx', 'tensorrt'],
                          help='Export format')
    p_export.add_argument('--output', '-o', type=str, default=None, help='Output path')
    p_export.add_argument('--opset', type=int, default=17, help='ONNX opset version')

    # --- benchmark ---
    p_bench = subparsers.add_parser('benchmark', help='Benchmark model speed')
    _add_family_args(p_bench)
    p_bench.add_argument('--device', type=str, default='cuda', help='Device to benchmark on')

    # --- classify ---
    p_classify = subparsers.add_parser('classify', help='Train/infer native classifier')
    p_classify_sub = p_classify.add_subparsers(dest='classify_cmd')
    p_cls_train = p_classify_sub.add_parser('train', help='Train a classifier')
    p_cls_train.add_argument('--data', type=str, required=True, help='ImageFolder dataset path')
    p_cls_train.add_argument('--epochs', type=int, default=50)
    p_cls_train.add_argument('--batch_size', type=int, default=32)
    p_cls_train.add_argument('--output', type=str, default='output/classifier.pth')
    p_cls_infer = p_classify_sub.add_parser('infer', help='Run classifier inference')
    p_cls_infer.add_argument('--weights', type=str, required=True)
    p_cls_infer.add_argument('--image', type=str, required=True)

    # --- info ---
    p_info = subparsers.add_parser('info', help='Show model family info')
    _add_family_args(p_info)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    # Dispatch
    commands = {
        'train': _cmd_train,
        'infer': _cmd_infer,
        'export': _cmd_export,
        'benchmark': _cmd_benchmark,
        'classify': _cmd_classify,
        'info': _cmd_info,
    }
    commands[args.command](args)


if __name__ == '__main__':
    main()
