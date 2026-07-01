"""
Consolidated detection training — single entry point for all model families.

Replaces 7 duplicate wrappers (detrdet_train.py, rtmodet_train.py, etc.)
with a single function that dispatches by family.

Usage:
    python -m tools.training.train_detect --family rtmdetdet --variant s
    python -m tools.training.train_detect --family detrpose --variant m --epochs 100
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from visionhub.detection_variants import (
    DETECTION_VARIANTS,
    normalize_detection_family,
    normalize_detection_variant,
    resolve_detection_config_file,
)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='VisionHub Detection Training — all families, one entry point'
    )
    parser.add_argument('--family', '-f', type=str, default='rtmdetdet',
                        choices=list(DETECTION_VARIANTS.keys()),
                        help='Model family')
    parser.add_argument('--variant', '-v', type=str, default='s',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size variant')
    parser.add_argument('--config_file', '--config-file', '-c', type=str, default=None,
                        help='Direct config file path (overrides --family/--variant)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--image_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--options', nargs='+', default=None,
                        help='Config overrides in key=value format')
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = get_args_parser().parse_args(argv)

    # Resolve config
    if args.config_file:
        config_file = args.config_file
    else:
        family = normalize_detection_family(args.family)
        variant = normalize_detection_variant(args.variant)
        config_file = resolve_detection_config_file(family, variant)

    print(f'🚀 VisionHub Training: {args.family}/{args.variant}')
    print(f'   Config: {config_file}')

    # Build argv for train.py
    train_argv = ['train.py', '--config_file', config_file]
    overrides = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'num_workers': args.num_workers,
        'output_dir': args.output_dir,
        'data_root': args.data_root,
        'resume': args.resume,
        'device': args.device,
        'seed': args.seed,
    }
    for key, val in overrides.items():
        if val is not None:
            train_argv.append(f'--{key}')
            train_argv.append(str(val))

    if args.amp:
        train_argv.append('--amp')
    else:
        train_argv.append('--no-amp')

    if args.options:
        train_argv.append('--options')
        train_argv.extend(args.options)

    sys.argv = train_argv
    import train as train_module
    parsed = train_module.get_args_parser().parse_args(train_argv[1:])
    train_module.main(parsed)


if __name__ == '__main__':
    main()
