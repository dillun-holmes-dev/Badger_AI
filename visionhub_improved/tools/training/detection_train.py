"""Variant-aware detection training entrypoint."""

from __future__ import annotations

import argparse
import sys
from typing import List

from visionhub.detection_variants import resolve_detection_config_file


def _prepare_argv(argv: List[str], default_family: str) -> List[str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--family", default=default_family)
    parser.add_argument("--variant", default="s")
    parsed, remaining = parser.parse_known_args(argv)

    has_config = any(token in {"--config_file", "--config-file", "-c"} for token in remaining)
    if not has_config:
        remaining = [
            "--config_file",
            resolve_detection_config_file(parsed.family, parsed.variant),
            *remaining,
        ]
    return remaining


def main(default_family: str) -> None:
    import train as train_module

    argv = _prepare_argv(sys.argv[1:], default_family)
    args = train_module.get_args_parser().parse_args(argv)
    train_module.main(args)


if __name__ == "__main__":
    raise SystemExit("Use a family-specific wrapper module.")
