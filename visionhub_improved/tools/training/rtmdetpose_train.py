from __future__ import annotations

import argparse
import sys
from typing import List


_POSE_CONFIGS = {
    "n": "configs/rtmdetpose/rtmdetpose_hgnetv2_n_custom.py",
    "s": "configs/rtmdetpose/rtmdetpose_hgnetv2_s_custom.py",
    "m": "configs/rtmdetpose/rtmdetpose_hgnetv2_m_custom.py",
    "l": "configs/rtmdetpose/rtmdetpose_hgnetv2_l_custom.py",
}


def _prepare_argv(argv: List[str]) -> List[str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", default="s")
    parsed, remaining = parser.parse_known_args(argv)

    has_config = any(token in {"--config_file", "--config-file", "-c"} for token in remaining)
    if has_config:
        return remaining

    variant = str(parsed.variant).strip().lower()
    if variant not in _POSE_CONFIGS:
        supported = ", ".join(sorted(_POSE_CONFIGS))
        raise ValueError(f"Unsupported RTMDetPose variant '{parsed.variant}'. Use one of: {supported}.")

    return ["--config_file", _POSE_CONFIGS[variant], *remaining]


def main() -> None:
    import train as train_module

    argv = _prepare_argv(sys.argv[1:])
    args = train_module.get_args_parser().parse_args(argv)
    train_module.main(args)


if __name__ == "__main__":
    main()
