"""Native PyTorch classifier training CLI."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from visionhub.models.classification import (
    CLASSIFIER_AUGMENTATION_PRESET_NONE,
    CLASSIFIER_AUGMENTATION_PRESETS,
    NATIVE_PYTORCH_CLASSIFIER_AVAILABLE,
    SUPPORTED_NATIVE_PYTORCH_CLASSIFIER_ARCHITECTURES,
    resolve_native_classifier_device,
    resolve_native_classifier_training_options,
    train_native_classifier,
)


def _bool_action():
    action = getattr(argparse, "BooleanOptionalAction", None)
    if action is not None:
        return action

    class _CompatBooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, **kwargs):
            expanded_option_strings = []
            for option in option_strings:
                expanded_option_strings.append(option)
                if option.startswith("--"):
                    expanded_option_strings.append("--no-" + option[2:])
            super().__init__(option_strings=expanded_option_strings, dest=dest, nargs=0, default=default, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            is_negative = bool(option_string and option_string.startswith("--no-"))
            setattr(namespace, self.dest, not is_negative)

    return _CompatBooleanOptionalAction


_TRAINING_CONFIG_ALIASES = {
    "data_root": "source_root",
    "dataset_root": "source_root",
    "output_dir": "output_root",
    "native_pytorch_architecture": "architecture",
}


def _load_training_config(path: str) -> Dict[str, Any]:
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("training_config must contain a top-level mapping/object.")

    normalized: Dict[str, Any] = {}
    for key, value in payload.items():
        canonical_key = _TRAINING_CONFIG_ALIASES.get(str(key), str(key))
        normalized[canonical_key] = value
    return normalized


def _resolve_parser_defaults(argv: List[str]) -> Dict[str, Any]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--training_config", "--training-config", dest="training_config", type=str, default=None)
    known_args, _ = bootstrap.parse_known_args(argv)
    if not known_args.training_config:
        return {}
    defaults = _load_training_config(known_args.training_config)
    defaults["training_config"] = known_args.training_config
    return defaults


def build_parser(defaults: Optional[Dict[str, Any]] = None) -> argparse.ArgumentParser:
    defaults = dict(defaults or {})
    parser = argparse.ArgumentParser(description="Train a native PyTorch image classifier")
    parser.set_defaults(**defaults)
    parser.add_argument("--training_config", "--training-config", dest="training_config", type=str, default=None,
                        help="Optional YAML config with the same training fields as the CLI flags")
    parser.add_argument("--source_root", "--data_root", dest="source_root", default=None,
                        help="ImageFolder root containing train/ and val/ folders")
    parser.add_argument("--output_root", "--output_dir", dest="output_root", default=None,
                        help="Directory where training artifacts will be written")
    parser.add_argument("--architecture", type=str, default="efficientnet_v2_s",
                        help="Classifier architecture name or alias")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=16)
    parser.add_argument("--image_size", "--image-size", dest="image_size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", "--wd", dest="weight_decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--use_pretrained_weights", "--use-pretrained-weights", "--pretrained",
                        dest="use_pretrained_weights", action=_bool_action(), default=True)
    parser.add_argument("--augmentation_preset", "--augmentation-preset", dest="augmentation_preset",
                        type=str, default=CLASSIFIER_AUGMENTATION_PRESET_NONE,
                        choices=sorted(CLASSIFIER_AUGMENTATION_PRESETS))
    parser.add_argument("--augmentation_enabled", "--augmentation-enabled", dest="augmentation_enabled",
                        action=_bool_action(), default=None)
    parser.add_argument("--augmentation_random_resized_crop", "--augmentation-random-resized-crop",
                        dest="augmentation_random_resized_crop", action=_bool_action(), default=None)
    parser.add_argument("--augmentation_horizontal_flip", "--augmentation-horizontal-flip",
                        dest="augmentation_horizontal_flip", action=_bool_action(), default=None)
    parser.add_argument("--auto_augment", "--auto-augment", dest="auto_augment", type=str, default=None,
                        help="Optional auto augmentation policy for native classifiers: randaugment, autoaugment, augmix")
    parser.add_argument("--mosaic", type=float, default=None)
    parser.add_argument("--mixup", type=float, default=None)
    parser.add_argument("--copy_paste", "--copy-paste", dest="copy_paste", type=float, default=None)
    parser.add_argument("--degrees", type=float, default=None)
    parser.add_argument("--translate", type=float, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--shear", type=float, default=None)
    parser.add_argument("--perspective", type=float, default=None)
    parser.add_argument("--flipud", type=float, default=None)
    parser.add_argument("--fliplr", type=float, default=None)
    parser.add_argument("--hsv_h", "--hsv-h", dest="hsv_h", type=float, default=None)
    parser.add_argument("--hsv_s", "--hsv-s", dest="hsv_s", type=float, default=None)
    parser.add_argument("--hsv_v", "--hsv-v", dest="hsv_v", type=float, default=None)
    parser.add_argument("--erasing", type=float, default=None)
    parser.add_argument("--blur", type=float, default=None)
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:<index>")
    return parser


def _write_cli_summary(
    output_root: Path,
    *,
    args: argparse.Namespace,
    resolved_options: dict,
    resolved_device: str,
) -> None:
    summary_path = output_root / "training_request.json"
    payload = {
        "command": "visionhub-train-classifier",
        "training_config": args.training_config,
        "requested": {
            "source_root": args.source_root,
            "output_root": args.output_root,
            "architecture": args.architecture,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "patience": args.patience,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "use_pretrained_weights": bool(args.use_pretrained_weights),
            "augmentation_preset": args.augmentation_preset,
            "auto_augment": args.auto_augment,
            "device": args.device,
        },
        "resolved": {
            "device": resolved_device,
            "architecture": resolved_options["architecture"],
            "image_size": resolved_options["image_size"],
            "requested_image_size": resolved_options["requested_image_size"],
            "optimizer": resolved_options["optimizer"],
            "requested_optimizer": resolved_options["requested_optimizer"],
            "optimizer_profile": resolved_options["optimizer_profile"],
            "learning_rate": resolved_options["learning_rate"],
            "weight_decay": resolved_options["weight_decay"],
            "dropout": resolved_options["dropout"],
            "augmentation_preset": resolved_options["augmentation_preset"],
            "augmentations": resolved_options["augmentations"],
            "source_stats": resolved_options["source_stats"],
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def main() -> None:
    defaults = _resolve_parser_defaults(sys.argv[1:])
    parser = build_parser(defaults=defaults)
    args = parser.parse_args()
    if not args.source_root:
        parser.error("--source_root is required unless provided via --training_config")
    if not args.output_root:
        parser.error("--output_root is required unless provided via --training_config")

    if not NATIVE_PYTORCH_CLASSIFIER_AVAILABLE:
        raise RuntimeError(
            "Native PyTorch classifier runtime is not available. "
            "Install torch, torchvision, and pillow."
        )

    resolved_options = resolve_native_classifier_training_options(
        source_root=args.source_root,
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        augmentation_preset=args.augmentation_preset,
        augmentation_enabled=args.augmentation_enabled,
        augmentation_random_resized_crop=args.augmentation_random_resized_crop,
        augmentation_horizontal_flip=args.augmentation_horizontal_flip,
        auto_augment=args.auto_augment,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        erasing=args.erasing,
        blur=args.blur,
        noise=args.noise,
    )
    if resolved_options["architecture"] not in SUPPORTED_NATIVE_PYTORCH_CLASSIFIER_ARCHITECTURES:
        raise ValueError(
            "Unsupported native PyTorch classifier architecture. "
            f"Supported values: {', '.join(SUPPORTED_NATIVE_PYTORCH_CLASSIFIER_ARCHITECTURES)}"
        )

    resolved_device = resolve_native_classifier_device(args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_cli_summary(output_root, args=args, resolved_options=resolved_options, resolved_device=resolved_device)

    source_stats = resolved_options["source_stats"]
    print(
        "Dataset summary: "
        f"classes={source_stats['class_count']}, "
        f"train_images={source_stats['train_images']}, "
        f"val_images={source_stats['val_images']}"
    )
    print(f"Architecture: {resolved_options['architecture']}")
    print(f"Device: {resolved_device}")
    print(
        "Optimizer resolved to "
        f"{resolved_options['optimizer']} "
        f"(requested={resolved_options['requested_optimizer']}, "
        f"profile={resolved_options['optimizer_profile'] or 'manual'}, "
        f"lr={resolved_options['learning_rate']}, "
        f"weight_decay={resolved_options['weight_decay']}, "
        f"dropout={resolved_options['dropout']})"
    )
    if resolved_options["requested_image_size"] != resolved_options["image_size"]:
        print(
            f"Requested image_size={resolved_options['requested_image_size']} adjusted "
            f"to image_size={resolved_options['image_size']}."
        )
    print(
        "Augmentation preset: "
        f"{resolved_options['augmentation_preset']} "
        f"with {resolved_options['augmentations']}"
    )

    result = train_native_classifier(
        source_root=args.source_root,
        output_root=str(output_root),
        architecture=resolved_options["architecture"],
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        image_size=int(resolved_options["image_size"]),
        patience=int(args.patience),
        optimizer_name=str(resolved_options["optimizer"]),
        learning_rate=resolved_options["learning_rate"],
        weight_decay=resolved_options["weight_decay"],
        dropout=resolved_options["dropout"],
        use_pretrained_weights=bool(args.use_pretrained_weights),
        augmentation_enabled=bool(resolved_options["augmentations"]["augmentation_enabled"]),
        augmentation_random_resized_crop=bool(
            resolved_options["augmentations"]["augmentation_random_resized_crop"]
        ),
        augmentation_horizontal_flip=bool(
            resolved_options["augmentations"]["augmentation_horizontal_flip"]
        ),
        auto_augment=str(resolved_options["augmentations"]["auto_augment"] or ""),
        mosaic=float(resolved_options["augmentations"]["mosaic"]),
        mixup=float(resolved_options["augmentations"]["mixup"]),
        copy_paste=float(resolved_options["augmentations"]["copy_paste"]),
        degrees=float(resolved_options["augmentations"]["degrees"]),
        translate=float(resolved_options["augmentations"]["translate"]),
        scale=float(resolved_options["augmentations"]["scale"]),
        shear=float(resolved_options["augmentations"]["shear"]),
        perspective=float(resolved_options["augmentations"]["perspective"]),
        flipud=float(resolved_options["augmentations"]["flipud"]),
        fliplr=float(resolved_options["augmentations"]["fliplr"]),
        hsv_h=float(resolved_options["augmentations"]["hsv_h"]),
        hsv_s=float(resolved_options["augmentations"]["hsv_s"]),
        hsv_v=float(resolved_options["augmentations"]["hsv_v"]),
        erasing=float(resolved_options["augmentations"]["erasing"]),
        blur=float(resolved_options["augmentations"]["blur"]),
        noise=float(resolved_options["augmentations"]["noise"]),
        device=resolved_device,
        log_callback=print,
    )

    print("")
    print(f"Training artifacts: {result['save_dir']}")
    print(f"Least-confused checkpoint: {result['least_confused_model_path']}")
    print(f"Last checkpoint: {result['last_model_path']}")
    print(f"Class names: {', '.join(result['class_names'])}")


if __name__ == "__main__":
    main()
