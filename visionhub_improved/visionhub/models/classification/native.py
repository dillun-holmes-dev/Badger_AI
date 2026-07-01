from __future__ import annotations

"""Native PyTorch classifier utilities for training, inference, and export."""

import contextlib
import inspect
import json
import logging
import math
import os
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:
    from PIL import Image
    import torch
    from torch import nn
    from torch.optim import Adam, AdamW, RMSprop, SGD
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.models import (
        ConvNeXt_Small_Weights,
        ConvNeXt_Tiny_Weights,
        EfficientNet_B0_Weights,
        EfficientNet_B1_Weights,
        EfficientNet_B3_Weights,
        EfficientNet_V2_L_Weights,
        EfficientNet_V2_M_Weights,
        EfficientNet_V2_S_Weights,
        MobileNet_V3_Large_Weights,
        RegNet_Y_400MF_Weights,
        RegNet_Y_800MF_Weights,
        ResNet18_Weights,
        ResNet34_Weights,
        convnext_small,
        convnext_tiny,
        efficientnet_b0,
        efficientnet_b1,
        efficientnet_b3,
        efficientnet_v2_l,
        efficientnet_v2_m,
        efficientnet_v2_s,
        mobilenet_v3_large,
        regnet_y_400mf,
        regnet_y_800mf,
        resnet18,
        resnet34,
    )
    from torchvision.transforms import (
        ColorJitter,
        GaussianBlur,
        InterpolationMode,
        RandomAffine,
        RandomErasing,
        RandomPerspective,
        RandomVerticalFlip,
    )

    NATIVE_PYTORCH_CLASSIFIER_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    Image = Any  # type: ignore
    torch = None  # type: ignore
    nn = Any  # type: ignore
    Adam = AdamW = RMSprop = SGD = None  # type: ignore
    DataLoader = object  # type: ignore
    datasets = transforms = None  # type: ignore
    ColorJitter = GaussianBlur = RandomAffine = RandomErasing = RandomPerspective = RandomVerticalFlip = Any  # type: ignore
    InterpolationMode = Any  # type: ignore
    ConvNeXt_Small_Weights = ConvNeXt_Tiny_Weights = None  # type: ignore
    EfficientNet_B0_Weights = EfficientNet_B1_Weights = EfficientNet_B3_Weights = None  # type: ignore
    EfficientNet_V2_L_Weights = EfficientNet_V2_M_Weights = EfficientNet_V2_S_Weights = None  # type: ignore
    MobileNet_V3_Large_Weights = None  # type: ignore
    RegNet_Y_400MF_Weights = RegNet_Y_800MF_Weights = None  # type: ignore
    ResNet18_Weights = ResNet34_Weights = None  # type: ignore
    convnext_small = convnext_tiny = None  # type: ignore
    efficientnet_b0 = efficientnet_b1 = efficientnet_b3 = None  # type: ignore
    efficientnet_v2_l = efficientnet_v2_m = efficientnet_v2_s = None  # type: ignore
    mobilenet_v3_large = None  # type: ignore
    regnet_y_400mf = regnet_y_800mf = None  # type: ignore
    resnet18 = resnet34 = None  # type: ignore
    NATIVE_PYTORCH_CLASSIFIER_AVAILABLE = False

logger = logging.getLogger(__name__)

CLASSIFIER_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

CLASSIFIER_AUGMENTATION_PRESET_NONE = "none"
CLASSIFIER_AUGMENTATION_PRESET_REFERENCE = "reference"
CLASSIFIER_AUGMENTATION_PRESET_PHANTOM = "phantom"
CLASSIFIER_AUGMENTATION_PRESETS = {
    CLASSIFIER_AUGMENTATION_PRESET_NONE,
    CLASSIFIER_AUGMENTATION_PRESET_REFERENCE,
    CLASSIFIER_AUGMENTATION_PRESET_PHANTOM,
}
CLASSIFIER_AUTO_AUGMENT_NONE = ""
CLASSIFIER_AUTO_AUGMENT_POLICIES = {
    CLASSIFIER_AUTO_AUGMENT_NONE,
    "randaugment",
    "autoaugment",
    "augmix",
}
LEAST_CONFUSED_SELECTION_METRIC = "least_confused(total_confusion,max_confusion,val/loss)"

SUPPORTED_NATIVE_PYTORCH_CLASSIFIER_ARCHITECTURES = (
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b3",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
    "convnext_tiny",
    "convnext_small",
    "mobilenet_v3_large",
    "regnet_y_400mf",
    "regnet_y_800mf",
    "resnet18",
    "resnet34",
)

_ARCHITECTURE_BUILDERS: Dict[str, Tuple[Any, Any]] = {
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.DEFAULT if EfficientNet_B0_Weights else None),
    "efficientnet_b1": (efficientnet_b1, EfficientNet_B1_Weights.DEFAULT if EfficientNet_B1_Weights else None),
    "efficientnet_b3": (efficientnet_b3, EfficientNet_B3_Weights.DEFAULT if EfficientNet_B3_Weights else None),
    "efficientnet_v2_s": (efficientnet_v2_s, EfficientNet_V2_S_Weights.DEFAULT if EfficientNet_V2_S_Weights else None),
    "efficientnet_v2_m": (efficientnet_v2_m, EfficientNet_V2_M_Weights.DEFAULT if EfficientNet_V2_M_Weights else None),
    "efficientnet_v2_l": (efficientnet_v2_l, EfficientNet_V2_L_Weights.DEFAULT if EfficientNet_V2_L_Weights else None),
    "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT if ConvNeXt_Tiny_Weights else None),
    "convnext_small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT if ConvNeXt_Small_Weights else None),
    "mobilenet_v3_large": (
        mobilenet_v3_large,
        MobileNet_V3_Large_Weights.DEFAULT if MobileNet_V3_Large_Weights else None,
    ),
    "regnet_y_400mf": (regnet_y_400mf, RegNet_Y_400MF_Weights.DEFAULT if RegNet_Y_400MF_Weights else None),
    "regnet_y_800mf": (regnet_y_800mf, RegNet_Y_800MF_Weights.DEFAULT if RegNet_Y_800MF_Weights else None),
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT if ResNet18_Weights else None),
    "resnet34": (resnet34, ResNet34_Weights.DEFAULT if ResNet34_Weights else None),
}

_NATIVE_CLASSIFIER_AUTO_OPTIMIZER_PROFILES: Dict[str, Dict[str, Any]] = {
    "efficientnet_b0": {
        "optimizer": "AdamW",
        "learning_rate": 0.0009,
        "weight_decay": 0.0005,
        "dropout": None,
        "profile": "efficientnet_b0:recipe_adamw",
    },
    "efficientnet_b1": {
        "optimizer": "AdamW",
        "learning_rate": 0.0008,
        "weight_decay": 0.0005,
        "dropout": None,
        "profile": "efficientnet_b1:recipe_adamw",
    },
    "efficientnet_b3": {
        "optimizer": "AdamW",
        "learning_rate": 0.00065,
        "weight_decay": 0.0005,
        "dropout": None,
        "profile": "efficientnet_b3:recipe_adamw",
    },
    "efficientnet_v2_s": {
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0005,
        "dropout": None,
        "profile": "efficientnet_v2_s:recipe_adamw",
    },
    "efficientnet_v2_m": {
        "optimizer": "AdamW",
        "learning_rate": 0.00085,
        "weight_decay": 0.0005,
        "dropout": None,
        "profile": "efficientnet_v2_m:recipe_adamw",
    },
    "efficientnet_v2_l": {
        "optimizer": "AdamW",
        "learning_rate": 0.0007,
        "weight_decay": 0.0005,
        "dropout": None,
        "profile": "efficientnet_v2_l:recipe_adamw",
    },
    "convnext_tiny": {
        "optimizer": "AdamW",
        "learning_rate": 0.0007,
        "weight_decay": 0.0005,
        "dropout": None,
        "profile": "convnext_tiny:recipe_adamw",
    },
    "convnext_small": {
        "optimizer": "AdamW",
        "learning_rate": 0.0006,
        "weight_decay": 0.0005,
        "dropout": None,
        "profile": "convnext_small:recipe_adamw",
    },
    "mobilenet_v3_large": {
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0005,
        "dropout": None,
        "profile": "mobilenet_v3_large:recipe_adamw",
    },
    "regnet_y_400mf": {
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "weight_decay": 0.0005,
        "dropout": None,
        "profile": "regnet_y_400mf:recipe_sgd",
    },
    "regnet_y_800mf": {
        "optimizer": "SGD",
        "learning_rate": 0.009,
        "weight_decay": 0.0005,
        "dropout": None,
        "profile": "regnet_y_800mf:recipe_sgd",
    },
    "resnet18": {
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "weight_decay": 0.0005,
        "dropout": None,
        "profile": "resnet18:recipe_sgd",
    },
    "resnet34": {
        "optimizer": "SGD",
        "learning_rate": 0.008,
        "weight_decay": 0.0005,
        "dropout": None,
        "profile": "resnet34:recipe_sgd",
    },
}

_AUGMENTATION_PRESET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    CLASSIFIER_AUGMENTATION_PRESET_NONE: {
        "augmentation_enabled": False,
        "augmentation_random_resized_crop": False,
        "augmentation_horizontal_flip": False,
        "auto_augment": CLASSIFIER_AUTO_AUGMENT_NONE,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "erasing": 0.0,
        "blur": 0.0,
        "noise": 0.0,
    },
    CLASSIFIER_AUGMENTATION_PRESET_REFERENCE: {
        "augmentation_enabled": True,
        "augmentation_random_resized_crop": True,
        "augmentation_horizontal_flip": True,
        "auto_augment": "randaugment",
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.4,
        "hsv_v": 0.4,
        "erasing": 0.4,
        "blur": 0.0,
        "noise": 0.0,
    },
    CLASSIFIER_AUGMENTATION_PRESET_PHANTOM: {
        "augmentation_enabled": True,
        "augmentation_random_resized_crop": False,
        "augmentation_horizontal_flip": False,
        "auto_augment": CLASSIFIER_AUTO_AUGMENT_NONE,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.005,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.1,
        "hsv_v": 0.1,
        "erasing": 0.0,
        "blur": 0.0,
        "noise": 0.0,
    },
}


def _require_native_runtime() -> None:
    if not NATIVE_PYTORCH_CLASSIFIER_AVAILABLE or torch is None or transforms is None:
        raise RuntimeError(
            "Native PyTorch classifier runtime is not available. "
            "Install torch, torchvision, and pillow."
        )


def _create_cuda_amp_grad_scaler(enabled: bool) -> Optional[Any]:
    if not enabled or torch is None:
        return None
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "GradScaler"):
        return amp_module.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _cuda_amp_autocast(enabled: bool) -> Any:
    if not enabled or torch is None:
        return contextlib.nullcontext()
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "autocast"):
        return amp_module.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def normalize_native_classifier_architecture(value: Optional[str]) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "efficientnetb0": "efficientnet_b0",
        "efficientnet-b0": "efficientnet_b0",
        "efficientnetb1": "efficientnet_b1",
        "efficientnet-b1": "efficientnet_b1",
        "efficientnetb3": "efficientnet_b3",
        "efficientnet-b3": "efficientnet_b3",
        "efficientnetv2_s": "efficientnet_v2_s",
        "efficientnetv2-s": "efficientnet_v2_s",
        "efficientnetv2_m": "efficientnet_v2_m",
        "efficientnetv2-m": "efficientnet_v2_m",
        "efficientnetv2_l": "efficientnet_v2_l",
        "efficientnetv2-l": "efficientnet_v2_l",
        "convnexttiny": "convnext_tiny",
        "convnext-tiny": "convnext_tiny",
        "convnextsmall": "convnext_small",
        "convnext-small": "convnext_small",
        "mobilenetv3large": "mobilenet_v3_large",
        "mobilenet-v3-large": "mobilenet_v3_large",
        "mobilenet_v3-large": "mobilenet_v3_large",
        "regnety400mf": "regnet_y_400mf",
        "regnet-y-400mf": "regnet_y_400mf",
        "regnety800mf": "regnet_y_800mf",
        "regnet-y-800mf": "regnet_y_800mf",
        "resnet-18": "resnet18",
        "resnet_18": "resnet18",
        "resnet-34": "resnet34",
        "resnet_34": "resnet34",
    }
    return aliases.get(normalized, normalized)


def normalize_classifier_augmentation_preset(value: Optional[str]) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in CLASSIFIER_AUGMENTATION_PRESETS:
        return normalized
    return CLASSIFIER_AUGMENTATION_PRESET_NONE


def normalize_classifier_auto_augment(value: Optional[str]) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "none", "off", "false", "0"}:
        return CLASSIFIER_AUTO_AUGMENT_NONE
    if normalized in CLASSIFIER_AUTO_AUGMENT_POLICIES:
        return normalized
    return CLASSIFIER_AUTO_AUGMENT_NONE


def resolve_native_classifier_device(requested_device: Optional[str] = None) -> str:
    requested = str(requested_device or "auto").strip().lower()
    if requested in {"", "auto"}:
        return "cuda:0" if bool(torch and torch.cuda.is_available()) else "cpu"
    if requested.startswith("cuda") and not bool(torch and torch.cuda.is_available()):
        logger.warning("CUDA was requested but is not available; falling back to CPU.")
        return "cpu"
    return str(requested_device or "cpu")


def _get_architecture_entry(architecture: str) -> Tuple[Any, Any]:
    normalized = normalize_native_classifier_architecture(architecture)
    entry = _ARCHITECTURE_BUILDERS.get(normalized)
    if entry is None:
        raise ValueError(
            "Unsupported native PyTorch classifier architecture. "
            f"Supported values: {', '.join(SUPPORTED_NATIVE_PYTORCH_CLASSIFIER_ARCHITECTURES)}"
        )
    return entry


def _resolve_default_image_size(weights: Any) -> int:
    preset = weights.transforms() if weights is not None else None
    crop_size = getattr(preset, "crop_size", None)
    if isinstance(crop_size, list) and crop_size:
        try:
            return int(crop_size[0])
        except (TypeError, ValueError):
            pass
    resize_size = getattr(preset, "resize_size", None)
    if isinstance(resize_size, list) and resize_size:
        try:
            return int(resize_size[0])
        except (TypeError, ValueError):
            pass
    return 384


def _resolve_normalization(weights: Any) -> Tuple[List[float], List[float]]:
    preset = weights.transforms() if weights is not None else None
    mean = list(getattr(preset, "mean", [0.485, 0.456, 0.406]))
    std = list(getattr(preset, "std", [0.229, 0.224, 0.225]))
    return mean, std


def _state_dict_to_cpu(
    model: Any,
    *,
    use_half: bool = False,
) -> Dict[str, Any]:
    state_dict: Dict[str, Any] = {}
    for key, value in model.state_dict().items():
        detached = value.detach().cpu()
        if use_half and torch is not None and torch.is_floating_point(detached):
            detached = detached.half()
        state_dict[key] = detached
    return state_dict


def build_native_classifier_model(
    architecture: str,
    *,
    num_classes: int,
    pretrained: bool,
    dropout: Optional[float] = None,
) -> Tuple[Any, Any]:
    """Build a supported native PyTorch classifier model and its weights enum."""
    _require_native_runtime()
    builder, weights = _get_architecture_entry(architecture)
    model = builder(weights=weights if pretrained else None)

    if hasattr(model, "classifier"):
        classifier = getattr(model, "classifier")
        if not isinstance(classifier, nn.Sequential) or len(classifier) < 1:
            raise RuntimeError(f"Unsupported classifier head layout for architecture '{architecture}'")
        classifier_layers = list(classifier.children())
        linear_index = next(
            (
                index
                for index in range(len(classifier_layers) - 1, -1, -1)
                if isinstance(classifier_layers[index], nn.Linear)
            ),
            None,
        )
        if linear_index is None:
            raise RuntimeError(f"Classifier head for architecture '{architecture}' does not contain a Linear layer")

        input_features = classifier_layers[linear_index].in_features
        for layer in classifier_layers:
            if isinstance(layer, nn.Dropout):
                effective_dropout = (
                    float(dropout)
                    if dropout is not None and math.isfinite(float(dropout))
                    else float(getattr(layer, "p", 0.2))
                )
                layer.p = max(0.0, min(1.0, effective_dropout))
        classifier_layers[linear_index] = nn.Linear(input_features, int(num_classes))
        model.classifier = nn.Sequential(*classifier_layers)
    elif hasattr(model, "fc") and isinstance(getattr(model, "fc"), nn.Linear):
        input_features = getattr(model, "fc").in_features
        model.fc = nn.Linear(input_features, int(num_classes))
    else:
        raise RuntimeError(f"Unsupported classifier head layout for architecture '{architecture}'")
    return model, weights


class _ModelEMA:
    """Maintain an exponential moving average of model weights."""

    def __init__(self, model: Any, decay: float = 0.9999) -> None:
        self.ema = deepcopy(model).eval()
        self.decay = max(0.0, min(0.999999, float(decay)))
        self.updates = 0
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    def update(self, model: Any) -> None:
        self.updates += 1
        decay = self.decay * (1.0 - math.exp(-float(self.updates) / 2000.0))
        with torch.no_grad():
            model_state = model.state_dict()
            ema_state = self.ema.state_dict()
            for key, ema_value in ema_state.items():
                model_value = model_state[key].detach()
                if not torch.is_floating_point(ema_value):
                    ema_value.copy_(model_value)
                    continue
                ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)


class _ResizeIfNeeded:
    def __init__(self, image_size: int) -> None:
        self.image_size = max(int(image_size), 32)

    def __call__(self, image: Any) -> Any:
        width, height = image.size
        if int(width) == self.image_size and int(height) == self.image_size:
            return image
        return transforms.Resize(
            (self.image_size, self.image_size),
            interpolation=InterpolationMode.BILINEAR,
        )(image)


class _AddGaussianNoise:
    def __init__(self, noise_std: float) -> None:
        self.noise_std = max(0.0, min(1.0, float(noise_std)))

    def __call__(self, tensor: Any) -> Any:
        return torch.clamp(tensor + torch.randn_like(tensor) * self.noise_std, -4.0, 4.0)


def _build_transforms(
    *,
    image_size: int,
    mean: List[float],
    std: List[float],
    augmentation_enabled: bool = False,
    use_random_resized_crop: bool = False,
    use_horizontal_flip: bool = False,
    auto_augment: str = CLASSIFIER_AUTO_AUGMENT_NONE,
    degrees: float = 0.0,
    translate: float = 0.0,
    scale: float = 0.0,
    shear: float = 0.0,
    perspective: float = 0.0,
    flipud: float = 0.0,
    fliplr: float = 0.0,
    hsv_h: float = 0.0,
    hsv_s: float = 0.0,
    hsv_v: float = 0.0,
    erasing: float = 0.0,
    blur: float = 0.0,
    noise: float = 0.0,
) -> Tuple[Any, Any]:
    resize_size = max(int(image_size), 32)
    resize_if_needed = _ResizeIfNeeded(resize_size)
    normalized_auto_augment = normalize_classifier_auto_augment(auto_augment)

    train_transform_steps: List[Any] = []
    crop_scale_min = max(0.08, min(1.0, 1.0 - max(0.0, min(0.99, float(scale)))))
    crop_scale = (crop_scale_min, 1.0)
    crop_ratio = (3.0 / 4.0, 4.0 / 3.0)
    if augmentation_enabled and use_random_resized_crop:
        train_transform_steps.append(
            transforms.RandomResizedCrop(
                resize_size,
                scale=crop_scale,
                ratio=crop_ratio,
                interpolation=InterpolationMode.BILINEAR,
            )
        )
    else:
        train_transform_steps.append(resize_if_needed)

    if augmentation_enabled and use_horizontal_flip:
        train_transform_steps.append(transforms.RandomHorizontalFlip())
    if augmentation_enabled and float(fliplr) > 0.0:
        train_transform_steps.append(transforms.RandomHorizontalFlip(p=max(0.0, min(1.0, float(fliplr)))))
    if augmentation_enabled and float(flipud) > 0.0:
        train_transform_steps.append(RandomVerticalFlip(p=max(0.0, min(1.0, float(flipud)))))
    if augmentation_enabled and (
        float(degrees) > 0.0
        or float(translate) > 0.0
        or float(scale) > 0.0
        or float(shear) > 0.0
    ):
        scale_lower = max(0.08, 1.0 - max(0.0, min(0.99, float(scale))))
        scale_upper = 1.0 + max(0.0, float(scale))
        train_transform_steps.append(
            RandomAffine(
                degrees=max(0.0, float(degrees)),
                translate=(
                    max(0.0, min(1.0, float(translate))),
                    max(0.0, min(1.0, float(translate))),
                )
                if float(translate) > 0.0
                else None,
                scale=(scale_lower, scale_upper) if float(scale) > 0.0 else None,
                shear=max(0.0, float(shear)),
                interpolation=InterpolationMode.BILINEAR,
            )
        )
    if augmentation_enabled and float(perspective) > 0.0:
        train_transform_steps.append(
            RandomPerspective(
                distortion_scale=max(0.0, min(1.0, float(perspective) * 1000.0)),
                p=1.0,
                interpolation=InterpolationMode.BILINEAR,
            )
        )

    if augmentation_enabled and normalized_auto_augment:
        if normalized_auto_augment == "randaugment" and hasattr(transforms, "RandAugment"):
            train_transform_steps.append(transforms.RandAugment(interpolation=InterpolationMode.BILINEAR))
        elif normalized_auto_augment == "augmix" and hasattr(transforms, "AugMix"):
            train_transform_steps.append(transforms.AugMix(interpolation=InterpolationMode.BILINEAR))
        elif normalized_auto_augment == "autoaugment" and hasattr(transforms, "AutoAugment"):
            train_transform_steps.append(transforms.AutoAugment(interpolation=InterpolationMode.BILINEAR))
        else:
            logger.warning(
                "Auto augmentation policy '%s' is unavailable in this torchvision version; disabling it.",
                normalized_auto_augment,
            )

    color_jitter_kwargs: Dict[str, float] = {}
    if float(hsv_v) > 0.0:
        color_jitter_kwargs["brightness"] = max(0.0, min(1.0, float(hsv_v)))
        color_jitter_kwargs["contrast"] = max(0.0, min(1.0, float(hsv_v)))
    if float(hsv_s) > 0.0:
        color_jitter_kwargs["saturation"] = max(0.0, min(1.0, float(hsv_s)))
    if float(hsv_h) > 0.0:
        color_jitter_kwargs["hue"] = max(0.0, min(0.5, float(hsv_h)))
    if augmentation_enabled and color_jitter_kwargs and not normalized_auto_augment:
        train_transform_steps.append(ColorJitter(**color_jitter_kwargs))
    if augmentation_enabled and float(blur) > 0.0:
        blur_strength = max(0.1, min(5.0, float(blur) * 5.0))
        train_transform_steps.append(GaussianBlur(kernel_size=3, sigma=(0.1, blur_strength)))
    train_transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    if augmentation_enabled and float(noise) > 0.0:
        train_transform_steps.append(_AddGaussianNoise(noise_std=max(0.0, min(1.0, float(noise)))))
    if augmentation_enabled and float(erasing) > 0.0:
        train_transform_steps.append(RandomErasing(p=max(0.0, min(1.0, float(erasing))), inplace=True))
    train_transform = transforms.Compose(train_transform_steps)
    eval_transform = transforms.Compose(
        [
            resize_if_needed,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def build_native_classifier_eval_transform(
    image_size: int,
    *,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> Any:
    _require_native_runtime()
    _, eval_transform = _build_transforms(
        image_size=int(image_size),
        mean=list(mean or [0.485, 0.456, 0.406]),
        std=list(std or [0.229, 0.224, 0.225]),
        augmentation_enabled=False,
    )
    return eval_transform


def _build_optimizer(
    optimizer_name: str,
    model: Any,
    *,
    learning_rate: Optional[float],
    weight_decay: Optional[float],
    momentum: float = 0.9,
) -> Any:
    normalized = str(optimizer_name or "auto").strip().lower()
    lr_value = float(learning_rate if learning_rate is not None else 0.001)
    decay_value = float(weight_decay if weight_decay is not None else 0.0001)
    modules = model.module if hasattr(model, "module") else model
    norm_layers = tuple(value for key, value in nn.__dict__.items() if "Norm" in key)
    parameter_groups: List[Dict[str, Any]] = [{}, {}, {}]

    for module_name, module in modules.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            if "bias" in full_name:
                parameter_groups[2][full_name] = param
            elif isinstance(module, norm_layers) or "logit_scale" in full_name:
                parameter_groups[1][full_name] = param
            else:
                parameter_groups[0][full_name] = param

    if normalized in {"auto", "adamw", "adam", "adamax", "nadam", "radam"}:
        optimizer_kwargs = dict(lr=lr_value, betas=(float(momentum), 0.999), weight_decay=0.0)
    elif normalized == "rmsprop":
        optimizer_kwargs = dict(lr=lr_value, momentum=float(momentum))
    elif normalized == "sgd":
        optimizer_kwargs = dict(lr=lr_value, momentum=float(momentum), nesterov=True)
    else:
        raise ValueError("Optimizer must be one of: auto, SGD, Adam, AdamW, RMSProp.")

    group_dicts = [
        {"params": list(parameter_groups[0].values()), **optimizer_kwargs, "weight_decay": decay_value, "param_group": "weight"},
        {"params": list(parameter_groups[1].values()), **optimizer_kwargs, "weight_decay": 0.0, "param_group": "bn"},
        {"params": list(parameter_groups[2].values()), **optimizer_kwargs, "weight_decay": 0.0, "param_group": "bias"},
    ]
    group_dicts = [group for group in group_dicts if group["params"]]

    if normalized in {"auto", "adamw"}:
        return AdamW(group_dicts)
    if normalized == "adam":
        return Adam(group_dicts)
    if normalized == "sgd":
        return SGD(group_dicts)
    if normalized == "rmsprop":
        return RMSprop(group_dicts)
    raise ValueError("Optimizer must be one of: auto, SGD, Adam, AdamW, RMSProp.")


def _checkpoint_payload(
    *,
    model: Any,
    ema_model: Optional[Any],
    ema_updates: int,
    architecture: str,
    class_names: List[str],
    image_size: int,
    mean: List[float],
    std: List[float],
    epoch: int,
    metrics: Dict[str, float],
    optimizer_name: str,
    learning_rate: Optional[float],
    weight_decay: Optional[float],
    dropout: Optional[float],
    augmentation_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    source_model = ema_model if ema_model is not None else model
    return {
        "model_family": "native_pytorch",
        "model_type": "classification",
        "architecture": normalize_native_classifier_architecture(architecture),
        "state_dict": _state_dict_to_cpu(source_model, use_half=True),
        "class_names": list(class_names),
        "num_classes": len(class_names),
        "image_size": int(image_size),
        "normalization": {
            "mean": list(mean),
            "std": list(std),
        },
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "augmentation": dict(augmentation_config or {}),
        "uses_ema": bool(ema_model is not None),
        "ema_updates": int(ema_updates),
    }


class _TensorCacheDataset:
    """Tensor-backed dataset used to avoid repeated image decode work."""

    def __init__(self, tensors: Any, labels: Any) -> None:
        self.tensors = tensors
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return self.tensors[index], self.labels[index]


class _TensorBatchLoader:
    """Direct batch iterator for cached tensor datasets."""

    def __init__(
        self,
        dataset: _TensorCacheDataset,
        *,
        batch_size: int,
        shuffle: bool,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)

    def __len__(self) -> int:
        total = len(self.dataset)
        return int(math.ceil(total / self.batch_size)) if total > 0 else 0

    def __iter__(self):
        total = len(self.dataset)
        if total <= 0:
            return
        index_device = self.dataset.labels.device if hasattr(self.dataset.labels, "device") else "cpu"
        if self.shuffle:
            indices = torch.randperm(total, device=index_device)
        else:
            indices = torch.arange(total, device=index_device)
        for start in range(0, total, self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            yield (
                self.dataset.tensors.index_select(0, batch_indices),
                self.dataset.labels.index_select(0, batch_indices),
            )


def _estimate_tensor_cache_bytes(image_count: int, image_size: int) -> int:
    channels = 3
    bytes_per_value = 4
    return int(max(0, image_count) * channels * int(image_size) * int(image_size) * bytes_per_value)


def _summarize_active_native_augmentations(config: Dict[str, Any]) -> str:
    active_parts: List[str] = []
    if bool(config.get("random_resized_crop")):
        active_parts.append("random_resized_crop=True")
    if bool(config.get("horizontal_flip")):
        active_parts.append("horizontal_flip=True")
    auto_augment = normalize_classifier_auto_augment(config.get("auto_augment"))
    if auto_augment:
        active_parts.append(f"auto_augment={auto_augment}")
    for key in (
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "erasing",
        "blur",
        "noise",
    ):
        value = config.get(key)
        if isinstance(value, (int, float)) and abs(float(value)) > 1e-12:
            active_parts.append(f"{key}={value}")
    return ", ".join(active_parts) if active_parts else "none"


def _format_native_misclassification_lines(
    *,
    epoch: int,
    val_loss: float,
    sorted_classes: List[Tuple[str, int]],
    per_class_prediction_breakdown: Dict[str, Dict[str, Any]],
    total_confusion: int,
) -> List[str]:
    lines = [
        f"Confusion analysis (epoch {epoch}):",
        "Per-class misclassifications (worst to best):",
    ]
    for class_name, miss_count in sorted_classes:
        details = per_class_prediction_breakdown.get(class_name) or {}
        total_instances = int(details.get("total", 0) or 0)
        raw_predicted_as = details.get("predicted_as")
        predicted_as = raw_predicted_as if isinstance(raw_predicted_as, list) else []
        predicted_as_text = ", ".join(
            f"{predicted_name}x{int(predicted_count)}"
            for predicted_name, predicted_count in predicted_as
            if predicted_name
        )
        percentage = (float(miss_count) / float(total_instances) * 100.0) if total_instances > 0 else 0.0
        suffix = f" -> predicted as: {predicted_as_text}" if predicted_as_text else ""
        lines.append(f"{class_name}: {miss_count}/{total_instances} instances ({percentage:.1f}%){suffix}")
    if sorted_classes:
        lines.append(f"Worst class: {sorted_classes[0][0]} ({sorted_classes[0][1]} misclassifications)")
    lines.append(f"Total misclassifications: {int(total_confusion)}")
    lines.append(f"Validation loss: {float(val_loss):.4f}")
    return lines


def _build_native_misclassification_brief(summary: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(summary, dict):
        return None
    raw_mispredicted_classes = summary.get("mispredicted_classes")
    if not isinstance(raw_mispredicted_classes, list):
        return None
    parts = []
    for entry in raw_mispredicted_classes[:10]:
        if not isinstance(entry, dict):
            continue
        class_name = str(entry.get("class_name") or "").strip()
        misses = int(entry.get("misses", 0) or 0)
        if class_name:
            parts.append(f"{class_name}({misses})")
    if not parts:
        return "No validation mispredictions remain."
    return ", ".join(parts)


def _interp(start: float, end: float, progress: float) -> float:
    progress = min(1.0, max(0.0, float(progress)))
    return float(start) + (float(end) - float(start)) * progress


def _resolve_epoch_lr_factor(
    *,
    epoch_index: int,
    total_epochs: int,
    cosine_lr: bool,
    min_lr_ratio: float,
) -> float:
    progress = min(1.0, max(0.0, float(epoch_index) / float(max(1, total_epochs))))
    min_ratio = max(0.0, min(1.0, float(min_lr_ratio)))
    if cosine_lr:
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))
    return max(1.0 - progress, 0.0) * (1.0 - min_ratio) + min_ratio


def _apply_classifier_optimizer_schedule(
    optimizer: Any,
    base_lrs: List[float],
    *,
    epoch_index: int,
    total_epochs: int,
    batch_step: int,
    warmup_steps: int,
    nominal_batch_size: int,
    effective_batch_size: int,
    base_accumulate: int,
    momentum: float,
    warmup_momentum: float,
    warmup_bias_lr: float,
    cosine_lr: bool,
    min_lr_ratio: float,
) -> Tuple[float, int]:
    epoch_factor = _resolve_epoch_lr_factor(
        epoch_index=epoch_index,
        total_epochs=total_epochs,
        cosine_lr=cosine_lr,
        min_lr_ratio=min_lr_ratio,
    )
    current_accumulate = max(1, int(base_accumulate))
    if warmup_steps > 0 and batch_step <= warmup_steps:
        warmup_progress = min(1.0, max(0.0, float(batch_step) / float(warmup_steps)))
        target_accumulate = float(nominal_batch_size) / float(max(1, effective_batch_size))
        current_accumulate = max(1, int(round(_interp(1.0, target_accumulate, warmup_progress))))
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            warmup_start_lr = float(warmup_bias_lr) if group.get("param_group") == "bias" else 0.0
            group["lr"] = _interp(warmup_start_lr, float(base_lr) * epoch_factor, warmup_progress)
            if "momentum" in group:
                group["momentum"] = _interp(warmup_momentum, momentum, warmup_progress)
    else:
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = float(base_lr) * epoch_factor
            if "momentum" in group:
                group["momentum"] = float(momentum)
    reported_lr = float(optimizer.param_groups[0].get("lr", 0.0)) if optimizer.param_groups else 0.0
    return reported_lr, current_accumulate


def _prepare_tensor_cache_dataset(
    *,
    image_folder: Any,
    transform: Any,
    cache_dir: Path,
    split_name: str,
    cache_device: Optional[str] = None,
    use_channels_last: bool = False,
    log_callback: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Any:
    def _log(message: str) -> None:
        if log_callback:
            log_callback(message)

    def _check_cancelled() -> None:
        if should_cancel and should_cancel():
            raise RuntimeError("Training cancelled by user.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{split_name}_tensor_cache.pt"
    if cache_file.is_file():
        payload = torch.load(str(cache_file), map_location="cpu")
        if isinstance(payload, dict) and "tensors" in payload and "labels" in payload:
            _log(f"Loaded runtime tensor cache for {split_name} from {cache_file}.")
            tensors = payload["tensors"]
            labels = payload["labels"]
            if cache_device:
                _log(f"Moving {split_name} runtime tensor cache to {cache_device}.")
                tensors = tensors.to(
                    cache_device,
                    memory_format=torch.channels_last if use_channels_last else torch.contiguous_format,
                )
                labels = labels.to(cache_device)
            return _TensorCacheDataset(tensors, labels)

    tensor_rows: List[Any] = []
    label_rows: List[int] = []
    sample_count = len(getattr(image_folder, "samples", []) or [])
    _log(f"Populating runtime tensor cache for {split_name} ({sample_count} images).")
    for sample_index, (image_path, class_index) in enumerate(image_folder.samples):
        _check_cancelled()
        with Image.open(image_path) as image:
            tensor_rows.append(transform(image.convert("RGB")))
        label_rows.append(int(class_index))
        if sample_count > 0 and (sample_index + 1) % 250 == 0:
            _log(f"Cached {sample_index + 1}/{sample_count} {split_name} images.")

    tensors = torch.stack(tensor_rows) if tensor_rows else torch.empty((0, 3, 1, 1), dtype=torch.float32)
    labels = torch.tensor(label_rows, dtype=torch.long)
    torch.save({"tensors": tensors, "labels": labels}, str(cache_file))
    _log(f"Saved runtime tensor cache for {split_name} to {cache_file}.")
    if cache_device:
        _log(f"Moving {split_name} runtime tensor cache to {cache_device}.")
        tensors = tensors.to(
            cache_device,
            memory_format=torch.channels_last if use_channels_last else torch.contiguous_format,
        )
        labels = labels.to(cache_device)
    return _TensorCacheDataset(tensors, labels)


def train_native_classifier(
    *,
    source_root: str,
    output_root: str,
    architecture: str,
    epochs: int,
    batch_size: int,
    image_size: int,
    patience: int,
    optimizer_name: str,
    learning_rate: Optional[float],
    weight_decay: Optional[float],
    dropout: Optional[float],
    use_pretrained_weights: bool,
    augmentation_enabled: bool,
    augmentation_random_resized_crop: bool,
    augmentation_horizontal_flip: bool,
    auto_augment: str,
    mosaic: float,
    mixup: float,
    copy_paste: float,
    degrees: float,
    translate: float,
    scale: float,
    shear: float,
    perspective: float,
    flipud: float,
    fliplr: float,
    hsv_h: float,
    hsv_s: float,
    hsv_v: float,
    erasing: float,
    blur: float,
    noise: float,
    device: str,
    log_callback: Optional[Callable[[str], None]] = None,
    epoch_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Train a native torchvision classifier from ImageFolder train/val splits."""
    _require_native_runtime()
    recipe_momentum = 0.937
    warmup_epochs = 3.0
    warmup_momentum = 0.8
    warmup_bias_lr = 0.1
    final_lr_ratio = 0.01
    cosine_lr = False
    nominal_batch_size = 64

    def _log(message: str) -> None:
        if log_callback:
            log_callback(message)

    def _check_cancelled() -> None:
        if should_cancel and should_cancel():
            raise RuntimeError("Training cancelled by user.")

    source_path = Path(source_root)
    train_dir = source_path / "train"
    val_dir = source_path / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError("Expected ImageFolder train/ and val/ directories for native classifier training.")

    normalized_architecture = normalize_native_classifier_architecture(architecture)
    _, weights = _get_architecture_entry(normalized_architecture)
    mean, std = _resolve_normalization(weights)
    effective_image_size = max(32, int(image_size or _resolve_default_image_size(weights)))
    augmentation_config = {
        "use_pretrained_weights": bool(use_pretrained_weights),
        "enabled": bool(augmentation_enabled),
        "random_resized_crop": bool(augmentation_random_resized_crop),
        "horizontal_flip": bool(augmentation_horizontal_flip),
        "auto_augment": normalize_classifier_auto_augment(auto_augment),
        "mosaic": float(mosaic),
        "mixup": float(mixup),
        "copy_paste": float(copy_paste),
        "degrees": float(degrees),
        "translate": float(translate),
        "scale": float(scale),
        "shear": float(shear),
        "perspective": float(perspective),
        "flipud": float(flipud),
        "fliplr": float(fliplr),
        "hsv_h": float(hsv_h),
        "hsv_s": float(hsv_s),
        "hsv_v": float(hsv_v),
        "erasing": float(erasing),
        "blur": float(blur),
        "noise": float(noise),
    }
    train_transform, eval_transform = _build_transforms(
        image_size=effective_image_size,
        mean=mean,
        std=std,
        augmentation_enabled=bool(augmentation_enabled),
        use_random_resized_crop=bool(augmentation_random_resized_crop),
        use_horizontal_flip=bool(augmentation_horizontal_flip),
        auto_augment=str(augmentation_config["auto_augment"]),
        degrees=float(degrees),
        translate=float(translate),
        scale=float(scale),
        shear=float(shear),
        perspective=float(perspective),
        flipud=float(flipud),
        fliplr=float(fliplr),
        hsv_h=float(hsv_h),
        hsv_s=float(hsv_s),
        hsv_v=float(hsv_v),
        erasing=float(erasing),
        blur=float(blur),
        noise=float(noise),
    )
    unsupported_requested = {
        key: value
        for key, value in {
            "mosaic": float(mosaic),
            "mixup": float(mixup),
            "copy_paste": float(copy_paste),
        }.items()
        if abs(float(value)) > 1e-12
    }
    if unsupported_requested:
        _log(
            "Requested native classifier augmentations not applied because this classifier training "
            f"does not use them in its dataset pipeline: {unsupported_requested}"
        )

    raw_train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
    raw_val_dataset = datasets.ImageFolder(str(val_dir), transform=eval_transform)
    if not raw_train_dataset.classes:
        raise RuntimeError("No classifier classes found under train/.")

    class_names = list(raw_train_dataset.classes)
    if list(raw_val_dataset.classes) != class_names:
        raise RuntimeError("Train/val class folders do not match for native classifier training.")

    use_cuda = str(device).startswith("cuda") and bool(torch.cuda.is_available())
    runtime_cache_dir = Path(output_root) / "training_run" / "runtime_cache"
    train_image_count = len(getattr(raw_train_dataset, "samples", []) or [])
    val_image_count = len(getattr(raw_val_dataset, "samples", []) or [])
    total_image_count = train_image_count + val_image_count
    estimated_cache_bytes = _estimate_tensor_cache_bytes(total_image_count, effective_image_size)
    tensor_cache_threshold_bytes = 6 * 1024 * 1024 * 1024
    use_runtime_tensor_cache = (
        not bool(augmentation_enabled)
        and total_image_count > 0
        and estimated_cache_bytes <= tensor_cache_threshold_bytes
    )
    gpu_tensor_cache_threshold_bytes = 2 * 1024 * 1024 * 1024
    use_gpu_tensor_cache = (
        use_runtime_tensor_cache
        and use_cuda
        and estimated_cache_bytes <= gpu_tensor_cache_threshold_bytes
    )
    if use_runtime_tensor_cache:
        _log(
            "Runtime tensor cache enabled "
            f"({total_image_count} images, estimated {estimated_cache_bytes / (1024 ** 3):.2f} GiB, "
            f"gpu_cache={use_gpu_tensor_cache})."
        )
        train_dataset = _prepare_tensor_cache_dataset(
            image_folder=raw_train_dataset,
            transform=eval_transform,
            cache_dir=runtime_cache_dir,
            split_name="train",
            cache_device=device if use_gpu_tensor_cache else None,
            use_channels_last=use_gpu_tensor_cache,
            log_callback=log_callback,
            should_cancel=should_cancel,
        )
        val_dataset = _prepare_tensor_cache_dataset(
            image_folder=raw_val_dataset,
            transform=eval_transform,
            cache_dir=runtime_cache_dir,
            split_name="val",
            cache_device=device if use_gpu_tensor_cache else None,
            use_channels_last=use_gpu_tensor_cache,
            log_callback=log_callback,
            should_cancel=should_cancel,
        )
    else:
        train_dataset = raw_train_dataset
        val_dataset = raw_val_dataset
        if bool(augmentation_enabled):
            _log(
                "Runtime tensor cache disabled because training augmentation is enabled; "
                "caching transformed tensors would freeze stochastic augmentations across epochs."
            )
        else:
            _log(
                "Runtime tensor cache skipped "
                f"(estimated {estimated_cache_bytes / (1024 ** 3):.2f} GiB exceeds limit or dataset empty)."
            )

    cpu_count = max(1, int(os.cpu_count() or 1))
    worker_count = 0 if use_runtime_tensor_cache else min(8, max(2, cpu_count // 2))
    persistent_workers = worker_count > 0
    prefetch_factor = 4 if worker_count > 0 else None
    pin_memory = use_cuda and not use_gpu_tensor_cache
    effective_batch_size = max(1, int(batch_size))
    accumulate = max(round(nominal_batch_size / effective_batch_size), 1)
    scaled_weight_decay = (
        float(weight_decay if weight_decay is not None else 0.0005)
        * effective_batch_size
        * accumulate
        / nominal_batch_size
    )
    if use_runtime_tensor_cache:
        train_loader = _TensorBatchLoader(train_dataset, batch_size=effective_batch_size, shuffle=True)
        val_loader = _TensorBatchLoader(val_dataset, batch_size=effective_batch_size, shuffle=False)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            num_workers=worker_count,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            num_workers=worker_count,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    model, _ = build_native_classifier_model(
        normalized_architecture,
        num_classes=len(class_names),
        pretrained=bool(use_pretrained_weights),
        dropout=dropout,
    )
    use_amp = use_cuda
    scaler = _create_cuda_amp_grad_scaler(enabled=use_amp)
    if use_cuda:
        with contextlib.suppress(Exception):
            torch.backends.cudnn.benchmark = True
        with contextlib.suppress(Exception):
            torch.set_float32_matmul_precision("high")
        model = model.to(device=device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    ema = _ModelEMA(model)
    criterion = nn.CrossEntropyLoss()
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(
        optimizer_name,
        model,
        learning_rate=learning_rate,
        weight_decay=scaled_weight_decay,
        momentum=recipe_momentum,
    )
    base_lrs = [float(group.get("lr", learning_rate or 0.001)) for group in optimizer.param_groups]
    train_batches_per_epoch = max(1, len(train_loader))
    warmup_steps = max(100, int(round(float(warmup_epochs) * train_batches_per_epoch))) if warmup_epochs > 0 else -1
    latest_lr = base_lrs[0] if base_lrs else float(learning_rate or 0.001)

    training_run_dir = Path(output_root) / "training_run"
    weights_dir = training_run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    least_confused_epoch = 0
    best_confusion_score: Optional[Tuple[float, float, float]] = None
    least_confused_max_confusion: Optional[int] = None
    least_confused_total_confusion: Optional[int] = None
    least_confused_summary_payload: Optional[Dict[str, Any]] = None
    last_checkpoint_payload: Optional[Dict[str, Any]] = None
    epochs_without_least_confused_improvement = 0

    _log(
        "Starting native PyTorch classifier training with "
        f"{normalized_architecture} on {device} at image_size={effective_image_size}"
    )
    _log(
        "Runtime optimizations: "
        f"workers={worker_count}, persistent_workers={persistent_workers}, "
        f"prefetch_factor={prefetch_factor if prefetch_factor is not None else 0}, "
        f"pin_memory={pin_memory}, amp={use_amp}, channels_last={use_cuda}, "
        f"tensor_cache={use_runtime_tensor_cache}, gpu_tensor_cache={use_gpu_tensor_cache}, "
        f"batch_size={effective_batch_size}"
    )
    _log(
        "Training recipe: "
        f"warmup_epochs={warmup_epochs:.1f}, "
        f"warmup_momentum={warmup_momentum:.3f}, "
        f"warmup_bias_lr={warmup_bias_lr:.3f}, "
        f"momentum={recipe_momentum:.3f}, "
        f"lrf={final_lr_ratio:.3f}, "
        f"cosine_lr={cosine_lr}, "
        f"scaled_weight_decay={scaled_weight_decay:.6f}, accumulate={accumulate}, "
        f"active_augmentations={_summarize_active_native_augmentations(augmentation_config)}"
    )

    try:
        for epoch_index in range(max(1, int(epochs))):
            _check_cancelled()
            epoch_number = epoch_index + 1
            epoch_started = time.perf_counter()
            train_started = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)

            running_train_loss = 0.0
            running_train_correct = 0
            running_train_total = 0

            last_batch_index = max(0, len(train_loader) - 1)
            last_opt_step = -1
            for batch_index, (inputs, labels) in enumerate(train_loader):
                _check_cancelled()
                if use_gpu_tensor_cache:
                    pass
                elif use_cuda:
                    inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
                else:
                    inputs = inputs.to(device, non_blocking=True)
                if not use_gpu_tensor_cache:
                    labels = labels.to(device, non_blocking=True)
                batch_step = batch_index + train_batches_per_epoch * epoch_index
                latest_lr, current_accumulate = _apply_classifier_optimizer_schedule(
                    optimizer,
                    base_lrs,
                    epoch_index=epoch_index,
                    total_epochs=max(1, int(epochs)),
                    batch_step=batch_step,
                    warmup_steps=warmup_steps,
                    nominal_batch_size=nominal_batch_size,
                    effective_batch_size=effective_batch_size,
                    base_accumulate=accumulate,
                    momentum=recipe_momentum,
                    warmup_momentum=warmup_momentum,
                    warmup_bias_lr=warmup_bias_lr,
                    cosine_lr=cosine_lr,
                    min_lr_ratio=final_lr_ratio,
                )
                with _cuda_amp_autocast(enabled=use_amp):
                    logits = model(inputs)
                    loss = criterion(logits, labels)
                    step_loss = loss / float(current_accumulate)
                if scaler is not None:
                    scaler.scale(step_loss).backward()
                else:
                    step_loss.backward()

                should_step = ((batch_step - last_opt_step) >= current_accumulate) or (batch_index == last_batch_index)
                if should_step:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    ema.update(model)
                    optimizer.zero_grad(set_to_none=True)
                    last_opt_step = batch_step

                batch_size_value = int(labels.size(0))
                running_train_loss += float(loss.item()) * batch_size_value
                running_train_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
                running_train_total += batch_size_value

            train_loss = running_train_loss / running_train_total if running_train_total > 0 else 0.0
            train_accuracy = running_train_correct / running_train_total if running_train_total > 0 else 0.0
            train_duration_seconds = time.perf_counter() - train_started

            validation_started = time.perf_counter()
            model.eval()
            ema.ema.eval()
            eval_model = ema.ema
            running_val_loss = 0.0
            running_val_correct = 0
            running_val_total = 0
            confusion_matrix = torch.zeros((len(class_names), len(class_names)), dtype=torch.int64)

            with torch.no_grad():
                for inputs, labels in val_loader:
                    _check_cancelled()
                    if use_gpu_tensor_cache:
                        pass
                    elif use_cuda:
                        inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
                    else:
                        inputs = inputs.to(device, non_blocking=True)
                    if not use_gpu_tensor_cache:
                        labels = labels.to(device, non_blocking=True)
                    with _cuda_amp_autocast(enabled=use_amp):
                        logits = eval_model(inputs)
                        loss = eval_criterion(logits, labels)
                    predictions = torch.argmax(logits, dim=1)

                    batch_size_value = int(labels.size(0))
                    running_val_loss += float(loss.item()) * batch_size_value
                    running_val_correct += int((predictions == labels).sum().item())
                    running_val_total += batch_size_value

                    labels_cpu = labels.detach().cpu()
                    predictions_cpu = predictions.detach().cpu()
                    for predicted_index, true_index in zip(predictions_cpu.tolist(), labels_cpu.tolist()):
                        confusion_matrix[int(predicted_index), int(true_index)] += 1

            val_loss = running_val_loss / running_val_total if running_val_total > 0 else 0.0
            val_accuracy = running_val_correct / running_val_total if running_val_total > 0 else 0.0
            validation_duration_seconds = time.perf_counter() - validation_started

            metrics = {
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accuracy),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_accuracy),
                "train/loss": float(train_loss),
                "train/accuracy": float(train_accuracy),
                "val/loss": float(val_loss),
                "val/accuracy": float(val_accuracy),
                "epoch_seconds": 0.0,
                "train_seconds": float(train_duration_seconds),
                "val_seconds": float(validation_duration_seconds),
                "checkpoint_seconds": 0.0,
                "learning_rate": float(latest_lr),
                "lr": float(latest_lr),
            }

            per_class_misclassifications: Dict[str, int] = {}
            per_class_prediction_breakdown: Dict[str, Dict[str, Any]] = {}
            sorted_classes: List[Tuple[str, int]] = []
            if len(class_names) > 0:
                for class_index, class_name in enumerate(class_names):
                    total_true_instances = int(confusion_matrix[:, class_index].sum().item())
                    correct_predictions = int(confusion_matrix[class_index, class_index].item())
                    misclassified = total_true_instances - correct_predictions
                    per_class_misclassifications[class_name] = misclassified

                    predicted_as: List[Tuple[str, int]] = []
                    for predicted_index, predicted_name in enumerate(class_names):
                        if predicted_index == class_index:
                            continue
                        predicted_count = int(confusion_matrix[predicted_index, class_index].item())
                        if predicted_count > 0:
                            predicted_as.append((predicted_name, predicted_count))
                    predicted_as.sort(key=lambda item: item[1], reverse=True)
                    per_class_prediction_breakdown[class_name] = {
                        "total": total_true_instances,
                        "predicted_as": predicted_as,
                    }

                sorted_classes = sorted(
                    per_class_misclassifications.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )

            max_confusion = int(sorted_classes[0][1]) if sorted_classes else 0
            total_confusion = int(sum(value for _, value in sorted_classes))
            metrics["max_confusion"] = float(max_confusion)
            metrics["total_confusion"] = float(total_confusion)
            for line in _format_native_misclassification_lines(
                epoch=epoch_number,
                val_loss=float(val_loss),
                sorted_classes=sorted_classes,
                per_class_prediction_breakdown=per_class_prediction_breakdown,
                total_confusion=total_confusion,
            ):
                _log(line)

            checkpoint_started = time.perf_counter()
            last_checkpoint_payload = _checkpoint_payload(
                model=model,
                ema_model=ema.ema,
                ema_updates=ema.updates,
                architecture=normalized_architecture,
                class_names=class_names,
                image_size=effective_image_size,
                mean=mean,
                std=std,
                epoch=epoch_number,
                metrics=metrics,
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                dropout=dropout,
                augmentation_config=augmentation_config,
            )

            current_confusion_score = (
                float(total_confusion),
                float(max_confusion),
                float(val_loss),
            )
            if best_confusion_score is None or current_confusion_score < best_confusion_score:
                best_confusion_score = current_confusion_score
                best_val_loss = float(val_loss)
                least_confused_epoch = epoch_number
                least_confused_max_confusion = max_confusion
                least_confused_total_confusion = total_confusion
                least_confused_checkpoint = dict(last_checkpoint_payload)
                least_confused_checkpoint["max_confusion_score"] = max_confusion
                least_confused_checkpoint["total_confusion_score"] = total_confusion
                least_confused_checkpoint["per_class_scores"] = dict(per_class_misclassifications)
                least_confused_checkpoint["per_class_prediction_breakdown"] = {
                    class_name: {
                        "total_instances": details["total"],
                        "predicted_as": details["predicted_as"],
                    }
                    for class_name, details in per_class_prediction_breakdown.items()
                }
                least_confused_checkpoint["val_loss"] = float(val_loss)
                torch.save(least_confused_checkpoint, weights_dir / "least_confused.pt")
                least_confused_summary_payload = {
                    "epoch": epoch_number,
                    "max_confusion": max_confusion,
                    "total_confusion": total_confusion,
                    "val_loss": float(val_loss),
                    "has_mispredictions": total_confusion > 0,
                    "perfect_validation_confusion": total_confusion == 0,
                    "mispredicted_classes": [
                        {
                            "class_name": class_name,
                            "misses": miss_count,
                        }
                        for class_name, miss_count in sorted_classes
                        if miss_count > 0
                    ],
                    "per_class_misclassifications": per_class_misclassifications,
                    "per_class_prediction_breakdown": {
                        class_name: {
                            "total_instances": details["total"],
                            "predicted_as": details["predicted_as"],
                        }
                        for class_name, details in per_class_prediction_breakdown.items()
                    },
                }
                with (weights_dir / "least_confused_summary.json").open("w", encoding="utf-8") as handle:
                    json.dump(least_confused_summary_payload, handle, ensure_ascii=True, indent=2)
                epochs_without_least_confused_improvement = 0
                _log(
                    "Least-confused score improved to "
                    f"(total={total_confusion}, max={max_confusion}, val_loss={val_loss:.4f})"
                )
                _log(f"least_confused updated at epoch {epoch_number}: {weights_dir / 'least_confused.pt'}")
            else:
                epochs_without_least_confused_improvement += 1
            checkpoint_duration_seconds = time.perf_counter() - checkpoint_started
            epoch_duration_seconds = time.perf_counter() - epoch_started
            metrics["checkpoint_seconds"] = float(checkpoint_duration_seconds)
            metrics["epoch_seconds"] = float(epoch_duration_seconds)
            history.append({"epoch": float(epoch_number), **metrics})

            with (training_run_dir / "results.json").open("w", encoding="utf-8") as handle:
                json.dump(history, handle, ensure_ascii=True, indent=2)

            with (training_run_dir / "class_names.json").open("w", encoding="utf-8") as handle:
                json.dump(class_names, handle, ensure_ascii=True, indent=2)

            _log(
                f"Epoch {epoch_number}/{epochs} "
                f"(train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}, "
                f"max_conf={max_confusion}, total_conf={total_confusion}, "
                f"train_s={train_duration_seconds:.2f}, val_s={validation_duration_seconds:.2f}, "
                f"ckpt_s={checkpoint_duration_seconds:.2f})"
            )
            if epoch_callback:
                epoch_callback(
                    {
                        "epoch": epoch_number,
                        "total_epochs": int(epochs),
                        "metrics": metrics,
                    }
                )

            if patience > 0 and epochs_without_least_confused_improvement >= int(patience):
                _log(
                    f"Early stopping triggered after {epoch_number} epochs without "
                    "least-confused improvement."
                )
                break
    finally:
        if last_checkpoint_payload is not None:
            torch.save(last_checkpoint_payload, weights_dir / "last.pt")
        if runtime_cache_dir.is_dir():
            shutil.rmtree(runtime_cache_dir, ignore_errors=True)
            _log(f"Cleared runtime cache at {runtime_cache_dir}.")

    summary_payload = {
        "model_family": "native_pytorch",
        "architecture": normalized_architecture,
        "selection_metric": LEAST_CONFUSED_SELECTION_METRIC,
        "use_pretrained_weights": bool(use_pretrained_weights),
        "class_names": class_names,
        "image_size": effective_image_size,
        "augmentation": augmentation_config,
        "least_confused_val_loss": best_val_loss if math.isfinite(best_val_loss) else None,
        "least_confused_epoch": least_confused_epoch,
        "least_confused_max_confusion": least_confused_max_confusion,
        "least_confused_total_confusion": least_confused_total_confusion,
        "least_confused_summary_path": str(weights_dir / "least_confused_summary.json"),
        "least_confused_summary": least_confused_summary_payload,
        "least_confused_misclassification_brief": _build_native_misclassification_brief(least_confused_summary_payload),
        "history_length": len(history),
    }
    with (training_run_dir / "native_training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=True, indent=2)

    if least_confused_summary_payload is not None:
        if bool(least_confused_summary_payload.get("has_mispredictions")):
            mispredicted_classes = least_confused_summary_payload.get("mispredicted_classes") or []
            class_summary = ", ".join(
                f"{entry['class_name']}({entry['misses']})"
                for entry in mispredicted_classes[:10]
                if isinstance(entry, dict) and entry.get("class_name")
            )
            _log(
                "Final least-confused summary: validation mispredictions remain"
                + (f" in {class_summary}" if class_summary else ".")
            )
        else:
            _log("Final least-confused summary: no validation mispredictions remain.")
        for line in _format_native_misclassification_lines(
            epoch=int(least_confused_summary_payload.get("epoch", least_confused_epoch) or least_confused_epoch or 0),
            val_loss=float(least_confused_summary_payload.get("val_loss", best_val_loss) or 0.0),
            sorted_classes=[
                (str(entry.get("class_name") or ""), int(entry.get("misses", 0) or 0))
                for entry in (least_confused_summary_payload.get("mispredicted_classes") or [])
                if isinstance(entry, dict) and str(entry.get("class_name") or "").strip()
            ] or sorted(
                [
                    (str(class_name), int(miss_count or 0))
                    for class_name, miss_count in (least_confused_summary_payload.get("per_class_misclassifications") or {}).items()
                ],
                key=lambda item: item[1],
                reverse=True,
            ),
            per_class_prediction_breakdown={
                str(class_name): {
                    "total": int((details or {}).get("total_instances", (details or {}).get("total", 0)) or 0),
                    "predicted_as": [
                        (str(item[0]), int(item[1]))
                        for item in ((details or {}).get("predicted_as") or [])
                        if isinstance(item, (list, tuple)) and len(item) >= 2
                    ] if isinstance((details or {}).get("predicted_as"), list) else [],
                }
                for class_name, details in (least_confused_summary_payload.get("per_class_prediction_breakdown") or {}).items()
                if isinstance(details, dict)
            },
            total_confusion=int(least_confused_summary_payload.get("total_confusion", 0) or 0),
        ):
            _log(line)

    return {
        "save_dir": str(training_run_dir),
        "weights_dir": str(weights_dir),
        "class_names": class_names,
        "image_size": effective_image_size,
        "augmentation": augmentation_config,
        "last_model_path": str(weights_dir / "last.pt"),
        "least_confused_model_path": str(weights_dir / "least_confused.pt"),
        "least_confused_summary_path": str(weights_dir / "least_confused_summary.json"),
        "least_confused_summary": least_confused_summary_payload,
        "selection_metric": LEAST_CONFUSED_SELECTION_METRIC,
        "least_confused_val_loss": best_val_loss if math.isfinite(best_val_loss) else None,
        "least_confused_epoch": least_confused_epoch,
        "least_confused_max_confusion": least_confused_max_confusion,
        "least_confused_total_confusion": least_confused_total_confusion,
        "least_confused_misclassification_brief": _build_native_misclassification_brief(least_confused_summary_payload),
        "least_confused_has_mispredictions": (
            bool(least_confused_summary_payload.get("has_mispredictions"))
            if least_confused_summary_payload is not None
            else None
        ),
        "history": history,
    }


def resolve_native_classifier_model_path(model_path: str) -> str:
    """Resolve a native classifier checkpoint file from a file or directory path."""
    candidate = Path(model_path)
    if candidate.is_file():
        return str(candidate)
    if not candidate.is_dir():
        raise FileNotFoundError(f"Native classifier path does not exist: {model_path}")

    preferred_files = [
        candidate / "training_run" / "weights" / "least_confused.pt",
        candidate / "training_run" / "weights" / "last.pt",
        candidate / "training_run" / "weights" / "least_confused.pth",
        candidate / "training_run" / "weights" / "last.pth",
    ]
    for preferred in preferred_files:
        if preferred.is_file():
            return str(preferred)

    for checkpoint_name in ("least_confused.pt", "last.pt", "least_confused.pth", "last.pth"):
        matches = list(candidate.rglob(checkpoint_name))
        if matches:
            return str(matches[0])

    model_files = list(candidate.rglob("*.pth")) + list(candidate.rglob("*.pt"))
    if model_files:
        return str(model_files[0])

    raise FileNotFoundError(f"No native classifier checkpoint found in: {model_path}")


def load_native_classifier_checkpoint(model_path: str) -> Dict[str, Any]:
    _require_native_runtime()
    resolved_path = resolve_native_classifier_model_path(model_path)
    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise RuntimeError("Native classifier checkpoint must be a dict containing 'state_dict'.")
    architecture = normalize_native_classifier_architecture(checkpoint.get("architecture"))
    class_names = checkpoint.get("class_names")
    if not architecture:
        raise RuntimeError("Native classifier checkpoint is missing architecture metadata.")
    if not isinstance(class_names, list) or not class_names:
        raise RuntimeError("Native classifier checkpoint is missing class_names metadata.")
    return checkpoint


def _classifier_sidecar_candidates(model_path: str) -> List[Path]:
    model_dir = Path(model_path).parent
    model_name = Path(model_path).stem
    return [
        model_dir / f"{model_name}_class_mappings.json",
        model_dir / "class_mappings.json",
        model_dir / f"{model_name}.json",
    ]


def save_classifier_metadata_json(
    output_path: str,
    *,
    architecture: str,
    class_names: List[str],
    image_size: int,
    mean: List[float],
    std: List[float],
    extra_payload: Optional[Dict[str, Any]] = None,
) -> str:
    payload: Dict[str, Any] = {
        "class_mappings": {str(index): name for index, name in enumerate(class_names)},
        "model_type": "classification",
        "model_family": "native_pytorch",
        "architecture": normalize_native_classifier_architecture(architecture),
        "class_names": list(class_names),
        "image_size": int(image_size),
        "normalization": {
            "mean": list(mean),
            "std": list(std),
        },
    }
    if isinstance(extra_payload, dict):
        payload.update(extra_payload)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
    return output_path


def find_classifier_metadata_json(model_path: str) -> Optional[str]:
    for json_path in _classifier_sidecar_candidates(model_path):
        if json_path.exists():
            return str(json_path)
    return None


def load_classifier_metadata_json(model_path_or_json: str) -> Dict[str, Any]:
    candidate = Path(model_path_or_json)
    if candidate.is_file() and candidate.suffix.lower() == ".json":
        json_path = candidate
    else:
        resolved = find_classifier_metadata_json(model_path_or_json)
        if not resolved:
            return {}
        json_path = Path(resolved)

    try:
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def summarize_classifier_imagefolder_root(source_root: str) -> Dict[str, Any]:
    source_path = Path(source_root)
    train_dir = source_path / "train"
    val_dir = source_path / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError("Expected ImageFolder train/ and val/ directories.")

    def _class_names(split_dir: Path) -> List[str]:
        return sorted(path.name for path in split_dir.iterdir() if path.is_dir())

    def _count_images(split_dir: Path) -> int:
        count = 0
        for path in split_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in CLASSIFIER_IMAGE_EXTENSIONS:
                count += 1
        return count

    train_class_names = _class_names(train_dir)
    val_class_names = _class_names(val_dir)
    if not train_class_names:
        raise RuntimeError("No classifier classes found under train/.")
    if train_class_names != val_class_names:
        raise RuntimeError("Train/val class folders do not match for native classifier training.")

    train_images = _count_images(train_dir)
    val_images = _count_images(val_dir)
    return {
        "source_root": str(source_path),
        "train_images": int(train_images),
        "val_images": int(val_images),
        "total_images": int(train_images + val_images),
        "class_count": len(train_class_names),
        "class_names": train_class_names,
    }


def resolve_auto_classifier_optimizer(
    *,
    model_family: str,
    native_pytorch_architecture: str = "",
    yolo_size: str = "",
    class_count: int = 0,
    train_image_count: int = 0,
    batch_size: int = 16,
    epochs: int = 50,
) -> Dict[str, Any]:
    normalized_family = str(model_family or "").strip().lower()
    normalized_architecture = normalize_native_classifier_architecture(native_pytorch_architecture)
    normalized_yolo_size = str(yolo_size or "").strip().lower()

    if normalized_family == "native_pytorch":
        profile = dict(
            _NATIVE_CLASSIFIER_AUTO_OPTIMIZER_PROFILES.get(normalized_architecture)
            or {
                "optimizer": "AdamW",
                "learning_rate": 0.001,
                "weight_decay": 0.0005,
                "dropout": None,
                "profile": f"{normalized_architecture or 'native_pytorch'}:recipe_adamw",
            }
        )
        return {
            "optimizer": str(profile["optimizer"]),
            "learning_rate": float(profile["learning_rate"]),
            "weight_decay": float(profile["weight_decay"]),
            "dropout": (
                float(profile["dropout"])
                if profile.get("dropout") is not None
                else None
            ),
            "profile": str(profile["profile"]),
        }

    if normalized_yolo_size in {"m", "l", "x"}:
        return {
            "optimizer": "SGD",
            "learning_rate": 0.01,
            "weight_decay": 0.0005,
            "dropout": 0.0,
            "profile": f"yolo_{normalized_yolo_size}:sgd_classifier",
        }
    return {
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0005,
        "dropout": 0.0,
        "profile": f"yolo_{normalized_yolo_size or 'n'}:adamw_classifier",
    }


def resolve_native_classifier_training_options(
    *,
    source_root: str,
    architecture: str,
    epochs: int,
    batch_size: int,
    image_size: int,
    optimizer: str,
    learning_rate: Optional[float],
    weight_decay: Optional[float],
    dropout: Optional[float],
    augmentation_preset: str,
    augmentation_enabled: Optional[bool] = None,
    augmentation_random_resized_crop: Optional[bool] = None,
    augmentation_horizontal_flip: Optional[bool] = None,
    auto_augment: Optional[str] = None,
    mosaic: Optional[float] = None,
    mixup: Optional[float] = None,
    copy_paste: Optional[float] = None,
    degrees: Optional[float] = None,
    translate: Optional[float] = None,
    scale: Optional[float] = None,
    shear: Optional[float] = None,
    perspective: Optional[float] = None,
    flipud: Optional[float] = None,
    fliplr: Optional[float] = None,
    hsv_h: Optional[float] = None,
    hsv_s: Optional[float] = None,
    hsv_v: Optional[float] = None,
    erasing: Optional[float] = None,
    blur: Optional[float] = None,
    noise: Optional[float] = None,
) -> Dict[str, Any]:
    normalized_architecture = normalize_native_classifier_architecture(architecture)
    if normalized_architecture not in SUPPORTED_NATIVE_PYTORCH_CLASSIFIER_ARCHITECTURES:
        raise ValueError(
            "native_pytorch_architecture must be one of: "
            + ", ".join(SUPPORTED_NATIVE_PYTORCH_CLASSIFIER_ARCHITECTURES)
        )

    source_stats = summarize_classifier_imagefolder_root(source_root)
    normalized_preset = normalize_classifier_augmentation_preset(augmentation_preset)
    effective_augmentations = dict(_AUGMENTATION_PRESET_DEFAULTS[normalized_preset])
    overrides = {
        "augmentation_enabled": augmentation_enabled,
        "augmentation_random_resized_crop": augmentation_random_resized_crop,
        "augmentation_horizontal_flip": augmentation_horizontal_flip,
        "auto_augment": auto_augment,
        "mosaic": mosaic,
        "mixup": mixup,
        "copy_paste": copy_paste,
        "degrees": degrees,
        "translate": translate,
        "scale": scale,
        "shear": shear,
        "perspective": perspective,
        "flipud": flipud,
        "fliplr": fliplr,
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
        "erasing": erasing,
        "blur": blur,
        "noise": noise,
    }
    for key, value in overrides.items():
        if value is not None:
            effective_augmentations[key] = value
    effective_augmentations["auto_augment"] = normalize_classifier_auto_augment(
        effective_augmentations.get("auto_augment")
    )

    optimizer_input = str(optimizer or "auto").strip()
    optimizer_lookup = {
        "auto": "auto",
        "sgd": "SGD",
        "adam": "Adam",
        "adamw": "AdamW",
        "rmsprop": "RMSProp",
    }
    optimizer_normalized = optimizer_lookup.get(optimizer_input.lower())
    if optimizer_normalized is None:
        raise ValueError("Optimizer must be one of: auto, SGD, Adam, AdamW, RMSProp.")

    requested_optimizer = optimizer_normalized
    optimizer_profile: Optional[str] = None
    if optimizer_normalized == "auto":
        auto_optimizer = resolve_auto_classifier_optimizer(
            model_family="native_pytorch",
            native_pytorch_architecture=normalized_architecture,
            class_count=int(source_stats.get("class_count", 0) or 0),
            train_image_count=int(source_stats.get("train_images", 0) or 0),
            batch_size=int(batch_size),
            epochs=int(epochs),
        )
        optimizer_normalized = str(auto_optimizer["optimizer"])
        effective_learning_rate = (
            float(auto_optimizer["learning_rate"]) if auto_optimizer.get("learning_rate") is not None else None
        )
        effective_weight_decay = (
            float(auto_optimizer["weight_decay"]) if auto_optimizer.get("weight_decay") is not None else None
        )
        effective_dropout = float(auto_optimizer["dropout"]) if auto_optimizer.get("dropout") is not None else None
        optimizer_profile = str(auto_optimizer.get("profile") or "").strip() or None
    else:
        effective_learning_rate = float(learning_rate) if learning_rate is not None else 0.01
        effective_weight_decay = float(weight_decay) if weight_decay is not None else 0.0005
        effective_dropout = float(dropout) if dropout is not None else 0.0

    if optimizer_normalized == "auto":
        optimizer_normalized = "AdamW"
    if learning_rate is not None and requested_optimizer != "auto":
        effective_learning_rate = float(learning_rate)
    if weight_decay is not None and requested_optimizer != "auto":
        effective_weight_decay = float(weight_decay)
    if dropout is not None and requested_optimizer != "auto":
        effective_dropout = float(dropout)

    requested_image_size = int(image_size)
    effective_image_size = max(64, requested_image_size)

    return {
        "source_stats": source_stats,
        "architecture": normalized_architecture,
        "augmentation_preset": normalized_preset,
        "augmentations": effective_augmentations,
        "requested_optimizer": requested_optimizer,
        "optimizer": optimizer_normalized,
        "optimizer_profile": optimizer_profile,
        "learning_rate": effective_learning_rate,
        "weight_decay": effective_weight_decay,
        "dropout": effective_dropout,
        "requested_image_size": requested_image_size,
        "image_size": effective_image_size,
    }


def _default_export_output_path(checkpoint_path: str) -> Path:
    checkpoint = Path(resolve_native_classifier_model_path(checkpoint_path))
    model_name = checkpoint.stem
    if checkpoint.parent.name == "weights" and checkpoint.parent.parent.name == "training_run":
        model_name = checkpoint.parent.parent.parent.name or model_name
    output_dir = Path("onnx_engines")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{model_name}.onnx"


def export_native_classifier_to_onnx(
    *,
    checkpoint_path: str,
    output_path: Optional[str] = None,
    check: bool = False,
    simplify: bool = False,
    opset_version: int = 16,
    batch_size: int = 1,
) -> Dict[str, Any]:
    _require_native_runtime()
    checkpoint = load_native_classifier_checkpoint(checkpoint_path)
    architecture = normalize_native_classifier_architecture(checkpoint.get("architecture"))
    class_names = [str(name) for name in checkpoint.get("class_names", [])]
    normalization = checkpoint.get("normalization") if isinstance(checkpoint.get("normalization"), dict) else {}
    mean = normalization.get("mean") if isinstance(normalization.get("mean"), list) else [0.485, 0.456, 0.406]
    std = normalization.get("std") if isinstance(normalization.get("std"), list) else [0.229, 0.224, 0.225]
    image_size = int(checkpoint.get("image_size") or 224)

    model, _ = build_native_classifier_model(
        architecture,
        num_classes=len(class_names),
        pretrained=False,
        dropout=checkpoint.get("dropout"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    class ExportWrapper(nn.Module):
        def __init__(self, module: Any) -> None:
            super().__init__()
            self.module = module

        def forward(self, images: Any) -> Tuple[Any, Any]:
            logits = self.module(images)
            probabilities = torch.softmax(logits, dim=1)
            return logits, probabilities

    export_model = ExportWrapper(model)
    export_model.eval()

    effective_output_path = Path(output_path) if output_path else _default_export_output_path(checkpoint_path)
    effective_output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.rand(max(1, int(batch_size)), 3, image_size, image_size)
    with torch.no_grad():
        _ = export_model(dummy)

    export_kwargs = dict(
        input_names=["images"],
        output_names=["logits", "probabilities"],
        dynamic_axes={
            "images": {0: "N"},
            "logits": {0: "N"},
            "probabilities": {0: "N"},
        },
        opset_version=int(opset_version),
        verbose=False,
        do_constant_folding=True,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        export_model,
        dummy,
        str(effective_output_path),
        **export_kwargs,
    )

    metadata_path = effective_output_path.with_name(f"{effective_output_path.stem}_class_mappings.json")
    save_classifier_metadata_json(
        str(metadata_path),
        architecture=architecture,
        class_names=class_names,
        image_size=image_size,
        mean=[float(value) for value in mean],
        std=[float(value) for value in std],
        extra_payload={
            "checkpoint_path": str(resolve_native_classifier_model_path(checkpoint_path)),
            "exported_from": "visionhub-export-classifier-onnx",
        },
    )

    if check or simplify:
        import onnx  # type: ignore

        onnx_model = onnx.load(str(effective_output_path))
        if check:
            onnx.checker.check_model(onnx_model)

        if simplify:
            try:
                import onnxsim  # type: ignore
            except ModuleNotFoundError:
                logger.warning("Skipping ONNX simplification because `onnxsim` is not installed.")
            else:
                model_sim, passed = onnxsim.simplify(
                    onnx_model,
                    dynamic_input_shape=True,
                    input_shapes={"images": list(dummy.shape)},
                )
                if passed:
                    onnx.save(model_sim, str(effective_output_path))

    return {
        "onnx_path": str(effective_output_path),
        "metadata_path": str(metadata_path),
        "image_size": image_size,
        "class_names": class_names,
        "architecture": architecture,
    }


class NativePyTorchClassifier:
    """Inference adapter for native PyTorch classification checkpoints."""

    def __init__(self, model_path: str):
        _require_native_runtime()
        self.model_path = resolve_native_classifier_model_path(model_path)
        self.device = resolve_native_classifier_device("auto")
        self.model: Optional[Any] = None
        self.class_names: List[str] = []
        self.architecture: str = ""
        self.config: Dict[str, Any] = {
            "model_img_size": 384,
            "batch_size": 32,
            "store_visualisation_frames": True,
        }
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]
        self._transform = None

    def update_config(self, config: Dict[str, Any]) -> None:
        if "model_img_size" in config:
            try:
                parsed_size = int(config["model_img_size"])
                if parsed_size >= 16:
                    self.config["model_img_size"] = parsed_size
            except (TypeError, ValueError):
                pass
        if "batch_size" in config:
            try:
                parsed_batch = int(config["batch_size"])
                if parsed_batch >= 1:
                    self.config["batch_size"] = parsed_batch
            except (TypeError, ValueError):
                pass
        if "store_visualisation_frames" in config:
            self.config["store_visualisation_frames"] = bool(config["store_visualisation_frames"])
        if self._transform is not None:
            self._transform = build_native_classifier_eval_transform(
                int(self.config["model_img_size"]),
                mean=self._mean,
                std=self._std,
            )

    def load_model(self) -> bool:
        try:
            checkpoint = load_native_classifier_checkpoint(self.model_path)
            architecture = normalize_native_classifier_architecture(checkpoint.get("architecture"))
            class_names = checkpoint.get("class_names")
            self.architecture = architecture
            self.class_names = [str(name) for name in class_names]
            image_size = int(checkpoint.get("image_size") or self.config["model_img_size"])
            normalization = checkpoint.get("normalization") if isinstance(checkpoint.get("normalization"), dict) else {}
            mean = normalization.get("mean") if isinstance(normalization.get("mean"), list) else self._mean
            std = normalization.get("std") if isinstance(normalization.get("std"), list) else self._std
            self._mean = [float(value) for value in mean]
            self._std = [float(value) for value in std]
            self.config["model_img_size"] = image_size

            model, _ = build_native_classifier_model(
                architecture,
                num_classes=len(self.class_names),
                pretrained=False,
                dropout=checkpoint.get("dropout"),
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            if str(self.device).startswith("cuda"):
                with contextlib.suppress(Exception):
                    torch.cuda.set_device(self.device)
                self.model = model.to(self.device, memory_format=torch.channels_last)
            else:
                self.model = model.to(self.device)
            self._transform = build_native_classifier_eval_transform(
                int(self.config["model_img_size"]),
                mean=self._mean,
                std=self._std,
            )
            return True
        except Exception as exc:
            logger.error("Failed to load native classifier model %s: %s", self.model_path, exc)
            return False

    def process_classification_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return result

    def run_inference_batch(
        self,
        image_paths: List[str],
        *,
        topk: int = 5,
    ) -> Tuple[List[Dict[str, Any]], float]:
        if self.model is None or self._transform is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if not image_paths:
            return [], 0.0

        tensors = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                tensors.append(self._transform(image.convert("RGB")))

        batch = torch.stack(tensors)
        use_cuda = str(self.device).startswith("cuda") and bool(torch.cuda.is_available())
        if use_cuda:
            with contextlib.suppress(Exception):
                torch.cuda.set_device(self.device)
                torch.cuda.synchronize()
            batch = batch.to(self.device, non_blocking=True, memory_format=torch.channels_last)
        else:
            batch = batch.to(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            with _cuda_amp_autocast(enabled=use_cuda):
                logits = self.model(batch)
            probabilities = torch.softmax(logits, dim=1).detach().cpu()
        if use_cuda:
            with contextlib.suppress(Exception):
                torch.cuda.synchronize()
        duration_ms = (time.perf_counter() - started) * 1000.0

        results: List[Dict[str, Any]] = []
        effective_topk = max(1, min(int(topk), len(self.class_names)))
        for row in probabilities:
            top_values, top_indices = torch.topk(row, k=effective_topk)
            predictions: List[Dict[str, Any]] = []
            for confidence_value, class_index_value in zip(top_values.tolist(), top_indices.tolist()):
                class_index = int(class_index_value)
                predictions.append(
                    {
                        "class_id": class_index,
                        "class_name": (
                            self.class_names[class_index]
                            if 0 <= class_index < len(self.class_names)
                            else f"Unknown_{class_index}"
                        ),
                        "confidence": float(confidence_value),
                        "type": "classification",
                    }
                )
            overall_confidence = predictions[0]["confidence"] if predictions else 0.0
            results.append(
                {
                    "predictions": predictions,
                    "inference_type": "classifier",
                    "overall_confidence": overall_confidence,
                }
            )
        return results, duration_ms
