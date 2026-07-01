"""Detection family/variant resolution helpers."""

from __future__ import annotations

from typing import Optional


DETECTION_VARIANTS = {
    "detrdet": {
        "n": {
            "config_module": "configs.detrdet.detrdet_hgnetv2_n_custom",
            "config_file": "configs/detrdet/detrdet_hgnetv2_n_custom.py",
        },
        "s": {
            "config_module": "configs.detrdet.detrdet_hgnetv2_s_custom",
            "config_file": "configs/detrdet/detrdet_hgnetv2_s_custom.py",
        },
        "m": {
            "config_module": "configs.detrdet.detrdet_hgnetv2_m_custom",
            "config_file": "configs/detrdet/detrdet_hgnetv2_m_custom.py",
        },
        "l": {
            "config_module": "configs.detrdet.detrdet_hgnetv2_l_custom",
            "config_file": "configs/detrdet/detrdet_hgnetv2_l_custom.py",
        },
        "x": {
            "config_module": "configs.detrdet.detrdet_hgnetv2_x_custom",
            "config_file": "configs/detrdet/detrdet_hgnetv2_x_custom.py",
        },
    },
    "rtmodet": {
        "n": {
            "config_module": "configs.rtmodet.rtmodet_hgnetv2_n_custom",
            "config_file": "configs/rtmodet/rtmodet_hgnetv2_n_custom.py",
        },
        "s": {
            "config_module": "configs.rtmodet.rtmodet_hgnetv2_s_custom",
            "config_file": "configs/rtmodet/rtmodet_hgnetv2_s_custom.py",
        },
        "m": {
            "config_module": "configs.rtmodet.rtmodet_hgnetv2_m_custom",
            "config_file": "configs/rtmodet/rtmodet_hgnetv2_m_custom.py",
        },
        "l": {
            "config_module": "configs.rtmodet.rtmodet_hgnetv2_l_custom",
            "config_file": "configs/rtmodet/rtmodet_hgnetv2_l_custom.py",
        },
        "x": {
            "config_module": "configs.rtmodet.rtmodet_hgnetv2_x_custom",
            "config_file": "configs/rtmodet/rtmodet_hgnetv2_x_custom.py",
        },
    },
    "rtmdetdet": {
        "n": {
            "config_module": "configs.rtmdetdet.rtmdetdet_hgnetv2_n_custom",
            "config_file": "configs/rtmdetdet/rtmdetdet_hgnetv2_n_custom.py",
        },
        "s": {
            "config_module": "configs.rtmdetdet.rtmdetdet_hgnetv2_s_custom",
            "config_file": "configs/rtmdetdet/rtmdetdet_hgnetv2_s_custom.py",
        },
        "m": {
            "config_module": "configs.rtmdetdet.rtmdetdet_hgnetv2_m_custom",
            "config_file": "configs/rtmdetdet/rtmdetdet_hgnetv2_m_custom.py",
        },
        "l": {
            "config_module": "configs.rtmdetdet.rtmdetdet_hgnetv2_l_custom",
            "config_file": "configs/rtmdetdet/rtmdetdet_hgnetv2_l_custom.py",
        },
        "x": {
            "config_module": "configs.rtmdetdet.rtmdetdet_hgnetv2_x_custom",
            "config_file": "configs/rtmdetdet/rtmdetdet_hgnetv2_x_custom.py",
        },
    },
}

_FAMILY_ALIASES = {
    "detr-detect": "detrdet",
    "detr_detect": "detrdet",
    "detrdet": "detrdet",
    "rtmo-detect": "rtmodet",
    "rtmo_detect": "rtmodet",
    "rtmodet": "rtmodet",
    "rtmdet": "rtmdetdet",
    "rtmdet-detect": "rtmdetdet",
    "rtmdet_detect": "rtmdetdet",
    "rtmdetdet": "rtmdetdet",
}

_VARIANT_ALIASES = {
    "nano": "n",
    "small": "s",
    "medium": "m",
    "large": "l",
    "xlarge": "x",
    "xl": "x",
    "extra-large": "x",
    "extra_large": "x",
}


def normalize_detection_family(value: Optional[str]) -> str:
    family = str(value or "").strip().lower().replace(" ", "_")
    family = _FAMILY_ALIASES.get(family, family)
    if family not in DETECTION_VARIANTS:
        supported = ", ".join(sorted(DETECTION_VARIANTS))
        raise ValueError(f"Unsupported detection family '{value}'. Use one of: {supported}.")
    return family


def normalize_detection_variant(value: Optional[str], *, default: str = "s") -> str:
    variant = str(value or default).strip().lower().replace("-", "_")
    variant = _VARIANT_ALIASES.get(variant, variant)
    if variant not in {"n", "s", "m", "l", "x"}:
        raise ValueError(f"Unsupported detection variant '{value}'. Use one of: n, s, m, l, x.")
    return variant


def resolve_detection_config_module(family: str, variant: Optional[str] = None) -> str:
    normalized_family = normalize_detection_family(family)
    normalized_variant = normalize_detection_variant(variant)
    return DETECTION_VARIANTS[normalized_family][normalized_variant]["config_module"]


def resolve_detection_config_file(family: str, variant: Optional[str] = None) -> str:
    normalized_family = normalize_detection_family(family)
    normalized_variant = normalize_detection_variant(variant)
    return DETECTION_VARIANTS[normalized_family][normalized_variant]["config_file"]
