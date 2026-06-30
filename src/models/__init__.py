"""
🦡 Badger AI — One Unified Super-Advanced Vision Library.

One library. No v1/v2 split. Everything in clean, powerful files.

Quick start:
    from src.models import create_model
    model = create_model('badger-s')
"""
# --- Blocks (ALL in ONE file) ---
from .blocks import (
    Conv, Bottleneck, C2f, SPPF, DFL, make_divisible, autopad,
    GhostConv, DepthwiseSeparableConv, GhostBottleneck, GhostC2f,
    HardSwish, LightweightDetectHead,
    PConv, PConvBlock, RepVGGBlock, CIB, C2f_CIB,
    RepBottleneck, RepC2f,
    AreaAttention, A2Block, RELAN,
    DCNv4, DCNBottleneck, GELAN,
    DyHeadBlock, DyHead,
    fuse_model_reparam, profile_model, print_profile,
)
from .backbone import CSPDarknet
from .neck import PAFPN
from .neck_advanced import BiFPN, ASFF, NASFPN
from .head import DecoupledHead, CoupledHead, DualHead, HungarianMatcher, nms_free_postprocess
from .attention import WindowAttention, SwinBlock, GlobalTokenMixer, AttentionNeck
from .badger import (Badger, create_model,
                       BadgerV2, BackboneV2, BiFPNNeck, DecoupledHeadV2,
                       BiFPNLayer, create_badger_v2, compare_models)
from .multitask import BadgerMultiTask, KeypointHead, ClassificationHead, create_efficient_model
