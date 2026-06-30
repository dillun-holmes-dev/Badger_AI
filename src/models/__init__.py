"""
Model package — exports all model components.

v1: Original Badger architecture (CSPDarknet + PAFPN + DecoupledHead)
v2: State-of-the-art architecture (PConv/RepC2f + BiFPN + DualHead)
"""
# --- v1 Components (backward compatible) ---
from .blocks import (Conv, Bottleneck, C2f, SPPF, DFL,
                     GhostConv, DepthwiseSeparableConv, GhostBottleneck,
                     GhostC2f, HardSwish, LightweightDetectHead)
from .backbone import CSPDarknet
from .neck import PAFPN
from .neck_advanced import BiFPN, ASFF, NASFPN
from .head import DecoupledHead, CoupledHead
from .attention import WindowAttention, SwinBlock, GlobalTokenMixer, AttentionNeck
from .badger import Badger, create_model
from .multitask import BadgerMultiTask, KeypointHead, ClassificationHead, create_efficient_model

# --- v2 Components (state-of-the-art) ---
from .blocks_v2 import (PConv, PConvBlock, RepVGGBlock, CIB, C2f_CIB,
                         RepBottleneck, RepC2f, AreaAttention, A2Block, RELAN,
                         DCNv4, DCNBottleneck, GELAN, DyHeadBlock, DyHead,
                         fuse_model_reparam, profile_model, print_profile)
from .badger_v2 import (BadgerV2, BackboneV2, BiFPNNeck, DecoupledHeadV2,
                         create_badger_v2, compare_models)
from .head_v2 import DualHead, HungarianMatcher, nms_free_postprocess
