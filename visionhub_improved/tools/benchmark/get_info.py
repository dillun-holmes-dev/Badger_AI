"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from visionhub.core import LazyConfig, instantiate

import argparse
from calflops import calculate_flops

import torch
import torch.nn as nn

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

def main(args, ):
    """main
    """
    cfg = LazyConfig.load(args.config_file)
    
    if hasattr(cfg.model.backbone, 'pretrained'):
        cfg.model.backbone.pretrained = False

    model = instantiate(cfg.model)
    
    model = model.deploy()
    model.eval()

    flops, macs, _ = calculate_flops(model=model,
                                     input_shape=(1, 3, 640, 640),
                                     output_as_string=True,
                                     output_precision=4)
    params = sum(p.numel() for p in model.parameters())
    print("Model FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', '-c', default="configs/rtmdetpose/rtmdetpose_hgnetv2_s_custom.py", type=str)
    args = parser.parse_args()

    main(args)
