"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""
import time
import torch
from torch import nn
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import argparse
try:
    from .dataset import Dataset
except ImportError:
    from dataset import Dataset
from tqdm import tqdm

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from visionhub.core import LazyConfig, instantiate

def parse_args():
    parser = argparse.ArgumentParser(description='Argument Parser Example')
    parser.add_argument('--config_file', '-c', default='./configs/detrpose/detrpose_hgnetv2_l_custom.py', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--infer_dir',
                        type=str,
                        default='./data/COCO2017/val2017',
                        help="Directory for images to perform inference on.")
    args = parser.parse_args()
    return args

@torch.no_grad()
def warmup(model, data, img_size, n):
    for _ in range(n):
        _ = model(data, img_size)

@torch.no_grad()
def speed(model, data, n):
    times = []
    for i in tqdm(range(n), desc="Running Inference", unit="iteration"):
        blob = data[i]
        samples, target_sizes = blob['images'].unsqueeze(0), blob['orig_target_sizes']
        torch.cuda.synchronize()
        t_ = time.perf_counter()
        _ = model(samples, target_sizes)
        torch.cuda.synchronize()
        t = time.perf_counter() - t_
        times.append(t)

    # end-to-end model only
    times = sorted(times)
    if len(times) > 100:
        times = times[:100]
    return sum(times) / len(times)

def main():
    FLAGS = parse_args()
    dataset = Dataset(FLAGS.infer_dir)
    blob = torch.ones(1, 3, 640, 640).cuda()

    img_size = torch.tensor([[640, 640]], device='cuda')

    cfg = LazyConfig.load(FLAGS.config_file)
    
    if hasattr(cfg.model.backbone, 'pretrained'):
        cfg.model.backbone.pretrained = False

    model = instantiate(cfg.model)
    postprocessor = instantiate(cfg.postprocessor)
    if hasattr(postprocessor, "set_dcc") and hasattr(model, "dcc"):
        postprocessor.set_dcc(model.dcc)

    if FLAGS.resume:
        checkpoint = torch.load(FLAGS.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = model.deploy()
            self.postprocessor = postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().cuda()
    
    warmup(model, blob, img_size, 400)
    t = []
    for _ in range(1):
        t.append(speed(model, dataset, 1000))
    avg_latency = 1000 * torch.tensor(t).mean()
    print(f"model: {FLAGS.config_file}, Latency: {avg_latency:.2f} ms")

    del model
    torch.cuda.empty_cache()
    time.sleep(1)


if __name__ == '__main__':
    main()
