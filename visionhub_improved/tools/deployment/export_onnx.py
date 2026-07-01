"""
---------------------------------------------------------------------------------
Modified from D-FINE
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
import inspect
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from visionhub.core import LazyConfig, instantiate

import torch
import torch.nn as nn

# Import class mapping utilities
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../inference'))
try:
    from tools.inference.checkpoint_metadata import (
        config_module_exists,
        config_module_to_file_path,
        resolve_checkpoint_architecture_metadata,
    )
    from tools.inference.class_mapping_utils import save_class_mappings_json # type: ignore
except ImportError:
    try:
        from checkpoint_metadata import (
            config_module_exists,
            config_module_to_file_path,
            resolve_checkpoint_architecture_metadata,
        )
        from class_mapping_utils import save_class_mappings_json # type: ignore
    except ImportError:
        def save_class_mappings_json(mappings, path, **kwargs):
            import json
            with open(path, "w") as f:
                json.dump({"class_mappings": mappings, **kwargs}, f, indent=2)


DEFAULT_CONFIG_FILE = "configs/detrpose/detrpose_hgnetv2_s_custom.py"


def _resolve_config_file(args, checkpoint_metadata):
    candidates = [
        args.config_file,
        checkpoint_metadata.get("config_file"),
        config_module_to_file_path(checkpoint_metadata.get("config_module")),
        DEFAULT_CONFIG_FILE,
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("Could not resolve a DETRPose config file for export.")


def _apply_checkpoint_metadata(cfg, metadata):
    num_classes = metadata.get("num_classes")
    num_body_points = metadata.get("num_body_points")

    if num_classes:
        cfg.model.transformer.num_classes = num_classes
        cfg.criterion.num_classes = num_classes

    if num_body_points:
        cfg.model.transformer.num_body_points = num_body_points
        cfg.criterion.num_body_points = num_body_points
        cfg.criterion.matcher.num_body_points = num_body_points
        cfg.postprocessor.num_body_points = num_body_points


def _resolve_image_size(cfg, metadata):
    image_size = metadata.get("image_size")
    if image_size:
        return int(image_size)

    try:
        return int(cfg.model.encoder.eval_spatial_size[0])
    except Exception:
        return 640

def main(args, ):
    """main
    """
    checkpoint_metadata = {}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        checkpoint_metadata = resolve_checkpoint_architecture_metadata(
            checkpoint,
            family_hint="detrpose",
            use_ema=not args.no_ema,
        )
        state = checkpoint_metadata["state_dict"]
        class_mappings = checkpoint_metadata["class_mappings"]
        raw_skeletons = checkpoint_metadata["skeleton_connections"]
        skeleton_connections = {str(k): [list(pair) for pair in v] for k, v in raw_skeletons.items()}
    else:
        checkpoint = None
        state = None
        class_mappings = {}
        skeleton_connections = {}

    config_file = _resolve_config_file(args, checkpoint_metadata)
    print(f"Using config file: {config_file}")
    cfg = LazyConfig.load(config_file)
    _apply_checkpoint_metadata(cfg, checkpoint_metadata)
    
    if hasattr(cfg.model.backbone, 'pretrained'):
        cfg.model.backbone.pretrained = False

    model = instantiate(cfg.model)
    postprocessor = instantiate(cfg.postprocessor)
    if hasattr(postprocessor, "set_dcc") and hasattr(model, "dcc"):
        postprocessor.set_dcc(model.dcc)

    if state is not None:
        if class_mappings:
            print("Found class mappings in checkpoint - will save alongside ONNX model")
        if skeleton_connections:
            print(f"Found skeleton connections for {len(skeleton_connections)} class(es) - will save alongside ONNX model")
        model.load_state_dict(state)
    else:
        print('not load model.state_dict, use default init state dict...')

    model = model.deploy()
    model.eval()

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = model
            self.postprocessor = postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()

    image_size = _resolve_image_size(cfg, checkpoint_metadata)
    data = torch.rand(1, 3, image_size, image_size)
    size = torch.tensor([[image_size, image_size]])
    with torch.no_grad():
        _ = model(data, size)

    dynamic_axes = {
        'images': {0: 'N', },
        'orig_target_sizes': {0: 'N'}
    }

    outout_folder = 'onnx_engines'
    os.makedirs(outout_folder , exist_ok=True)
    output_file = os.path.basename(config_file).replace('.py', '.onnx')
    output_file = f'{outout_folder}/{output_file}'
    # args.resume.replace('.pth', '.onnx') if args.resume else 'model.onnx'

    export_kwargs = dict(
        input_names=['images', 'orig_target_sizes'],
        output_names=['scores', 'labels', 'keypoints', 'boxes'],
        dynamic_axes=dynamic_axes,
        opset_version=16,
        verbose=False,
        do_constant_folding=True,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        model,
        (data, size),
        output_file,
        **export_kwargs,
    )
    
    print(f"Exported ONNX model to: {output_file}")
    
    sidecar_metadata = {
        "family": checkpoint_metadata.get("family") or "detrpose",
        "variant": checkpoint_metadata.get("model_size"),
        "image_size": image_size,
        "config_module": checkpoint_metadata.get("config_module"),
        "config_file": config_file,
        "num_classes": checkpoint_metadata.get("num_classes"),
        "num_body_points": checkpoint_metadata.get("num_body_points"),
    }

    # Save class mappings, skeleton connections, and runtime metadata alongside ONNX model
    if class_mappings or skeleton_connections or sidecar_metadata:
        json_path = output_file.replace('.onnx', '_class_mappings.json')
        save_class_mappings_json(
            class_mappings,
            json_path,
            skeleton_connections=skeleton_connections,
            model_metadata=sidecar_metadata,
        )

    if args.check:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if args.simplify:
        import onnx
        try:
            import onnxsim
        except ModuleNotFoundError:
            print("Skipping ONNX simplification because `onnxsim` is not installed.")
            print("Install it with: pip install onnxsim")
            return
        dynamic = True
        # input_shapes = {'images': [1, 3, 640, 640], 'orig_target_sizes': [1, 2]} if dynamic else None
        input_shapes = {'images': data.shape, 'orig_target_sizes': size.shape} if dynamic else None
        onnx_model_simplify, check = onnxsim.simplify(output_file, test_input_shapes=input_shapes)
        onnx.save(onnx_model_simplify, output_file)
        print(f'Simplify onnx model {check}...')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', '--config', '-c',
                        default=None,
                        type=str)
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--no-ema', action='store_true', default=False)
    parser.add_argument('--check',  action='store_true', default=False,)
    parser.add_argument('--simplify',  action='store_true', default=False,)
    args = parser.parse_args()
    main(args)
