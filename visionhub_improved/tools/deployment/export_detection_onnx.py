"""Family-aware detection ONNX export."""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys

import torch  # type: ignore
import torch.nn as nn  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from visionhub.core import LazyConfig, instantiate
from visionhub.detection_variants import resolve_detection_config_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../inference"))
try:
    from tools.inference.checkpoint_metadata import (
        config_module_to_file_path,
        resolve_checkpoint_architecture_metadata,
    )
    from tools.inference.class_mapping_utils import save_class_mappings_json  # type: ignore
except ImportError:
    from checkpoint_metadata import config_module_to_file_path, resolve_checkpoint_architecture_metadata
    from class_mapping_utils import save_class_mappings_json  # type: ignore


class DetectionExportWrapper(nn.Module):
    def __init__(self, model, postprocessor):
        super().__init__()
        self.model = model
        self.postprocessor = postprocessor

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)


def main(default_family: str) -> None:
    parser = argparse.ArgumentParser(description=f"Export {default_family} detection model to ONNX")
    parser.add_argument("--config_file", "--config", "-c", default=None)
    parser.add_argument("--variant", type=str, default="s")
    parser.add_argument("--resume", "-r", type=str, default=None)
    parser.add_argument("--no-ema", action="store_true", default=False)
    parser.add_argument("--check", action="store_true", default=False)
    parser.add_argument("--simplify", action="store_true", default=False)
    args = parser.parse_args()

    checkpoint_metadata = {}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        checkpoint_metadata = resolve_checkpoint_architecture_metadata(
            checkpoint,
            family_hint=default_family,
            use_ema=not args.no_ema,
        )
        state = checkpoint_metadata["state_dict"]
        class_mappings = checkpoint_metadata["class_mappings"]
    else:
        state = None
        class_mappings = {}

    config_file = (
        args.config_file
        or checkpoint_metadata.get("config_file")
        or config_module_to_file_path(checkpoint_metadata.get("config_module"))
        or resolve_detection_config_file(default_family, args.variant)
    )
    cfg = LazyConfig.load(config_file)
    num_classes = checkpoint_metadata.get("num_classes")
    if not num_classes and class_mappings:
        num_classes = max(class_mappings) + 1
    if num_classes:
        if default_family == "detrdet":
            cfg.model.transformer.num_classes = num_classes
        else:
            cfg.model.num_classes = num_classes
        cfg.criterion.num_classes = num_classes

    if hasattr(cfg.model.backbone, "pretrained"):
        cfg.model.backbone.pretrained = False

    model = instantiate(cfg.model)
    postprocessor = instantiate(cfg.postprocessor)
    if hasattr(postprocessor, "deploy"):
        postprocessor.deploy()
    elif hasattr(postprocessor, "deploy_mode"):
        postprocessor.deploy_mode = True

    if state is not None:
        model.load_state_dict(state, strict=True)
    model.eval()
    export_model = DetectionExportWrapper(model, postprocessor)
    export_model.eval()

    image_size = int(checkpoint_metadata.get("image_size") or 640)
    dummy_img = torch.rand(1, 3, image_size, image_size)
    dummy_size = torch.tensor([[image_size, image_size]])
    with torch.no_grad():
        _ = export_model(dummy_img, dummy_size)

    os.makedirs("onnx_engines", exist_ok=True)
    cfg_name = os.path.basename(config_file).replace(".py", "")
    output_file = f"onnx_engines/{cfg_name}.onnx"

    export_kwargs = dict(
        input_names=["images", "orig_target_sizes"],
        output_names=["scores", "labels", "boxes"],
        dynamic_axes={
            "images": {0: "N"},
            "orig_target_sizes": {0: "N"},
        },
        opset_version=16,
        verbose=False,
        do_constant_folding=True,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(export_model, (dummy_img, dummy_size), output_file, **export_kwargs)
    print(f"Exported ONNX model to: {output_file}")

    sidecar_metadata = {
        "family": default_family,
        "variant": checkpoint_metadata.get("model_size") or args.variant,
        "image_size": image_size,
        "config_module": checkpoint_metadata.get("config_module"),
        "config_file": config_file,
        "num_classes": num_classes,
    }
    if class_mappings or sidecar_metadata:
        json_path = output_file.replace(".onnx", "_class_mappings.json")
        save_class_mappings_json(class_mappings, json_path, model_metadata=sidecar_metadata)

    if args.check:
        import onnx  # type: ignore

        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed.")

    if args.simplify:
        import onnx  # type: ignore

        try:
            import onnxsim  # type: ignore
        except ModuleNotFoundError:
            print("Skipping ONNX simplification because `onnxsim` is not installed.")
            return
        input_shapes = {"images": [1, 3, image_size, image_size], "orig_target_sizes": [1, 2]}
        model_sim, check = onnxsim.simplify(output_file, test_input_shapes=input_shapes)
        onnx.save(model_sim, output_file)
        print(f"ONNX simplification: {check}")


if __name__ == "__main__":
    raise SystemExit("Use a family-specific wrapper module.")
