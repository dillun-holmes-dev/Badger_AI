"""
RTMDetPose ONNX Export Tool
Exports a trained RTMDetPose checkpoint to an ONNX model.

Usage:
    visionhub-export-rtmdetpose-onnx \
        --config  configs/rtmdetpose/rtmdetpose_hgnetv2_s_custom.py \
        --resume  output/rtmdetpose_hgnetv2_s_custom/checkpoint_best_regular.pth \
        --check --simplify

The exported ONNX model has:
    Inputs:  images [N, 3, 640, 640], orig_target_sizes [N, 2]
    Outputs: scores, labels, keypoints, boxes  (same as DETRPose/RTMOPose ONNX)
"""

import os
import sys
import inspect
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import torch # type: ignore
import torch.nn as nn # type: ignore

from visionhub.core import LazyConfig, instantiate

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../inference"))
try:
    from tools.inference.checkpoint_metadata import (
        config_module_to_file_path,
        resolve_checkpoint_architecture_metadata,
    )
    from tools.inference.class_mapping_utils import save_class_mappings_json  # type: ignore
except ImportError:
    try:
        from checkpoint_metadata import (
            config_module_to_file_path,
            resolve_checkpoint_architecture_metadata,
        )
        from class_mapping_utils import save_class_mappings_json  # type: ignore
    except ImportError:
        def save_class_mappings_json(mappings, path, **kwargs):
            import json
            with open(path, "w") as f:
                json.dump({"class_mappings": mappings, **kwargs}, f, indent=2)


DEFAULT_CONFIG_FILE = "configs/rtmdetpose/rtmdetpose_hgnetv2_s_custom.py"


# ── Exportable wrapper ────────────────────────────────────────────────────────

class RTMDetExportWrapper(nn.Module):
    """
    Thin wrapper that runs the model in eval mode and returns
    flat tensors for ONNX export:
        scores, labels, keypoints, boxes
    """

    def __init__(self, model, postprocessor):
        super().__init__()
        self.model = model
        self.postprocessor = postprocessor

    def forward(self, images, orig_target_sizes):
        """
        images:            [N, 3, H, W]
        orig_target_sizes: [N, 2]  (H, W per image)

        Returns (scores, labels, keypoints, boxes) – each [N, max_det] or [N, max_det, ...]
        Padded to max_det = 300.
        """
        outputs = self.model(images)
        results = self.postprocessor(outputs, orig_target_sizes)

        if isinstance(results, tuple):
            return results

        # Stack into batch tensors (padded to max_detections)
        max_det = self.postprocessor.max_detections
        B = len(results)
        device = images.device

        scores_out    = torch.zeros(B, max_det, device=device)
        labels_out    = torch.zeros(B, max_det, dtype=torch.int64, device=device)
        boxes_out     = torch.zeros(B, max_det, 4, device=device)
        K = results[0]["keypoints"].shape[-1]
        keypoints_out = torch.zeros(B, max_det, K, device=device)

        for i, r in enumerate(results):
            n = min(r["scores"].shape[0], max_det)
            scores_out[i,    :n]    = r["scores"][:n]
            labels_out[i,    :n]    = r["labels"][:n]
            boxes_out[i,     :n]    = r["boxes"][:n]
            keypoints_out[i, :n]    = r["keypoints"][:n]

        return scores_out, labels_out, keypoints_out, boxes_out


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    checkpoint_metadata = {}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        checkpoint_metadata = resolve_checkpoint_architecture_metadata(
            checkpoint,
            family_hint="rtmdetpose",
            use_ema=not args.no_ema,
        )
        state = checkpoint_metadata["state_dict"]
        class_mappings = checkpoint_metadata["class_mappings"]
        raw_skeletons = checkpoint_metadata["skeleton_connections"]
        skeleton_connections = {str(k): [list(pair) for pair in v] for k, v in raw_skeletons.items()}
    else:
        state = None
        class_mappings = {}
        skeleton_connections = {}

    config_file = (
        args.config_file
        or checkpoint_metadata.get("config_file")
        or config_module_to_file_path(checkpoint_metadata.get("config_module"))
        or DEFAULT_CONFIG_FILE
    )
    cfg = LazyConfig.load(config_file)
    num_classes = checkpoint_metadata.get("num_classes")
    num_body_points = checkpoint_metadata.get("num_body_points")
    if num_classes:
        cfg.model.num_classes = num_classes
        cfg.criterion.num_classes = num_classes
    if num_body_points:
        cfg.model.num_body_points = num_body_points
        cfg.criterion.num_body_points = num_body_points
        cfg.criterion.sigmas = [0.05] * num_body_points

    if hasattr(cfg.model.backbone, "pretrained"):
        cfg.model.backbone.pretrained = False

    model = instantiate(cfg.model)
    postprocessor = instantiate(cfg.postprocessor)
    postprocessor.deploy_mode = True

    if state is not None:
        if class_mappings:
            print(f"Loaded class mappings: {class_mappings}")
        model.load_state_dict(state)
    else:
        print("Warning: no checkpoint provided; exporting with random weights.")

    model.eval()

    export_model = RTMDetExportWrapper(model, postprocessor)
    export_model.eval()

    # Warm-up pass
    image_size = int(checkpoint_metadata.get("image_size") or 640)
    dummy_img  = torch.rand(1, 3, image_size, image_size)
    dummy_size = torch.tensor([[image_size, image_size]])
    with torch.no_grad():
        _ = export_model(dummy_img, dummy_size)

    os.makedirs("onnx_engines", exist_ok=True)
    cfg_name = os.path.basename(config_file).replace(".py", "")
    output_file = f"onnx_engines/{cfg_name}.onnx"

    dynamic_axes = {
        "images":            {0: "N"},
        "orig_target_sizes": {0: "N"},
    }

    export_kwargs = dict(
        input_names=["images", "orig_target_sizes"],
        output_names=["scores", "labels", "keypoints", "boxes"],
        dynamic_axes=dynamic_axes,
        opset_version=16,
        verbose=False,
        do_constant_folding=True,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        export_model,
        (dummy_img, dummy_size),
        output_file,
        **export_kwargs,
    )
    print(f"Exported ONNX model to: {output_file}")

    if class_mappings or skeleton_connections:
        json_path = output_file.replace(".onnx", "_class_mappings.json")
        save_class_mappings_json(class_mappings, json_path, skeleton_connections=skeleton_connections)
        print(f"Saved class mappings to: {json_path}")

    if args.check:
        import onnx # type: ignore
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed.")

    if args.simplify:
        import onnx # type: ignore
        try:
            import onnxsim # type: ignore
        except ModuleNotFoundError:
            print("Skipping ONNX simplification because `onnxsim` is not installed.")
            print("Install it with: pip install onnxsim")
            return
        input_shapes = {"images": [1, 3, 640, 640], "orig_target_sizes": [1, 2]}
        model_sim, check = onnxsim.simplify(output_file, test_input_shapes=input_shapes)
        onnx.save(model_sim, output_file)
        print(f"ONNX simplification: {check}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export RTMDetPose to ONNX")
    parser.add_argument("--config_file", "--config", "-c",
                        default=None)
    parser.add_argument("--resume", "-r", type=str, default=None)
    parser.add_argument("--no-ema", action="store_true", default=False)
    parser.add_argument("--check", action="store_true", default=False)
    parser.add_argument("--simplify", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
