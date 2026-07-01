"""
RTMDetPose PyTorch Inference Tool
Runs inference using a trained RTMDetPose checkpoint.

Usage:
    visionhub-infer-rtmdetpose \
        --checkpoint output/rtmdetpose_hgnetv2_s_custom/checkpoint_best_regular.pth \
        --config   configs.rtmdetpose.rtmdetpose_hgnetv2_s_custom \
        --source   data/coco/val/images \
        --device   cuda
"""

import argparse
import importlib
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from .checkpoint_metadata import (
        disable_pretrained_backbone_downloads,
        config_module_exists,
        resolve_checkpoint_architecture_metadata,
    )
    from .detection_utils import suppress_duplicate_detections
    from .report_utils import InferenceProgress, emit_prediction_log, ensure_report_path
except ImportError:
    from checkpoint_metadata import (
        disable_pretrained_backbone_downloads,
        config_module_exists,
        resolve_checkpoint_architecture_metadata,
    )
    from detection_utils import suppress_duplicate_detections
    from report_utils import InferenceProgress, emit_prediction_log, ensure_report_path

# ── Path setup ────────────────────────────────────────────────────────────────
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from visionhub.core import instantiate


class RTMDetInference:
    def __init__(
        self,
        checkpoint_path,
        device="cuda",
        conf_thresh=0.35,
        nms_thresh=None,
        config_module=None,
        use_ema=True,
        image_size=640,
        duplicate_iou_thresh=0.5,
    ):
        requested_device = str(device).strip()
        self.device = (
            requested_device
            if torch.cuda.is_available() and requested_device.startswith("cuda")
            else "cpu"
        )
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.image_size = image_size
        self.duplicate_iou_thresh = duplicate_iou_thresh

        print(f"Using device: {self.device}")

        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = resolve_checkpoint_architecture_metadata(
            ckpt,
            family_hint="rtmdetpose",
            use_ema=use_ema,
        )
        state_dict = metadata["state_dict"]
        config_module = self._resolve_config_module(config_module, metadata)
        print(f"Loading config: {config_module}")

        cfg = importlib.import_module(config_module)
        disable_pretrained_backbone_downloads(cfg)
        self._apply_checkpoint_metadata(cfg, metadata)

        self.model = instantiate(cfg.model)
        self.postprocessor = instantiate(cfg.postprocessor)
        self.postprocessor.score_threshold = conf_thresh
        if nms_thresh is not None:
            self.postprocessor.nms_threshold = nms_thresh

        self.class_mappings = metadata["class_mappings"]
        if self.class_mappings:
            print("Loaded class mappings from checkpoint:")
            for cid, name in sorted(self.class_mappings.items()):
                print(f"  ID {cid}: {name}")
        else:
            print("Warning: No class mappings in checkpoint. Using numeric IDs.")

        self.skeleton_connections = metadata["skeleton_connections"]

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")

    def _resolve_config_module(self, explicit_config_module, metadata):
        candidates = [
            explicit_config_module,
            metadata.get("config_module"),
            metadata.get("config_file"),
            "configs.rtmdetpose.rtmdetpose_hgnetv2_s_custom",
        ]
        for candidate in candidates:
            if candidate and config_module_exists(candidate):
                return candidate
        raise ValueError("Could not resolve an RTMDetPose config module for this checkpoint.")

    def _apply_checkpoint_metadata(self, cfg, metadata):
        num_classes = metadata.get("num_classes")
        num_body_points = metadata.get("num_body_points")

        if num_classes:
            cfg.model.num_classes = num_classes
            cfg.criterion.num_classes = num_classes

        if num_body_points:
            cfg.model.num_body_points = num_body_points
            cfg.criterion.num_body_points = num_body_points
            cfg.criterion.sigmas = [0.05] * num_body_points

        if num_classes or num_body_points:
            print(
                "Patched RTMDetPose config from checkpoint metadata: "
                f"num_classes={num_classes}, num_body_points={num_body_points}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    def preprocess(self, image_bgr):
        """BGR uint8 -> RGB [1, 3, H, W] float32 tensor (values 0-1)."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        return tensor.to(self.device)

    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def infer(self, image_bgr):
        """
        Run inference on a single BGR image.

        Returns list of dicts:
            {'score', 'label', 'label_name', 'box' (xyxy), 'keypoints' (K, 3)}
        """
        tensor = self.preprocess(image_bgr)
        orig_h, orig_w = image_bgr.shape[:2]

        with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
            outputs = self.model(tensor)

        if isinstance(outputs, list):
            results_raw = outputs[0]
        else:
            # Match training-time stretch resize, then let the postprocessor map
            # predictions back to the original image size with independent x/y scales.
            orig_sizes = torch.tensor([[orig_h, orig_w]], device=self.device)
            results_raw = self.postprocessor(outputs, orig_sizes)[0]

        detections = []
        for i in range(len(results_raw["scores"])):
            score = results_raw["scores"][i].item()
            if score < self.conf_thresh:
                continue

            label = results_raw["labels"][i].item()
            box   = results_raw["boxes"][i].cpu().numpy()
            kpts  = results_raw["keypoints"][i].cpu().numpy()

            K = len(kpts) // 3
            kpts_arr = kpts.reshape(K, 3)

            detections.append({
                "score":      score,
                "label":      label,
                "label_name": self.class_mappings.get(label, str(label)),
                "box":        box,
                "keypoints":  kpts_arr,
            })

        return suppress_duplicate_detections(detections, self.duplicate_iou_thresh)

    # ─────────────────────────────────────────────────────────────────────────
    def visualize(self, image_bgr, detections, show_box=True, show_keypoints=True):
        """Draw detections on image."""
        img = image_bgr.copy()

        for det in detections:
            label_name = det["label_name"]
            score = det["score"]
            box = det["box"]
            kpts = det["keypoints"]

            if show_box:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label_name} {score:.2f}"
                cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 1)

            if show_keypoints:
                # Draw skeleton
                label = det["label"]
                skeleton = self.skeleton_connections.get(label, [])
                for pt_a, pt_b in skeleton:
                    if pt_a < len(kpts) and pt_b < len(kpts):
                        xa, ya, va = kpts[pt_a]
                        xb, yb, vb = kpts[pt_b]
                        if va > 0.3 and vb > 0.3:
                            cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)),
                                   (255, 128, 0), 2)

                # Draw keypoints
                for x, y, vis in kpts:
                    if vis > 0.3:
                        cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)

        return img


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    inferencer = RTMDetInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        config_module=args.config,
        use_ema=not args.no_ema,
        image_size=args.image_size,
        duplicate_iou_thresh=args.duplicate_iou_thresh,
    )

    source_path = Path(args.source)

    if source_path.is_file():
        image_paths = [source_path]
    elif source_path.is_dir():
        # Recursively search for images in all subdirectories
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            image_paths.extend(source_path.rglob(ext))
        image_paths = sorted(image_paths)
    else:
        raise ValueError(f"Source path not found: {source_path}")

    if not image_paths:
        print(f"No images found in: {source_path}")
        return

    print(f"Found {len(image_paths)} images")

    output_dir = Path(args.output_dir)
    draw_images = not (args.no_draw or args.benchmark)
    save_images = not (args.no_save_images or args.benchmark or args.draw_only)
    write_report = not (args.no_report or args.benchmark or args.draw_only)
    if save_images or write_report:
        output_dir.mkdir(exist_ok=True, parents=True)
    report_path = ensure_report_path(output_dir, args.report_file) if write_report else None
    progress = InferenceProgress(total=len(image_paths))
    show_results = args.show and draw_images and not args.benchmark

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            emit_prediction_log(
                report_path=report_path,
                image_name=img_path.name,
                detections=[],
                failure_message=f"Failed to load {img_path}",
                progress=progress,
            )
            continue

        detections = inferencer.infer(image)
        output_path = None
        vis_image = None
        if draw_images:
            vis_image = inferencer.visualize(image, detections)
        if save_images:
            output_path = output_dir / f"rtmdetpose_{img_path.name}"
            cv2.imwrite(str(output_path), vis_image if vis_image is not None else image)
        emit_prediction_log(
            report_path=report_path,
            image_name=img_path.name,
            detections=[(det["label_name"], det["score"]) for det in detections],
            output_path=output_path,
            progress=progress,
        )

        if show_results and vis_image is not None:
            cv2.imshow("RTMDetPose", vis_image)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break

    if show_results:
        cv2.destroyAllWindows()
    print(progress.summary_line())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTMDetPose Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file")
    parser.add_argument("--config", type=str,
                       default=None,
                       help="Config module (dot-separated)")
    parser.add_argument("--source", type=str, required=True,
                       help="Image file or directory")
    parser.add_argument("--output_dir", type=str, default="inference_output",
                       help="Output directory for visualizations")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"])
    parser.add_argument("--conf_thresh", type=float, default=0.35,
                       help="Confidence threshold")
    parser.add_argument("--nms_thresh", type=float, default=None,
                       help="NMS IoU threshold (default: use config value)")
    parser.add_argument("--image_size", type=int, default=640,
                       help="Input image size")
    parser.add_argument("--use_ema", action="store_true", default=None,
                       help="Deprecated; EMA weights are used by default")
    parser.add_argument("--no-ema", action="store_true",
                       help="Use ckpt['model'] instead of EMA weights if EMA exists")
    parser.add_argument("--show", action="store_true",
                       help="Show results interactively")
    parser.add_argument("--report-file", "--report_file", dest="report_file", type=str, default=None,
                       help="Optional prediction report filename or path")
    parser.add_argument("--duplicate-iou-thresh", type=float, default=0.5,
                       help="Suppress lower-confidence overlapping detections above this IoU")
    parser.add_argument("--no-save-images", action="store_true",
                       help="Skip drawing and writing output images.")
    parser.add_argument("--no-draw", action="store_true",
                       help="Skip annotation rendering. If images are still saved, raw inputs are written.")
    parser.add_argument("--no-report", action="store_true",
                       help="Skip writing the prediction report file.")
    parser.add_argument("--draw-only", action="store_true",
                       help="Render annotations in memory only. Skips image writes and report writing.")
    parser.add_argument("--benchmark", action="store_true",
                       help="Fast path: skip image rendering, saving, and report writing.")

    args = parser.parse_args()
    main(args)
