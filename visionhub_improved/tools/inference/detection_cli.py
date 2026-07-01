from __future__ import annotations

import argparse
import importlib
import sys
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

script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from visionhub.core import instantiate
from visionhub.detection_variants import (
    normalize_detection_family,
    normalize_detection_variant,
    resolve_detection_config_module,
)


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


class DetectionInference:
    def __init__(
        self,
        family,
        checkpoint_path,
        device="cuda",
        conf_thresh=0.35,
        nms_thresh=None,
        config_module=None,
        variant="s",
        use_ema=True,
        image_size=640,
        duplicate_iou_thresh=0.5,
    ):
        self.family = normalize_detection_family(family)
        self.variant = normalize_detection_variant(variant)
        requested_device = str(device).strip()
        self.device = (
            requested_device
            if torch.cuda.is_available() and requested_device.startswith("cuda")
            else "cpu"
        )
        self.conf_thresh = float(conf_thresh)
        self.duplicate_iou_thresh = float(duplicate_iou_thresh)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = resolve_checkpoint_architecture_metadata(
            ckpt,
            family_hint=self.family,
            use_ema=use_ema,
        )
        metadata_image_size = metadata.get("image_size")
        self.image_size = int(image_size or metadata_image_size or 640)
        state_dict = metadata["state_dict"]
        config_module = self._resolve_config_module(config_module, metadata)
        cfg = importlib.import_module(config_module)
        disable_pretrained_backbone_downloads(cfg)
        self._apply_checkpoint_metadata(cfg, metadata)

        self.model = instantiate(cfg.model)
        self.postprocessor = instantiate(cfg.postprocessor)
        if nms_thresh is not None:
            if hasattr(self.postprocessor, "nms_threshold"):
                self.postprocessor.nms_threshold = float(nms_thresh)
            if hasattr(self.postprocessor, "nms_iou_threshold"):
                self.postprocessor.nms_iou_threshold = float(nms_thresh)
        if hasattr(self.postprocessor, "score_threshold"):
            self.postprocessor.score_threshold = self.conf_thresh

        self.class_mappings = metadata["class_mappings"]
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def _resolve_config_module(self, explicit_config_module, metadata):
        candidates = [
            explicit_config_module,
            metadata.get("config_module"),
            metadata.get("config_file"),
            resolve_detection_config_module(self.family, self.variant),
        ]
        for candidate in candidates:
            if candidate and config_module_exists(candidate):
                return candidate
        raise ValueError(f"Could not resolve a config module for detection family '{self.family}'.")

    def _apply_checkpoint_metadata(self, cfg, metadata):
        num_classes = metadata.get("num_classes")
        if not num_classes and metadata.get("class_mappings"):
            num_classes = max(metadata["class_mappings"]) + 1

        if not num_classes:
            return

        if self.family == "detrdet":
            cfg.model.transformer.num_classes = num_classes
        else:
            cfg.model.num_classes = num_classes
        cfg.criterion.num_classes = num_classes

    def preprocess(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # Training uses RandomResize with an explicit (640, 640)-style target, which
        # stretches images to the square input size rather than letterboxing them.
        # Inference must mirror that preprocessing exactly or decoded boxes land in
        # the wrong place on non-square images.
        resized = cv2.resize(image_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        return tensor.to(self.device)

    @torch.no_grad()
    def infer(self, image_bgr):
        tensor = self.preprocess(image_bgr)
        outputs = self.model(tensor)
        h, w = image_bgr.shape[:2]
        target_sizes = torch.tensor([[h, w]], dtype=torch.float32, device=self.device)
        results = self.postprocessor(outputs, target_sizes)

        if isinstance(results, list):
            result = results[0]
        else:
            result = {
                "scores": results[0][0] if isinstance(results[0], torch.Tensor) and results[0].dim() > 1 else results["scores"][0],
                "labels": results[1][0] if isinstance(results[1], torch.Tensor) and results[1].dim() > 1 else results["labels"][0],
                "boxes": results[2][0] if isinstance(results[2], torch.Tensor) and results[2].dim() > 2 else results["boxes"][0],
            }

        scores = result["scores"].detach().cpu().numpy()
        labels = result["labels"].detach().cpu().numpy()
        boxes = result["boxes"].detach().cpu().numpy()

        keep = scores > self.conf_thresh
        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]

        detections = []
        for score, label, box in zip(scores, labels, boxes):
            detections.append(
                {
                    "score": float(score),
                    "label": int(label),
                    "label_name": self.class_mappings.get(int(label), f"class_{int(label)}"),
                    "box": box,
                }
            )
        return suppress_duplicate_detections(detections, self.duplicate_iou_thresh)

    def visualize(self, image_bgr, detections):
        img = image_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{det['label_name']} {det['score']:.2f}"
            cv2.putText(
                img,
                text,
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return img


def _parse_args(default_family: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{default_family} detection inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--config", type=str, default=None, help="Optional config module override")
    parser.add_argument("--variant", type=str, default="s", help="Model size variant shortcut")
    parser.add_argument("--source", type=str, required=True, help="Image file or directory")
    parser.add_argument("--output_dir", type=str, default="inference_output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--conf_thresh", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--nms_thresh", type=float, default=None, help="Optional NMS IoU override")
    parser.add_argument(
        "--duplicate_iou_thresh",
        "--duplicate-iou-thresh",
        dest="duplicate_iou_thresh",
        type=float,
        default=0.5,
        help="Suppress lower-confidence overlapping detections above this IoU after model postprocess.",
    )
    parser.add_argument("--image_size", type=int, default=None, help="Input image size (defaults to checkpoint metadata when available)")
    parser.add_argument("--no-ema", action="store_true", help="Use ckpt['model'] instead of EMA weights")
    parser.add_argument("--show", action="store_true", help="Show results interactively")
    parser.add_argument("--report-file", "--report_file", dest="report_file", type=str, default=None)
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--no-draw", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--draw-only", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    return parser.parse_args()


def _iter_image_paths(source_path: Path):
    if source_path.is_file():
        return [source_path]
    if source_path.is_dir():
        image_paths = []
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(source_path.rglob(f"*{ext}"))
            image_paths.extend(source_path.rglob(f"*{ext.upper()}"))
        return sorted(set(image_paths))
    raise ValueError(f"Source path not found: {source_path}")


def main(default_family: str) -> None:
    args = _parse_args(default_family)
    inferencer = DetectionInference(
        family=default_family,
        checkpoint_path=args.checkpoint,
        device=args.device,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        config_module=args.config,
        variant=args.variant,
        use_ema=not args.no_ema,
        image_size=args.image_size,
        duplicate_iou_thresh=args.duplicate_iou_thresh,
    )

    source_path = Path(args.source)
    image_paths = _iter_image_paths(source_path)
    if not image_paths:
        print(f"No images found in: {source_path}")
        return

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
        vis_image = inferencer.visualize(image, detections) if draw_images else None
        output_path = None
        if save_images:
            output_path = output_dir / f"{default_family}_{img_path.name}"
            cv2.imwrite(str(output_path), vis_image if vis_image is not None else image)

        emit_prediction_log(
            report_path=report_path,
            image_name=img_path.name,
            detections=[(det["label_name"], det["score"]) for det in detections],
            output_path=output_path,
            progress=progress,
        )

        if show_results and vis_image is not None:
            cv2.imshow(default_family, vis_image)
            key = cv2.waitKey(0)
            if key == ord("q"):
                break

    if show_results:
        cv2.destroyAllWindows()
    print(progress.summary_line())


if __name__ == "__main__":
    raise SystemExit("Use a family-specific wrapper module.")
