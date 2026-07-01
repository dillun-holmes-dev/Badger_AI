"""
RTMOPose PyTorch Inference Tool
Runs inference using a trained RTMOPose checkpoint.

Usage:
    visionhub-infer-rtmo \
        --checkpoint output/rtmopose_hgnetv2_s_custom/checkpoint_best_regular.pth \
        --config   configs.rtmopose.rtmopose_hgnetv2_s_custom \
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


class RTMOInference:
    def __init__(
        self,
        checkpoint_path,
        device="cuda",
        conf_thresh=0.35,
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
        self.image_size = image_size
        self.duplicate_iou_thresh = duplicate_iou_thresh

        print(f"Using device: {self.device}")

        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = resolve_checkpoint_architecture_metadata(
            ckpt,
            family_hint="rtmopose",
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
        if hasattr(self.postprocessor, "set_dcc") and hasattr(self.model, "dcc"):
            self.postprocessor.set_dcc(self.model.dcc)

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

    def _apply_checkpoint_metadata(self, cfg, inferred_shape):
        num_classes = inferred_shape.get("num_classes")
        num_body_points = inferred_shape.get("num_body_points")

        if num_classes:
            cfg.model.num_classes = num_classes
            cfg.criterion.num_classes = num_classes

        if num_body_points:
            cfg.model.num_body_points = num_body_points
            cfg.criterion.num_body_points = num_body_points
            cfg.criterion.sigmas = [0.05] * num_body_points

        if num_classes or num_body_points:
            print(
                "Patched RTMOPose config from checkpoint metadata: "
                f"num_classes={num_classes}, num_body_points={num_body_points}"
            )

    def _resolve_config_module(self, explicit_config_module, metadata):
        candidates = [
            explicit_config_module,
            metadata.get("config_module"),
            metadata.get("config_file"),
            "configs.rtmopose.rtmopose_hgnetv2_s_custom",
        ]
        for candidate in candidates:
            if candidate and config_module_exists(candidate):
                return candidate
        raise ValueError("Could not resolve an RTMOPose config module for this checkpoint.")

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
    def draw(self, image_bgr, detections, show_skeleton=True):
        """Annotate and return an image."""
        img = image_bgr.copy()
        palette = np.random.RandomState(42).randint(50, 220, (256, 3)).tolist()

        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            cid = det["label"] % 256
            color = palette[cid]

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{det['label_name']} {det['score']:.2f}",
                        (x1, max(y1 - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            kpts = det["keypoints"]
            K = len(kpts)
            for k in range(K):
                x, y, vis = kpts[k]
                if vis > 0.3:
                    cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)

            if show_skeleton:
                skel = self.skeleton_connections.get(
                    det["label"],
                    self.skeleton_connections.get(str(det["label"]), [])
                )
                for a, b in skel:
                    if a < K and b < K and kpts[a, 2] > 0.3 and kpts[b, 2] > 0.3:
                        cv2.line(img,
                                 (int(kpts[a, 0]), int(kpts[a, 1])),
                                 (int(kpts[b, 0]), int(kpts[b, 1])),
                                 (255, 200, 0), 2)
        return img


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="RTMOPose inference")
    p.add_argument("--checkpoint", "-r", required=True)
    p.add_argument("--config", "-c",
                   default=None)
    p.add_argument("--source", "-s", required=True,
                   help="Image file, directory, or glob pattern")
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf",   type=float, default=0.35)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--output", "-o", default="predictions")
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--show", action="store_true",
                   help="Display results with cv2.imshow (needs display)")
    p.add_argument("--report-file", "--report_file", dest="report_file", type=str, default=None,
                   help="Optional prediction report filename or path")
    p.add_argument("--duplicate-iou-thresh", type=float, default=0.5,
                   help="Suppress lower-confidence overlapping detections above this IoU")
    p.add_argument("--no-save-images", action="store_true",
                   help="Skip drawing and writing output images.")
    p.add_argument("--no-draw", action="store_true",
                   help="Skip annotation rendering. If images are still saved, raw inputs are written.")
    p.add_argument("--no-report", action="store_true",
                   help="Skip writing the prediction report file.")
    p.add_argument("--draw-only", action="store_true",
                   help="Render annotations in memory only. Skips image writes and report writing.")
    p.add_argument("--benchmark", action="store_true",
                   help="Fast path: skip image rendering, saving, and report writing.")
    return p.parse_args()


def _iter_images(source):
    """Recursively find images in source path."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    p = Path(source)
    if p.is_file():
        yield p
    elif p.is_dir():
        # Recursively search all subdirectories
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.suffix.lower() in exts:
                yield f
    else:
        import glob
        for f in sorted(glob.glob(source)):
            yield Path(f)


def main():
    args = _parse_args()
    infer = RTMOInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        conf_thresh=args.conf,
        config_module=args.config,
        use_ema=not args.no_ema,
        image_size=args.image_size,
        duplicate_iou_thresh=args.duplicate_iou_thresh,
    )

    out_dir = Path(args.output)
    draw_images = not (args.no_draw or args.benchmark)
    save_images = not (args.no_save_images or args.benchmark or args.draw_only)
    write_report = not (args.no_report or args.benchmark or args.draw_only)
    if save_images or write_report:
        out_dir.mkdir(parents=True, exist_ok=True)
    report_path = ensure_report_path(out_dir, args.report_file) if write_report else None
    image_paths = list(_iter_images(args.source))
    progress = InferenceProgress(total=len(image_paths))
    show_results = args.show and draw_images and not args.benchmark

    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            emit_prediction_log(
                report_path=report_path,
                image_name=img_path.name,
                detections=[],
                failure_message=f"Could not read: {img_path}",
                progress=progress,
            )
            continue

        detections = infer.infer(img_bgr)
        annotated = None
        out_path = None
        if draw_images:
            annotated = infer.draw(img_bgr, detections)
        if save_images:
            out_path = out_dir / img_path.name
            cv2.imwrite(str(out_path), annotated if annotated is not None else img_bgr)
        emit_prediction_log(
            report_path=report_path,
            image_name=img_path.name,
            detections=[(det["label_name"], det["score"]) for det in detections],
            output_path=out_path,
            progress=progress,
        )

        if show_results and annotated is not None:
            cv2.imshow("RTMOPose", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if show_results:
        cv2.destroyAllWindows()
    print(progress.summary_line())


if __name__ == "__main__":
    main()
