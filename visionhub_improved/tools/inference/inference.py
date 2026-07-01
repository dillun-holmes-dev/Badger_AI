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
        extract_state_dict,
        infer_pose_model_shape,
        resolve_checkpoint_architecture_metadata,
        sanitize_state_dict_keys,
    )
    from .detection_utils import class_agnostic_nms_indices
    from .report_utils import InferenceProgress, emit_prediction_log, ensure_report_path
except ImportError:
    from checkpoint_metadata import (
        disable_pretrained_backbone_downloads,
        config_module_exists,
        extract_state_dict,
        infer_pose_model_shape,
        resolve_checkpoint_architecture_metadata,
        sanitize_state_dict_keys,
    )
    from detection_utils import class_agnostic_nms_indices
    from report_utils import InferenceProgress, emit_prediction_log, ensure_report_path

# ==================== PATH SETUP ====================
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
print(f"Project root: {project_root}")

from visionhub.core import instantiate


# Deprecated hardcoded class names - will be replaced by checkpoint class_mappings
# CLASS_NAMES = {
#     2: "black_bx_side",
#     ...
# }


class DETRPoseInference:
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
            family_hint="detrpose",
            use_ema=use_ema,
        )
        state_dict = metadata["state_dict"]
        config_module = self._resolve_config_module(config_module, metadata)
        print(f"Loading config: {config_module}")

        # Import as a package module so relative imports inside the config work.
        cfg = importlib.import_module(config_module)
        disable_pretrained_backbone_downloads(cfg)
        self._apply_checkpoint_metadata(cfg, metadata)

        # Build the exact model/postprocessor from the same config used for training.
        self.model = instantiate(cfg.model)
        self.postprocessor = instantiate(cfg.postprocessor)
        if hasattr(self.postprocessor, "set_dcc") and hasattr(self.model, "dcc"):
            self.postprocessor.set_dcc(self.model.dcc)

        # Load class mappings from checkpoint
        self.class_mappings = metadata["class_mappings"]
        if self.class_mappings:
            print(f"\n{'='*60}")
            print("Loaded class mappings from checkpoint:")
            for class_id, class_name in sorted(self.class_mappings.items()):
                print(f"  ID {class_id}: {class_name}")
            print(f"{'='*60}\n")
        else:
            print("\nWarning: No class mappings found in checkpoint. Using numeric IDs.\n")

        # Load skeleton connections from checkpoint: {class_id: [[a, b], ...]}
        self.skeleton_connections = metadata["skeleton_connections"]
        if self.skeleton_connections:
            print("Loaded skeleton connections from checkpoint.")
        else:
            print("Warning: No skeleton connections in checkpoint. Using linear chain fallback.")

        print("Loading model weights...")
        self.model.load_state_dict(state_dict, strict=True)

        self.model.to(self.device)
        self.model.eval()

        print("✅ Model loaded successfully!")

    def _resolve_config_module(self, explicit_config_module, metadata):
        candidates = [
            explicit_config_module,
            metadata.get("config_module"),
            metadata.get("config_file"),
            auto_detect_config(metadata=metadata),
            "configs.detrpose.detrpose_hgnetv2_s_custom",
        ]
        for candidate in candidates:
            if candidate and config_module_exists(candidate):
                return candidate
        raise ValueError("Could not resolve a DETRPose config module for this checkpoint.")

    def _apply_checkpoint_metadata(self, cfg, metadata):
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

        if num_classes or num_body_points:
            print(
                "Patched DETRPose config from checkpoint metadata: "
                f"num_classes={num_classes}, num_body_points={num_body_points}"
            )

    def preprocess(self, image):
        """
        Manual preprocessing to avoid relying on visionhub.data.transforms.Compose.
        Output shape: [1, 3, image_size, image_size]
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(
            img_rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        tensor = torch.from_numpy(img_resized).float()
        tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        tensor = tensor / 255.0
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def get_object_color(self, index):
        """
        Deterministic BGR colour per detected object.
        OpenCV uses BGR, not RGB.
        """
        colors = [
            (0, 255, 0),      # green
            (255, 0, 0),      # blue
            (0, 0, 255),      # red
            (0, 255, 255),    # yellow
            (255, 0, 255),    # magenta
            (255, 255, 0),    # cyan
            (0, 165, 255),    # orange
            (128, 0, 255),    # purple
            (255, 128, 0),    # light blue
            (128, 255, 0),    # lime
            (180, 105, 255),  # pink-ish
            (42, 42, 165),    # brown-ish
        ]
        return colors[index % len(colors)]

    def _normalize_keypoints_array(self, kps, image_w, image_h):
        """
        Converts keypoints into a list of visible pixel points.
        Supports:
        - [x1, y1, x2, y2, ...]
        - [x1, y1, score1, x2, y2, score2, ...]
        - [[x, y], [x, y], ...]
        - [[x, y, score], ...]
        """
        kps = np.asarray(kps)

        if kps.ndim == 1:
            if len(kps) % 3 == 0:
                kps = kps.reshape(-1, 3)[:, :2]
            else:
                kps = kps.reshape(-1, 2)
        elif kps.ndim == 2 and kps.shape[1] >= 2:
            kps = kps[:, :2]
        else:
            return []

        points = []

        for kp in kps:
            x, y = float(kp[0]), float(kp[1])

            # If normalized, scale to original image size.
            if x <= 1.5 and y <= 1.5:
                x *= image_w
                y *= image_h

            x, y = int(round(x)), int(round(y))

            if 0 <= x < image_w and 0 <= y < image_h:
                points.append((x, y))

        return points

    def draw_skeleton(self, img, kps, color, skeleton=None):
        h, w = img.shape[:2]
        points = self._normalize_keypoints_array(kps, w, h)

        # Draw keypoints first.
        for x, y in points:
            cv2.circle(img, (x, y), 3, color, -1)
            cv2.circle(img, (x, y), 4, (0, 0, 0), 1)

        # Draw skeleton using topology from dataset, or fall back to linear chain.
        if skeleton:
            for a, b in skeleton:
                if a < len(points) and b < len(points):
                    cv2.line(img, points[a], points[b], color, 1)
        else:
            for i in range(len(points) - 1):
                cv2.line(img, points[i], points[i + 1], color, 1)

        return points

    def draw_label(self, img, text, anchor_point, color):
        x, y = anchor_point
        h, w = img.shape[:2]

        x = max(0, min(int(x), w - 1))
        y = max(20, min(int(y), h - 1))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1

        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size

        box_x1 = x
        box_y1 = y - text_h - baseline - 8
        box_x2 = x + text_w + 10
        box_y2 = y + baseline

        # Keep label inside image bounds.
        if box_x2 >= w:
            shift = box_x2 - w + 2
            box_x1 -= shift
            box_x2 -= shift

        if box_y1 < 0:
            box_y1 = y
            box_y2 = y + text_h + baseline + 8
            text_y = box_y1 + text_h + 4
        else:
            text_y = box_y2 - baseline - 4

        box_x1 = max(0, box_x1)
        box_y1 = max(0, box_y1)
        box_x2 = min(w - 1, box_x2)
        box_y2 = min(h - 1, box_y2)

        cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), color, -1)

        # White text is easier to read across most object colours.
        cv2.putText(
            img,
            text,
            (box_x1 + 5, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _get_label_anchor_from_keypoints(self, points, image_w, image_h):
        """
        Places label near the top-left of the object's visible keypoint cluster.
        """
        if not points:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x = max(0, min(xs))
        y = max(20, min(ys) - 10)

        return x, y

    @torch.no_grad()
    def infer(self, image_bgr):
        """Run inference on a single BGR image and return normalized detections."""
        tensor = self.preprocess(image_bgr)

        outputs = self.model(tensor)

        h, w = image_bgr.shape[:2]
        target_sizes = torch.tensor(
            [[h, w]],
            dtype=torch.float32,
            device=self.device,
        )
        results = self.postprocessor(outputs, target_sizes)

        if isinstance(results, list):
            result = results[0]
        else:
            result = {
                "scores": results["scores"][0],
                "labels": results["labels"][0],
                "keypoints": results["keypoints"][0],
            }
            if "boxes" in results:
                result["boxes"] = results["boxes"][0]

        scores = result["scores"].detach().cpu().numpy()
        labels = result["labels"].detach().cpu().numpy()
        keypoints = result["keypoints"].detach().cpu().numpy()

        boxes = None
        if "boxes" in result:
            boxes = result["boxes"].detach().cpu().numpy()

        keep = scores > self.conf_thresh
        scores = scores[keep]
        labels = labels[keep]
        keypoints = keypoints[keep]
        if boxes is not None:
            boxes = boxes[keep]

        if boxes is not None and len(scores) > 0:
            keep_dup = class_agnostic_nms_indices(boxes, scores, self.duplicate_iou_thresh)
            scores = scores[keep_dup]
            labels = labels[keep_dup]
            keypoints = keypoints[keep_dup]
            boxes = boxes[keep_dup]

        detections = []
        for idx, (score, label, kps) in enumerate(zip(scores, labels, keypoints)):
            detection = {
                "score": float(score),
                "label": int(label),
                "label_name": self.class_mappings.get(int(label), f"class_{int(label)}"),
                "keypoints": kps,
            }
            if boxes is not None and len(boxes) > idx:
                detection["box"] = boxes[idx]
            detections.append(detection)

        return detections

    def infer_image(self, image_path, output_dir, report_path, progress=None, draw_images=True, save_images=True):
        img = cv2.imread(str(image_path))

        if img is None:
            emit_prediction_log(
                report_path=report_path,
                image_name=image_path.name,
                detections=[],
                failure_message=f"Could not read image: {image_path}",
                progress=progress,
            )
            return

        tensor = self.preprocess(img)

        with torch.no_grad():
            outputs = self.model(tensor)

            h, w = img.shape[:2]

            # PostProcess expects a tensor, not torch.Size.
            # Order is [height, width].
            target_sizes = torch.tensor(
                [[h, w]],
                dtype=torch.float32,
                device=self.device,
            )

            results = self.postprocessor(outputs, target_sizes)

        # Handle either list-style or dict-style postprocessor output.
        if isinstance(results, list):
            result = results[0]
        else:
            result = {
                "scores": results["scores"][0],
                "labels": results["labels"][0],
                "keypoints": results["keypoints"][0],
            }
            if "boxes" in results:
                result["boxes"] = results["boxes"][0]

        scores = result["scores"].detach().cpu().numpy()
        labels = result["labels"].detach().cpu().numpy()
        keypoints = result["keypoints"].detach().cpu().numpy()

        boxes = None
        if "boxes" in result:
            boxes = result["boxes"].detach().cpu().numpy()

        keep = scores > self.conf_thresh
        scores = scores[keep]
        labels = labels[keep]
        keypoints = keypoints[keep]

        if boxes is not None:
            boxes = boxes[keep]

        if boxes is not None and len(scores) > 0:
            keep_dup = class_agnostic_nms_indices(boxes, scores, self.duplicate_iou_thresh)
            scores = scores[keep_dup]
            labels = labels[keep_dup]
            keypoints = keypoints[keep_dup]
            boxes = boxes[keep_dup]

        vis = None
        if draw_images:
            vis = img.copy()

            for i, (score, label, kps) in enumerate(zip(scores, labels, keypoints)):
                color = self.get_object_color(i)
                class_name = self.class_mappings.get(int(label), f"class_{int(label)}")
                text = f"{class_name} {score:.3f}"

                # Draw keypoints/skeleton first in the object's colour.
                skeleton = self.skeleton_connections.get(int(label), [])
                points = self.draw_skeleton(vis, kps, color, skeleton)

                # Draw bounding box and label only if the model provides bbox information
                if boxes is not None and len(boxes) > i:
                    x1, y1, x2, y2 = boxes[i][:4]

                    # If normalized, scale to original image size.
                    if x2 <= 1.5 and y2 <= 1.5:
                        x1 *= w
                        x2 *= w
                        y1 *= h
                        y2 *= h

                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w - 1))
                    y2 = max(0, min(y2, h - 1))

                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
                    label_anchor = (x1, y1)
                    self.draw_label(vis, text, label_anchor, color)
        output_path = None
        if save_images:
            output_path = output_dir / f"pred_{image_path.name}"
            cv2.imwrite(str(output_path), vis if vis is not None else img)
        emit_prediction_log(
            report_path=report_path,
            image_name=image_path.name,
            detections=[
                (self.class_mappings.get(int(label), f"class_{int(label)}"), float(score))
                for score, label in zip(scores, labels)
            ],
            output_path=output_path,
            progress=progress,
        )

    def infer_path(self, input_path, output_dir, report_file=None, draw_images=True, save_images=True, write_report=True):
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        if save_images or write_report:
            output_dir.mkdir(parents=True, exist_ok=True)
        report_path = ensure_report_path(output_dir, report_file) if write_report else None
        progress = InferenceProgress(total=1 if input_path.is_file() else None)

        if input_path.is_file():
            self.infer_image(
                input_path,
                output_dir,
                report_path,
                progress=progress,
                draw_images=draw_images,
                save_images=save_images,
            )
            print(progress.summary_line())
            return

        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            image_paths.extend(input_path.rglob(ext))

        image_paths = sorted(image_paths)

        if not image_paths:
            print(f"No images found in: {input_path}")
            return

        print(f"Found {len(image_paths)} images")
        progress.total = len(image_paths)

        for img_path in image_paths:
            self.infer_image(
                img_path,
                output_dir,
                report_path,
                progress=progress,
                draw_images=draw_images,
                save_images=save_images,
            )
        print(progress.summary_line())


def resolve_and_extract_archives(input_path):
    """
    Resolves the input path before inference:
    - If input_path is a .7z or .zip archive, extract it to a same-named folder.
    - Then recursively scan the folder for any nested archives, extract and delete
      them until none remain.
    Returns the resolved folder Path.
    """
    import subprocess

    ARCHIVE_EXTS = {'.7z', '.zip'}

    def _extract_one(archive_path):
        archive_path = Path(archive_path)
        out_dir = archive_path.parent / archive_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ['7z', 'x', str(archive_path), f'-o{out_dir}', '-y'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"7z extraction failed for {archive_path.name}:\n"
                f"{result.stderr.decode(errors='replace')}"
            )
        archive_path.unlink()
        print(f"Extracted and removed archive: {archive_path.name} -> {out_dir}")
        return out_dir

    input_path = Path(input_path)
    # If the input itself is an archive, extract it to a folder first.
    if input_path.is_file() and input_path.suffix.lower() in ARCHIVE_EXTS:
        input_path = _extract_one(input_path)

    # Recursively extract any nested archives until none remain.
    if input_path.is_dir():
        while True:
            archives = sorted(
                p for p in input_path.rglob('*')
                if p.is_file() and p.suffix.lower() in ARCHIVE_EXTS
            )
            if not archives:
                break
            for arch in archives:
                _extract_one(arch)

    return input_path


def auto_detect_config(checkpoint_path=None, metadata=None):
    """
    Auto-detect the config module from checkpoint path.
    Example: output/detrpose_hgnetv2_n_custom/checkpoint.pth
             -> configs.detrpose.detrpose_hgnetv2_n_custom
    """
    metadata = metadata or {}
    if metadata.get("config_module"):
        print(f"Using checkpoint config metadata: {metadata['config_module']}")
        return metadata["config_module"]

    if checkpoint_path is None:
        return "configs.detrpose.detrpose_hgnetv2_s_custom"

    checkpoint_path = Path(checkpoint_path)
    
    # Try to extract config name from parent directory
    parent_dir = checkpoint_path.parent.name
    
    # Check if parent directory matches a config pattern
    if parent_dir.startswith("detrpose_"):
        config_module = f"configs.detrpose.{parent_dir}"
        print(f"Auto-detected config: {config_module}")
        return config_module
    
    # Fallback to default
    print("Warning: Could not auto-detect config from checkpoint path.")
    print("Using default: configs.detrpose.detrpose_hgnetv2_s_custom")
    return "configs.detrpose.detrpose_hgnetv2_s_custom"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="predictions")
    parser.add_argument("--report-file", type=str, default=None,
        help="Optional prediction report filename or path")
    parser.add_argument("--duplicate-iou-thresh", type=float, default=0.5,
        help="Suppress lower-confidence overlapping detections above this IoU")
    parser.add_argument("--no-save-images", action="store_true",
        help="Skip drawing and writing output images")
    parser.add_argument("--no-draw", action="store_true",
        help="Skip annotation rendering. If images are still saved, raw inputs are written")
    parser.add_argument("--no-report", action="store_true",
        help="Skip writing the prediction report file")
    parser.add_argument("--draw-only", action="store_true",
        help="Render annotations in memory only. Skips image writes and report writing")
    parser.add_argument("--benchmark", action="store_true",
        help="Fast path: skip image rendering, saving, and report writing")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Python module path to the training config (auto-detected from checkpoint path if not specified)",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Use ckpt['model'] instead of EMA weights if EMA exists",
    )

    args = parser.parse_args()

    infer = DETRPoseInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        conf_thresh=args.conf,
        config_module=args.config,
        use_ema=not args.no_ema,
        image_size=args.image_size,
        duplicate_iou_thresh=args.duplicate_iou_thresh,
    )

    resolved_input = resolve_and_extract_archives(args.input)
    draw_images = not (args.no_draw or args.benchmark)
    save_images = not (args.no_save_images or args.benchmark or args.draw_only)
    write_report = not (args.no_report or args.benchmark or args.draw_only)
    infer.infer_path(
        resolved_input,
        args.output,
        args.report_file,
        draw_images=draw_images,
        save_images=save_images,
        write_report=write_report,
    )

    if save_images and draw_images:
        print(f"\nDone. Results saved to: {args.output}")
    elif save_images:
        print(f"\nDone. Raw images saved to: {args.output}")
    elif args.draw_only:
        print("\nDone. Draw-only benchmark completed without saving outputs.")
    elif write_report:
        print(f"\nDone. Report saved under: {args.output}")
    else:
        print("\nDone. Benchmark mode completed without saving outputs.")
