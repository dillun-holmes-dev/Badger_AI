from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from .checkpoint_metadata import (
        config_module_exists,
        load_checkpoint,
        normalize_config_module,
        resolve_checkpoint_architecture_metadata,
    )
except ImportError:
    from checkpoint_metadata import (
        config_module_exists,
        load_checkpoint,
        normalize_config_module,
        resolve_checkpoint_architecture_metadata,
    )


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_MODEL_SUFFIXES = {".pth", ".pt", ".ckpt", ".onnx", ".engine", ".trt"}
RUNTIME_SUFFIXES = {
    "pytorch": {".pth", ".pt", ".ckpt"},
    "onnx": {".onnx"},
    "tensorrt": {".engine", ".trt"},
}
DEFAULT_CONFIGS = {
    "detrpose": "configs.detrpose.detrpose_hgnetv2_s_custom",
    "rtmopose": "configs.rtmopose.rtmopose_hgnetv2_s_custom",
    "rtmdetpose": "configs.rtmdetpose.rtmdetpose_hgnetv2_s_custom",
}
COLOR_PALETTE = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 165, 255),
    (128, 0, 255),
    (255, 128, 0),
    (128, 255, 0),
    (180, 105, 255),
    (42, 42, 165),
]


@dataclass
class RunnerBundle:
    runner: Any
    runtime: str
    model_family: str
    model_path: Path
    config_module: Optional[str]


@dataclass
class ProgressState:
    total_images: int
    started_at: float
    processed_images: int = 0
    successful_images: int = 0
    error_images: int = 0
    total_inference_time_ms: float = 0.0

    def advance(self, result: Dict[str, Any]) -> Dict[str, Any]:
        self.processed_images += 1
        if result.get("inference_type") == "error":
            self.error_images += 1
        else:
            self.successful_images += 1
            self.total_inference_time_ms += float(result.get("inference_time_ms") or 0.0)
        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        elapsed_ms = (time.perf_counter() - self.started_at) * 1000.0
        average_inference_ms = (
            self.total_inference_time_ms / self.successful_images if self.successful_images else 0.0
        )
        return {
            "total_images": self.total_images,
            "processed_images": self.processed_images,
            "successful_images": self.successful_images,
            "error_images": self.error_images,
            "remaining_images": max(self.total_images - self.processed_images, 0),
            "average_inference_time_ms": average_inference_ms,
            "total_inference_time_ms": self.total_inference_time_ms,
            "total_elapsed_ms": elapsed_ms,
        }


class ProgressEmitter:
    def __init__(
        self,
        *,
        progress_file: Optional[Path],
        event_stream: str,
        result_ndjson: Optional[Path],
        pretty: bool,
    ) -> None:
        self.progress_file = progress_file
        self.event_stream = event_stream
        self.result_ndjson = result_ndjson
        self.pretty = pretty

    def emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.progress_file is not None:
            snapshot = dict(payload)
            snapshot["event"] = event_type
            self._write_snapshot(self.progress_file, snapshot)

        if self.event_stream != "none":
            line = json.dumps(
                {"event": event_type, **payload},
                ensure_ascii=True,
                separators=(",", ":"),
            )
            target = sys.stderr if self.event_stream == "stderr" else sys.stdout
            print(f"VISIONHUB_PROGRESS {line}", file=target, flush=True)

    def emit_result(self, result: Dict[str, Any]) -> None:
        if self.result_ndjson is None:
            return

        self.result_ndjson.parent.mkdir(parents=True, exist_ok=True)
        with self.result_ndjson.open("a", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=True)
            handle.write("\n")

    def _write_snapshot(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2 if self.pretty else None)
            handle.write("\n")
        temp_path.replace(path)


def _stderr(message: str) -> None:
    print(message, file=sys.stderr)


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _looks_like_glob(value: str) -> bool:
    return any(token in value for token in ("*", "?", "["))


def _normalize_runtime(value: str) -> str:
    runtime = str(value or "auto").strip().lower()
    aliases = {
        "torch": "pytorch",
        "pt": "pytorch",
        "onnxruntime": "onnx",
        "trt": "tensorrt",
        "engine": "tensorrt",
    }
    runtime = aliases.get(runtime, runtime)
    if runtime not in {"auto", "pytorch", "onnx", "tensorrt"}:
        raise ValueError(f"Unsupported runtime '{value}'. Use auto, pytorch, onnx, or tensorrt.")
    return runtime


def _normalize_family(value: Optional[str]) -> str:
    family = str(value or "auto").strip().lower()
    aliases = {
        "detr": "detrpose",
        "rtmo": "rtmopose",
        "rtmdetpose": "rtmdetpose",
        "rtmdet-pose": "rtmdetpose",
        "rtmdet_pose": "rtmdetpose",
    }
    family = aliases.get(family, family)
    if family not in {"auto", "detrpose", "rtmopose", "rtmdetpose"}:
        raise ValueError(
            f"Unsupported model family '{value}'. Use auto, detrpose, rtmopose, or rtmdetpose."
        )
    return family


def _config_to_module(config_value: Optional[str]) -> Optional[str]:
    return normalize_config_module(config_value)


def _infer_family_from_strings(values: Sequence[str]) -> str:
    joined = " ".join(values).lower()
    if "rtmopose" in joined or "rtmo" in joined:
        return "rtmopose"
    if "rtmdetpose" in joined or "rtmdet-pose" in joined or "rtmdet_pose" in joined:
        return "rtmdetpose"
    if "rtmdet" in joined:
        return "rtmdetpose"
    if "detrpose" in joined or "detr" in joined:
        return "detrpose"
    return "auto"


def _candidate_checkpoint_names(model_path: Path) -> List[Path]:
    candidates: List[Path] = []
    candidates.extend([model_path.parent, *model_path.parents])
    return candidates


def _module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _auto_detect_config_module(model_path: Path, family_hint: str) -> Optional[str]:
    families = [family_hint] if family_hint != "auto" else ["detrpose", "rtmopose", "rtmdetpose"]
    config_dirs = {
        "detrpose": "detrpose",
        "rtmopose": "rtmopose",
        "rtmdetpose": "rtmdetpose",
    }

    def candidate_names(family: str, name: str) -> List[str]:
        names: List[str] = []
        if family in {"detrpose", "rtmopose"} and name.startswith(family):
            names.append(name)
        elif family == "rtmdetpose":
            if name.startswith("rtmdetpose_hgnetv2_"):
                names.append(name)
            if name.startswith("rtmdet_hgnetv2_"):
                names.append(name.replace("rtmdet_hgnetv2_", "rtmdetpose_hgnetv2_", 1))
        return names

    names_to_try: List[str] = []
    for candidate_path in _candidate_checkpoint_names(model_path):
        name = candidate_path.name
        if name and name not in names_to_try:
            names_to_try.append(name)

    for family in families:
        config_dir = config_dirs[family]
        for name in names_to_try:
            for candidate_name in candidate_names(family, name):
                module_name = f"configs.{config_dir}.{candidate_name}"
                if _module_exists(module_name):
                    return module_name

    if family_hint != "auto":
        default_module = DEFAULT_CONFIGS.get(family_hint)
        if default_module and _module_exists(default_module):
            return default_module
    return None


def _resolve_model_path(model_value: str, runtime_hint: str) -> Path:
    input_path = Path(model_value).expanduser()
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_MODEL_SUFFIXES:
            raise FileNotFoundError(f"Unsupported model file type: {input_path}")
        return input_path.resolve()

    if not input_path.is_dir():
        raise FileNotFoundError(f"Model path not found: {input_path}")

    runtime_order = (
        [runtime_hint]
        if runtime_hint != "auto"
        else ["pytorch", "onnx", "tensorrt"]
    )
    preferred_names = {
        "pytorch": ["checkpoint_best_regular.pth", "checkpoint_best.pth", "latest.pth"],
        "onnx": [],
        "tensorrt": [],
    }

    for runtime in runtime_order:
        suffixes = RUNTIME_SUFFIXES[runtime]
        for preferred_name in preferred_names.get(runtime, []):
            matches = sorted(input_path.rglob(preferred_name))
            if matches:
                return matches[0].resolve()

        matches = sorted(
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        )
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            preview = ", ".join(str(path) for path in matches[:5])
            raise FileNotFoundError(
                f"Multiple {runtime} model files found under {input_path}. "
                f"Please pass a file path instead. Candidates: {preview}"
            )

    raise FileNotFoundError(
        f"No supported model file found under {input_path}. "
        "Expected .pth, .pt, .ckpt, .onnx, .engine, or .trt."
    )


def _detect_runtime_from_path(model_path: Path, runtime_value: str) -> str:
    runtime = _normalize_runtime(runtime_value)
    if runtime != "auto":
        return runtime

    suffix = model_path.suffix.lower()
    for detected_runtime, suffixes in RUNTIME_SUFFIXES.items():
        if suffix in suffixes:
            return detected_runtime
    raise ValueError(f"Could not infer runtime from model path: {model_path}")


def _load_runner(args: argparse.Namespace) -> RunnerBundle:
    runtime_hint = _normalize_runtime(args.runtime)
    model_path = _resolve_model_path(args.model, runtime_hint)
    runtime = _detect_runtime_from_path(model_path, runtime_hint)
    checkpoint_metadata: Dict[str, Any] = {}

    if runtime == "pytorch":
        checkpoint_metadata = resolve_checkpoint_architecture_metadata(
            load_checkpoint(str(model_path)),
            use_ema=not args.no_ema,
        )

    family = _normalize_family(args.family)
    if family == "auto":
        family = checkpoint_metadata.get("family") or _infer_family_from_strings(
            [str(model_path), str(args.config or ""), str(model_path.parent.name)]
        )

    config_module = _config_to_module(args.config)

    if runtime == "pytorch":
        if family == "auto":
            raise ValueError(
                "Could not infer the PyTorch pose family from the checkpoint path. "
                "Pass --family or --config."
            )
        if config_module is None:
            checkpoint_config = checkpoint_metadata.get("config_module")
            if checkpoint_config and config_module_exists(checkpoint_config):
                config_module = checkpoint_config
        if config_module is None:
            config_module = _auto_detect_config_module(model_path, family)
        if config_module is None:
            raise ValueError(
                "Could not auto-detect the training config for this checkpoint. "
                "Pass --config explicitly."
            )

        runner_spec = {
            "detrpose": ("tools.inference.inference", "DETRPoseInference"),
            "rtmopose": ("tools.inference.rtmo_inf", "RTMOInference"),
            "rtmdetpose": ("tools.inference.rtmdetpose_inf", "RTMDetInference"),
        }[family]
        module_name, class_name = runner_spec
        module = importlib.import_module(module_name)
        runner_cls = getattr(module, class_name)
        runner = runner_cls(
            checkpoint_path=str(model_path),
            device=args.device,
            conf_thresh=args.conf,
            config_module=config_module,
            use_ema=not args.no_ema,
            image_size=args.image_size,
            duplicate_iou_thresh=args.duplicate_iou_thresh,
        )
        return RunnerBundle(
            runner=runner,
            runtime=runtime,
            model_family=family,
            model_path=model_path,
            config_module=config_module,
        )

    if runtime == "onnx":
        module = importlib.import_module("tools.inference.onnx_inf")
        runner = module.ONNXInference(
            onnx_path=str(model_path),
            conf_thresh=args.conf,
            image_size=args.image_size,
            duplicate_iou_thresh=args.duplicate_iou_thresh,
        )
        if family == "auto":
            family = _infer_family_from_strings([str(model_path)])
        return RunnerBundle(
            runner=runner,
            runtime=runtime,
            model_family=family,
            model_path=model_path,
            config_module=None,
        )

    module = importlib.import_module("tools.inference.trt_inf")
    runner = module.TRTInferenceRunner(
        engine_path=str(model_path),
        conf_thresh=args.conf,
        image_size=args.image_size,
        device=args.device,
        duplicate_iou_thresh=args.duplicate_iou_thresh,
    )
    if family == "auto":
        family = _infer_family_from_strings([str(model_path)])
    return RunnerBundle(
        runner=runner,
        runtime=runtime,
        model_family=family,
        model_path=model_path,
        config_module=None,
    )


def _iter_source_images(source_value: str) -> Iterable[Path]:
    source_path = Path(source_value).expanduser()
    if _is_image_file(source_path):
        yield source_path.resolve()
        return

    if source_path.is_dir():
        for path in sorted(source_path.rglob("*")):
            if _is_image_file(path):
                yield path.resolve()
        return

    if _looks_like_glob(source_value):
        import glob

        for match in sorted(glob.glob(source_value, recursive=True)):
            path = Path(match).expanduser()
            if _is_image_file(path):
                yield path.resolve()
        return

    raise FileNotFoundError(
        f"Source path not found or unsupported: {source_value}. "
        "Pass an image file, directory, or glob pattern."
    )


def _infer_source_root(source_value: str, image_paths: Sequence[Path]) -> Path:
    source_path = Path(source_value).expanduser()
    if source_path.is_dir():
        return source_path.resolve()
    if source_path.is_file():
        return source_path.resolve().parent
    if not image_paths:
        return Path.cwd().resolve()

    common_path = Path(os.path.commonpath([str(path) for path in image_paths]))
    return common_path if common_path.is_dir() else common_path.parent


def _remap_reported_path(
    actual_image_path: Path,
    source_root: Path,
    reported_source_root: Optional[Path],
) -> str:
    if reported_source_root is None:
        return str(actual_image_path)

    try:
        relative = actual_image_path.relative_to(source_root)
    except ValueError:
        return str(actual_image_path)
    return str(reported_source_root / relative)


def _normalize_box(box: Any, width: int, height: int) -> Optional[List[int]]:
    if box is None:
        return None

    values = np.asarray(box, dtype=np.float32).reshape(-1)
    if values.size < 4:
        return None

    x1, y1, x2, y2 = (float(v) for v in values[:4])
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height

    x1 = int(round(max(0.0, min(x1, width - 1))))
    y1 = int(round(max(0.0, min(y1, height - 1))))
    x2 = int(round(max(0.0, min(x2, width - 1))))
    y2 = int(round(max(0.0, min(y2, height - 1))))
    return [x1, y1, x2, y2]


def _reshape_keypoints(raw_keypoints: Any) -> Optional[np.ndarray]:
    values = np.asarray(raw_keypoints, dtype=np.float32)
    if values.size == 0:
        return None

    if values.ndim == 1:
        if values.size % 3 == 0:
            return values.reshape(-1, 3)
        if values.size % 2 == 0:
            return values.reshape(-1, 2)
        return None

    if values.ndim == 2:
        if values.shape[1] >= 2:
            return values[:, :3] if values.shape[1] >= 3 else values[:, :2]
        flattened = values.reshape(-1)
        return _reshape_keypoints(flattened)

    reshaped = values.reshape(values.shape[0], -1)
    return _reshape_keypoints(reshaped[0]) if len(reshaped) == 1 else None


def _build_keypoint_payload(
    raw_keypoints: Any,
    width: int,
    height: int,
    default_confidence: float,
) -> List[Dict[str, Any]]:
    reshaped = _reshape_keypoints(raw_keypoints)
    if reshaped is None:
        return []

    keypoints: List[Dict[str, Any]] = []
    for index, row in enumerate(reshaped):
        x = float(row[0])
        y = float(row[1])
        confidence = float(row[2]) if len(row) >= 3 else float(default_confidence)

        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(confidence):
            continue

        if max(abs(x), abs(y)) <= 1.5:
            x *= width
            y *= height

        x_value = int(round(max(0.0, min(x, width - 1))))
        y_value = int(round(max(0.0, min(y, height - 1))))
        if x_value == 0 and y_value == 0 and confidence <= 0.0:
            continue

        keypoints.append(
            {
                "index": index,
                "x": x_value,
                "y": y_value,
                "confidence": confidence,
            }
        )

    return keypoints


def _build_prediction(det: Dict[str, Any], image_shape: Tuple[int, int, int]) -> Dict[str, Any]:
    height, width = image_shape[:2]
    confidence = float(det.get("score", 0.0) or 0.0)
    class_id = int(det.get("label", -1))
    class_name = str(det.get("label_name") or f"class_{class_id}")

    prediction: Dict[str, Any] = {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "type": "pose",
    }

    bbox = _normalize_box(det.get("box"), width, height)
    if bbox is not None:
        prediction["bbox"] = bbox

    keypoints = _build_keypoint_payload(det.get("keypoints"), width, height, confidence)
    if keypoints:
        prediction["keypoints"] = keypoints
        prediction["keypoint_count"] = len(keypoints)
        prediction["avg_keypoint_confidence"] = (
            sum(point["confidence"] for point in keypoints) / len(keypoints)
        )

    return prediction


def _draw_label(image: np.ndarray, text: str, anchor: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    x, y = anchor
    height, width = image.shape[:2]
    x = max(0, min(int(x), width - 1))
    y = max(20, min(int(y), height - 1))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    box_x1 = x
    box_y1 = y - text_height - baseline - 8
    box_x2 = x + text_width + 10
    box_y2 = y + baseline

    if box_x2 >= width:
        shift = box_x2 - width + 2
        box_x1 -= shift
        box_x2 -= shift

    if box_y1 < 0:
        box_y1 = y
        box_y2 = y + text_height + baseline + 8
        text_y = box_y1 + text_height + 4
    else:
        text_y = box_y2 - baseline - 4

    box_x1 = max(0, box_x1)
    box_y1 = max(0, box_y1)
    box_x2 = min(width - 1, box_x2)
    box_y2 = min(height - 1, box_y2)

    cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), color, -1)
    cv2.putText(
        image,
        text,
        (box_x1 + 5, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _resolve_label_anchor(prediction: Dict[str, Any]) -> Tuple[int, int]:
    bbox = prediction.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return int(bbox[0]), int(bbox[1])

    keypoints = prediction.get("keypoints") or []
    if keypoints:
        xs = [int(point["x"]) for point in keypoints]
        ys = [int(point["y"]) for point in keypoints]
        return max(0, min(xs)), max(20, min(ys) - 10)

    return 0, 20


def _skeleton_for_label(skeleton_connections: Dict[Any, Any], class_id: int) -> List[Sequence[int]]:
    skeleton = skeleton_connections.get(class_id)
    if skeleton is None:
        skeleton = skeleton_connections.get(str(class_id), [])
    return skeleton or []


def _render_pose_predictions(
    image_bgr: np.ndarray,
    predictions: Sequence[Dict[str, Any]],
    skeleton_connections: Dict[Any, Any],
) -> np.ndarray:
    rendered = image_bgr.copy()

    for index, prediction in enumerate(predictions):
        color = COLOR_PALETTE[index % len(COLOR_PALETTE)]
        bbox = prediction.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 1)

        keypoints = prediction.get("keypoints") or []
        point_lookup = {
            int(point["index"]): (
                int(point["x"]),
                int(point["y"]),
                float(point.get("confidence", prediction.get("confidence", 0.0)) or 0.0),
            )
            for point in keypoints
        }

        for point in keypoints:
            if float(point.get("confidence", 0.0) or 0.0) <= 0.0:
                continue
            cv2.circle(
                rendered,
                (int(point["x"]), int(point["y"])),
                3,
                color,
                -1,
            )
            cv2.circle(
                rendered,
                (int(point["x"]), int(point["y"])),
                4,
                (0, 0, 0),
                1,
            )

        skeleton = _skeleton_for_label(skeleton_connections, int(prediction["class_id"]))
        for connection in skeleton:
            if len(connection) != 2:
                continue
            point_a = point_lookup.get(int(connection[0]))
            point_b = point_lookup.get(int(connection[1]))
            if point_a is None or point_b is None:
                continue
            if point_a[2] <= 0.0 or point_b[2] <= 0.0:
                continue
            cv2.line(rendered, point_a[:2], point_b[:2], color, 1)

        label = f"{prediction['class_name']} {float(prediction['confidence']):.3f}"
        _draw_label(rendered, label, _resolve_label_anchor(prediction), color)

    return rendered


def _resolve_output_path(
    output_dir: Path,
    actual_image_path: Path,
    source_root: Path,
    annotated: bool,
) -> Path:
    try:
        relative = actual_image_path.relative_to(source_root)
    except ValueError:
        relative = Path(actual_image_path.name)

    filename = f"annotated_{relative.name}" if annotated else relative.name
    return output_dir / relative.parent / filename


def _error_result(image_path: str, error_message: str) -> Dict[str, Any]:
    return {
        "image_path": image_path,
        "annotated_path": None,
        "predictions": [],
        "inference_type": "error",
        "confidence": 0.0,
        "inference_time_ms": 0.0,
        "error": error_message,
    }


def _write_json(path: Path, payload: Dict[str, Any], pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2 if pretty else None)
        handle.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified pose inference endpoint for VisionHub checkpoints, ONNX exports, "
            "and TensorRT engines."
        )
    )
    parser.add_argument("--model", required=True, help="Path to a pose model file or directory.")
    parser.add_argument(
        "--source",
        "--input",
        dest="source",
        required=True,
        help="Image file, directory, or glob pattern.",
    )
    parser.add_argument(
        "--runtime",
        default="auto",
        help="auto, pytorch, onnx, or tensorrt.",
    )
    parser.add_argument(
        "--family",
        default="auto",
        help="auto, detrpose, rtmopose, or rtmdetpose. Needed only for ambiguous checkpoints.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Training config module or path. Used for PyTorch checkpoints.",
    )
    parser.add_argument("--device", default="cuda", help="Runtime device hint.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--image-size", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--duplicate-iou-thresh",
        type=float,
        default=0.5,
        help="Suppress lower-confidence overlapping detections above this IoU.",
    )
    parser.add_argument(
        "--output-dir",
        default="predictions",
        help="Directory for annotated outputs when image saving is enabled.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional file path to write the JSON payload.",
    )
    parser.add_argument(
        "--progress-file",
        default=None,
        help="Optional JSON file that is updated incrementally as each image finishes.",
    )
    parser.add_argument(
        "--progress-stream",
        choices=("none", "stderr", "stdout"),
        default="stderr",
        help="Emit structured progress events as JSON lines to stderr or stdout.",
    )
    parser.add_argument(
        "--result-ndjson",
        default=None,
        help="Optional NDJSON file to append one completed per-image result per line.",
    )
    parser.add_argument(
        "--reported-source-root",
        default=None,
        help="Optional root used to remap image_path values in the JSON payload.",
    )
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Skip writing image outputs.",
    )
    parser.add_argument(
        "--no-draw",
        action="store_true",
        help="If saving images, store the raw source image instead of annotations.",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Use ckpt['model'] instead of EMA weights when available.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON to stdout and --json-output.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    save_images = not args.no_save_images
    draw_images = not args.no_draw

    try:
        with contextlib.redirect_stdout(sys.stderr):
            bundle = _load_runner(args)

        image_paths = list(_iter_source_images(args.source))
        if not image_paths:
            raise FileNotFoundError(f"No images found for source: {args.source}")

        source_root = _infer_source_root(args.source, image_paths)
        reported_source_root = (
            Path(args.reported_source_root).expanduser()
            if args.reported_source_root
            else None
        )
        output_dir = Path(args.output_dir).expanduser().resolve()
        if save_images:
            output_dir.mkdir(parents=True, exist_ok=True)

        results: List[Dict[str, Any]] = []
        total_inference_ms = 0.0
        total_start = time.perf_counter()
        progress = ProgressState(total_images=len(image_paths), started_at=total_start)
        progress_emitter = ProgressEmitter(
            progress_file=Path(args.progress_file).expanduser() if args.progress_file else None,
            event_stream=args.progress_stream,
            result_ndjson=Path(args.result_ndjson).expanduser() if args.result_ndjson else None,
            pretty=args.pretty,
        )
        skeleton_connections = getattr(bundle.runner, "skeleton_connections", {}) or {}

        progress_emitter.emit_event(
            "start",
            {
                "success": True,
                "model_input": args.model,
                "model_path": str(bundle.model_path),
                "runtime": bundle.runtime,
                "model_family": bundle.model_family,
                "source_input": args.source,
                "source_root": str(source_root),
                "reported_source_root": str(reported_source_root) if reported_source_root else None,
                "output_dir": str(output_dir) if save_images else None,
                "summary": progress.snapshot(),
            },
        )

        for index, image_path in enumerate(image_paths, start=1):
            reported_path = _remap_reported_path(image_path, source_root, reported_source_root)
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise RuntimeError("Failed to decode image")

                inference_start = time.perf_counter()
                with contextlib.redirect_stdout(sys.stderr):
                    detections = bundle.runner.infer(image)
                inference_time_ms = (time.perf_counter() - inference_start) * 1000.0
                total_inference_ms += inference_time_ms

                predictions = [_build_prediction(det, image.shape) for det in detections]
                overall_confidence = (
                    sum(float(pred["confidence"]) for pred in predictions) / len(predictions)
                    if predictions
                    else 0.0
                )

                annotated_path: Optional[str] = None
                if save_images:
                    output_path = _resolve_output_path(
                        output_dir,
                        image_path,
                        source_root,
                        annotated=draw_images,
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    frame_to_write = (
                        _render_pose_predictions(image, predictions, skeleton_connections)
                        if draw_images
                        else image
                    )
                    if not cv2.imwrite(str(output_path), frame_to_write):
                        raise RuntimeError(f"Failed to write output image: {output_path}")
                    annotated_path = str(output_path)

                result = {
                    "image_path": reported_path,
                    "annotated_path": annotated_path,
                    "predictions": predictions,
                    "inference_type": "pose",
                    "confidence": overall_confidence,
                    "inference_time_ms": inference_time_ms,
                }
            except Exception as image_exc:
                result = _error_result(reported_path, str(image_exc))

            results.append(result)
            progress_summary = progress.advance(result)
            progress_emitter.emit_result(result)
            progress_emitter.emit_event(
                "image_complete",
                {
                    "image_index": index,
                    "image_path": reported_path,
                    "result": result,
                    "summary": progress_summary,
                },
            )

        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0
        error_count = sum(1 for item in results if item.get("inference_type") == "error")
        success_count = len(results) - error_count
        average_inference_ms = total_inference_ms / success_count if success_count else 0.0

        payload = {
            "success": True,
            "model_input": args.model,
            "model_path": str(bundle.model_path),
            "runtime": bundle.runtime,
            "model_family": bundle.model_family,
            "config_module": bundle.config_module,
            "source_input": args.source,
            "source_root": str(source_root),
            "reported_source_root": str(reported_source_root) if reported_source_root else None,
            "output_dir": str(output_dir) if save_images else None,
            "results": results,
            "summary": {
                "total_images": len(results),
                "successful_images": success_count,
                "error_images": error_count,
                "average_inference_time_ms": average_inference_ms,
                "total_inference_time_ms": total_inference_ms,
                "total_elapsed_ms": total_elapsed_ms,
            },
        }

        progress_emitter.emit_event("complete", payload)

        if args.json_output:
            _write_json(Path(args.json_output).expanduser(), payload, pretty=args.pretty)

        json.dump(payload, sys.stdout, indent=2 if args.pretty else None)
        sys.stdout.write("\n")
        return 0
    except Exception as exc:
        _stderr(f"visionhub-infer-pose failed: {exc}")
        error_payload = {
            "success": False,
            "error": str(exc),
            "results": [],
            "summary": {
                "total_images": 0,
                "successful_images": 0,
                "error_images": 0,
                "average_inference_time_ms": 0.0,
                "total_inference_time_ms": 0.0,
                "total_elapsed_ms": 0.0,
            },
        }
        if getattr(args, "progress_file", None) or getattr(args, "progress_stream", "none") != "none":
            ProgressEmitter(
                progress_file=Path(args.progress_file).expanduser() if getattr(args, "progress_file", None) else None,
                event_stream=getattr(args, "progress_stream", "none"),
                result_ndjson=Path(getattr(args, "result_ndjson", "")).expanduser()
                if getattr(args, "result_ndjson", None)
                else None,
                pretty=getattr(args, "pretty", False),
            ).emit_event("failed", error_payload)
        if getattr(args, "json_output", None):
            _write_json(Path(args.json_output).expanduser(), error_payload, pretty=getattr(args, "pretty", False))
        json.dump(error_payload, sys.stdout, indent=2 if getattr(args, "pretty", False) else None)
        sys.stdout.write("\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
