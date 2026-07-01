"""ONNX classifier inference CLI."""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from visionhub.models.classification import (
    build_native_classifier_eval_transform,
    load_classifier_metadata_json,
)

try:
    from .report_utils import InferenceProgress, ensure_report_path
except ImportError:
    from report_utils import InferenceProgress, ensure_report_path


IMAGE_EXTENSIONS = (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp")


def _collect_image_paths(source_path: Path) -> List[Path]:
    if source_path.is_file():
        return [source_path]
    if not source_path.is_dir():
        raise ValueError(f"Source path not found: {source_path}")
    image_paths: List[Path] = []
    for extension in IMAGE_EXTENSIONS:
        image_paths.extend(source_path.rglob(f"*{extension}"))
    return sorted(image_paths)


def _build_report_lines(image_name: str, predictions: List[Dict[str, float]], output_path: Optional[Path]) -> List[str]:
    lines = [f"Processing: {image_name}", f"  Top {len(predictions)} predictions"]
    for prediction in predictions:
        lines.append(f"    {prediction['class_name']}: {prediction['confidence']:.3f}")
    if output_path is not None:
        lines.append(f"  Saved: {output_path}")
    return lines


def _append_report(report_path: Optional[Path], lines: Iterable[str], progress: Optional[InferenceProgress]) -> None:
    if progress is not None:
        stats = progress.record()
        total_text = f"/{stats['total']}" if stats["total"] is not None else ""
        print(
            f"\rProgress: {stats['processed']}{total_text} | "
            f"{stats['avg_fps']:.2f} FPS avg | "
            f"{stats['last_fps']:.2f} FPS last | "
            f"{stats['avg_ms']:.1f} ms/img avg",
            end="",
            flush=True,
        )
    if report_path is not None:
        with report_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
            handle.write("\n\n")


def _render_prediction(cv2_module, image, predictions: List[Dict[str, float]]):
    rendered = image.copy()
    if not predictions:
        return rendered

    box_x1 = 12
    box_y1 = 12
    line_height = 22
    box_width = max(220, max(len(pred["class_name"]) for pred in predictions[:3]) * 10 + 90)
    box_height = 16 + line_height * min(3, len(predictions))
    cv2_module.rectangle(rendered, (box_x1, box_y1), (box_x1 + box_width, box_y1 + box_height), (30, 30, 30), -1)
    cv2_module.rectangle(rendered, (box_x1, box_y1), (box_x1 + box_width, box_y1 + box_height), (90, 220, 90), 2)
    for index, prediction in enumerate(predictions[:3]):
        label = f"{index + 1}. {prediction['class_name']} {prediction['confidence']:.3f}"
        cv2_module.putText(
            rendered,
            label,
            (box_x1 + 10, box_y1 + 22 + index * line_height),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2_module.LINE_AA,
        )
    return rendered


class ONNXClassifierInference:
    def __init__(self, ort_module, onnx_path: str, *, image_size: Optional[int] = None):
        self.onnx_path = onnx_path
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort_module.InferenceSession(onnx_path, providers=providers)
        metadata = load_classifier_metadata_json(onnx_path)
        normalization = metadata.get("normalization") if isinstance(metadata.get("normalization"), dict) else {}
        mean = normalization.get("mean") if isinstance(normalization.get("mean"), list) else [0.485, 0.456, 0.406]
        std = normalization.get("std") if isinstance(normalization.get("std"), list) else [0.229, 0.224, 0.225]
        class_mappings = metadata.get("class_mappings") if isinstance(metadata.get("class_mappings"), dict) else {}
        if class_mappings:
            self.class_names = [class_mappings[str(index)] for index in sorted(int(key) for key in class_mappings.keys())]
        else:
            self.class_names = [str(name) for name in metadata.get("class_names", [])] if isinstance(metadata.get("class_names"), list) else []
        self.image_size = int(image_size or metadata.get("image_size") or 224)
        self.transform = build_native_classifier_eval_transform(
            self.image_size,
            mean=[float(value) for value in mean],
            std=[float(value) for value in std],
        )

    def run_inference_batch(self, image_paths: List[Path], *, topk: int = 5) -> List[Dict[str, object]]:
        import numpy as np
        import torch
        from PIL import Image

        tensors = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                tensors.append(self.transform(image.convert("RGB")))
        batch = torch.stack(tensors).numpy().astype(np.float32)
        outputs = self.session.run(None, {"images": batch})
        if len(outputs) >= 2:
            probabilities = outputs[1]
        else:
            logits = outputs[0]
            shifted = logits - np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(shifted)
            probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        results: List[Dict[str, object]] = []
        effective_topk = max(1, min(int(topk), probabilities.shape[1]))
        for row in probabilities:
            top_indices = np.argsort(row)[::-1][:effective_topk]
            predictions = []
            for class_index in top_indices.tolist():
                class_name = self.class_names[class_index] if 0 <= class_index < len(self.class_names) else f"class_{class_index}"
                predictions.append(
                    {
                        "class_id": int(class_index),
                        "class_name": class_name,
                        "confidence": float(row[class_index]),
                    }
                )
            results.append({"predictions": predictions})
        return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ONNX classifier inference")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--source", "-s", required=True)
    parser.add_argument("--output", "-o", default="predictions_classifier_onnx")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--report-file", "--report_file", dest="report_file", type=str, default=None)
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--no-draw", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--draw-only", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import cv2 as cv2_module
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for classifier ONNX inference. Install opencv-python-headless."
        ) from exc
    try:
        import onnxruntime as ort_module
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for classifier ONNX inference."
        ) from exc

    inferencer = ONNXClassifierInference(ort_module, args.onnx, image_size=args.image_size)
    source_path = Path(args.source)
    image_paths = _collect_image_paths(source_path)
    if not image_paths:
        print(f"No images found in: {source_path}")
        return

    output_dir = Path(args.output)
    draw_images = not (args.no_draw or args.benchmark)
    save_images = not (args.no_save_images or args.benchmark or args.draw_only)
    write_report = not (args.no_report or args.benchmark or args.draw_only)
    if save_images or write_report:
        output_dir.mkdir(parents=True, exist_ok=True)
    report_path = ensure_report_path(output_dir, args.report_file) if write_report else None
    report_path_obj = Path(report_path) if report_path else None
    progress = InferenceProgress(total=len(image_paths))
    show_results = args.show and draw_images and not args.benchmark

    print(f"Found {len(image_paths)} images")
    for start_index in range(0, len(image_paths), max(1, int(args.batch_size))):
        batch_paths = image_paths[start_index:start_index + max(1, int(args.batch_size))]
        results = inferencer.run_inference_batch(batch_paths, topk=args.topk)
        for image_path, result in zip(batch_paths, results):
            predictions = result.get("predictions", [])
            output_path = None
            rendered = None
            image = None
            if draw_images or save_images:
                image = cv2_module.imread(str(image_path))
                if image is None:
                    _append_report(
                        report_path_obj,
                        [f"Processing: {image_path.name}", f"  Could not read image: {image_path}"],
                        progress,
                    )
                    continue
            if draw_images and image is not None:
                rendered = _render_prediction(cv2_module, image, predictions)
            if save_images and image is not None:
                output_path = output_dir / f"classifier_onnx_{image_path.name}"
                cv2_module.imwrite(str(output_path), rendered if rendered is not None else image)
            _append_report(
                report_path_obj,
                _build_report_lines(image_path.name, predictions, output_path),
                progress,
            )
            if show_results and rendered is not None:
                cv2_module.imshow("VisionHub Classifier ONNX", rendered)
                key = cv2_module.waitKey(0)
                if key == ord("q"):
                    break

    if show_results:
        cv2_module.destroyAllWindows()
    print("")
    print(progress.summary_line())


if __name__ == "__main__":
    main()
