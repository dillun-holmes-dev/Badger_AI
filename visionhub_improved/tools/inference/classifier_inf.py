"""Native PyTorch classifier inference CLI."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from visionhub.models.classification import NativePyTorchClassifier, resolve_native_classifier_device


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
cv2 = None


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


def _looks_like_glob(value: str) -> bool:
    return any(token in value for token in ("*", "?", "["))


def _iter_source_images(source_value: str) -> Iterator[Path]:
    source_path = Path(source_value).expanduser()
    if source_path.is_file() and source_path.suffix.lower() in IMAGE_EXTENSIONS:
        yield source_path.resolve()
        return

    if source_path.is_dir():
        for path in sorted(source_path.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path.resolve()
        return

    if _looks_like_glob(source_value):
        parent = source_path.parent if str(source_path.parent) not in {"", "."} else Path.cwd()
        for path in sorted(parent.glob(source_path.name)):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
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


def _render_prediction(image, predictions: Sequence[Dict[str, Any]]):
    if cv2 is None:
        raise RuntimeError("OpenCV is required for classifier rendering. Install opencv-python-headless.")
    rendered = image.copy()
    if not predictions:
        return rendered

    top_prediction = predictions[0]
    class_name = str(top_prediction.get("class_name") or "unknown")
    confidence = float(top_prediction.get("confidence", 0.0) or 0.0)
    label = f"{class_name} {confidence:.3f}"

    box_x1 = 12
    box_y1 = 12
    box_width = max(220, len(label) * 11 + 24)
    box_height = 40
    cv2.rectangle(rendered, (box_x1, box_y1), (box_x1 + box_width, box_y1 + box_height), (30, 30, 30), -1)
    cv2.rectangle(rendered, (box_x1, box_y1), (box_x1 + box_width, box_y1 + box_height), (0, 180, 255), 2)
    cv2.putText(
        rendered,
        label,
        (box_x1 + 10, box_y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return rendered


def _build_report_lines(image_name: str, predictions: Sequence[Dict[str, Any]], output_path: Optional[Path]) -> List[str]:
    lines = [f"Processing: {image_name}", f"  Top {len(predictions)} predictions"]
    for prediction in predictions:
        lines.append(f"    {prediction['class_name']}: {float(prediction['confidence']):.3f}")
    if output_path is not None:
        lines.append(f"  Saved: {output_path}")
    return lines


def _append_report(report_path: Optional[Path], lines: Iterable[str]) -> None:
    if report_path is None:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n\n")


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Native PyTorch classifier inference")
    parser.add_argument("--checkpoint", "--model", "-r", dest="checkpoint", required=True)
    parser.add_argument("--source", "--input", "-s", dest="source", required=True)
    parser.add_argument("--output", "--output-dir", "-o", dest="output", default="predictions_classifier")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=None, help="Optional override for inference resize/crop size")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--topk", type=int, default=1, help="Top-k classes to keep per image. Default matches app output.")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--report-file", "--report_file", dest="report_file", type=str, default=None)
    parser.add_argument("--json-output", dest="json_output", type=str, default=None)
    parser.add_argument("--progress-file", dest="progress_file", type=str, default=None)
    parser.add_argument(
        "--progress-stream",
        choices=("none", "stderr", "stdout"),
        default="stderr",
        help="Emit structured progress events as JSON lines to stderr or stdout.",
    )
    parser.add_argument("--result-ndjson", dest="result_ndjson", type=str, default=None)
    parser.add_argument("--reported-source-root", dest="reported_source_root", type=str, default=None)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--no-draw", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--draw-only", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    return parser


def main() -> int:
    global cv2
    args = build_parser().parse_args()
    try:
        if cv2 is None:
            try:
                import cv2 as cv2_module
            except ImportError as exc:
                raise RuntimeError(
                    "OpenCV is required for classifier inference. Install opencv-python-headless."
                ) from exc
            cv2 = cv2_module

        inferencer = NativePyTorchClassifier(args.checkpoint)
        inferencer.device = resolve_native_classifier_device(args.device)
        if not inferencer.load_model():
            raise RuntimeError(f"Failed to load classifier checkpoint: {args.checkpoint}")
        inferencer.update_config({"batch_size": args.batch_size})
        if args.image_size is not None:
            inferencer.update_config({"model_img_size": int(args.image_size)})

        image_paths = list(_iter_source_images(args.source))
        if not image_paths:
            raise FileNotFoundError(f"No images found for source: {args.source}")

        source_root = _infer_source_root(args.source, image_paths)
        reported_source_root = Path(args.reported_source_root).expanduser() if args.reported_source_root else None
        output_dir = Path(args.output).expanduser().resolve()
        draw_images = not (args.no_draw or args.benchmark)
        save_images = not (args.no_save_images or args.benchmark or args.draw_only)
        write_report = not (args.no_report or args.benchmark or args.draw_only)
        if save_images or write_report:
            output_dir.mkdir(parents=True, exist_ok=True)

        report_path = None
        if write_report:
            report_name = args.report_file or "classifier_predictions.txt"
            report_path = (output_dir / report_name) if not os.path.isabs(report_name) else Path(report_name)
            if report_path.exists():
                report_path.unlink()

        total_start = time.perf_counter()
        progress = ProgressState(total_images=len(image_paths), started_at=total_start)
        progress_emitter = ProgressEmitter(
            progress_file=Path(args.progress_file).expanduser() if args.progress_file else None,
            event_stream=args.progress_stream,
            result_ndjson=Path(args.result_ndjson).expanduser() if args.result_ndjson else None,
            pretty=args.pretty,
        )

        progress_emitter.emit_event(
            "start",
            {
                "success": True,
                "model_input": args.checkpoint,
                "model_path": inferencer.model_path,
                "runtime": "pytorch",
                "model_family": "native_pytorch",
                "source_input": args.source,
                "source_root": str(source_root),
                "reported_source_root": str(reported_source_root) if reported_source_root else None,
                "output_dir": str(output_dir) if (save_images or write_report) else None,
                "summary": progress.snapshot(),
            },
        )

        results: List[Dict[str, Any]] = []
        effective_batch_size = max(1, int(args.batch_size))
        show_results = bool(args.show and draw_images and not args.benchmark)
        should_stop_show = False

        for start_index in range(0, len(image_paths), effective_batch_size):
            batch_paths = image_paths[start_index:start_index + effective_batch_size]
            batch_results, batch_duration_ms = inferencer.run_inference_batch(
                [str(path) for path in batch_paths],
                topk=max(1, int(args.topk)),
            )
            per_image_ms = float(batch_duration_ms) / float(len(batch_paths)) if batch_paths else 0.0

            for offset, (image_path, result) in enumerate(zip(batch_paths, batch_results), start=1):
                image_index = start_index + offset
                reported_path = _remap_reported_path(image_path, source_root, reported_source_root)
                predictions = result.get("predictions", [])
                top_prediction = predictions[: max(1, int(args.topk))]
                annotated_path: Optional[str] = None
                current_result: Dict[str, Any]

                try:
                    raw_image = None
                    rendered_image = None
                    if draw_images or save_images:
                        raw_image = cv2.imread(str(image_path))
                        if raw_image is None:
                            raise RuntimeError(f"Failed to decode image: {image_path}")

                    if draw_images and raw_image is not None:
                        rendered_image = _render_prediction(raw_image, top_prediction)

                    if save_images and raw_image is not None:
                        target_path = _resolve_output_path(
                            output_dir,
                            image_path,
                            source_root,
                            annotated=draw_images,
                        )
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        frame_to_write = rendered_image if rendered_image is not None else raw_image
                        if not cv2.imwrite(str(target_path), frame_to_write):
                            raise RuntimeError(f"Failed to write output image: {target_path}")
                        annotated_path = str(target_path)

                    confidence = float(top_prediction[0].get("confidence", 0.0) or 0.0) if top_prediction else 0.0
                    current_result = {
                        "image_path": reported_path,
                        "annotated_path": annotated_path,
                        "predictions": top_prediction,
                        "inference_type": "classifier",
                        "confidence": confidence,
                        "inference_time_ms": per_image_ms,
                    }
                    _append_report(report_path, _build_report_lines(image_path.name, top_prediction, Path(annotated_path) if annotated_path else None))

                    if show_results and rendered_image is not None:
                        cv2.imshow("VisionHub Classifier", rendered_image)
                        key = cv2.waitKey(0)
                        if key == ord("q"):
                            should_stop_show = True
                except Exception as image_exc:
                    current_result = _error_result(reported_path, str(image_exc))
                    _append_report(report_path, [f"Processing: {image_path.name}", f"  Error: {image_exc}"])

                results.append(current_result)
                progress_summary = progress.advance(current_result)
                progress_emitter.emit_result(current_result)
                progress_emitter.emit_event(
                    "image_complete",
                    {
                        "image_index": image_index,
                        "image_path": reported_path,
                        "result": current_result,
                        "summary": progress_summary,
                    },
                )

                if should_stop_show:
                    break
            if should_stop_show:
                break

        if show_results:
            cv2.destroyAllWindows()

        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0
        error_count = sum(1 for item in results if item.get("inference_type") == "error")
        success_count = len(results) - error_count
        total_inference_ms = sum(float(item.get("inference_time_ms", 0.0) or 0.0) for item in results)
        average_inference_ms = total_inference_ms / success_count if success_count else 0.0

        payload = {
            "success": True,
            "model_input": args.checkpoint,
            "model_path": inferencer.model_path,
            "runtime": "pytorch",
            "model_family": "native_pytorch",
            "source_input": args.source,
            "source_root": str(source_root),
            "reported_source_root": str(reported_source_root) if reported_source_root else None,
            "output_dir": str(output_dir) if (save_images or write_report) else None,
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
        print(f"visionhub-infer-classifier failed: {exc}", file=sys.stderr)
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
