from pathlib import Path
import time


def ensure_report_path(output_dir, report_file=None):
    output_dir = Path(output_dir)
    if report_file:
        report_path = Path(report_file)
        if not report_path.is_absolute():
            report_path = output_dir / report_path
    else:
        report_path = output_dir / "prediction_report.txt"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("", encoding="utf-8")
    return report_path


def build_prediction_log(image_name, detections, output_path=None, failure_message=None):
    lines = [f"Processing: {image_name}"]

    if failure_message is not None:
        lines.append(f"  {failure_message}")
        return lines

    lines.append(f"  Found {len(detections)} detections")
    for label_name, score in detections:
        lines.append(f"    {label_name}: {score:.3f}")

    if output_path is not None:
        lines.append(f"  Saved: {output_path}")

    return lines


class InferenceProgress:
    def __init__(self, total=None):
        self.total = total
        self.start_time = time.perf_counter()
        self.last_time = self.start_time
        self.processed = 0

    def record(self):
        now = time.perf_counter()
        self.processed += 1
        elapsed = max(now - self.start_time, 1e-9)
        delta = max(now - self.last_time, 1e-9)
        self.last_time = now
        return {
            "processed": self.processed,
            "total": self.total,
            "elapsed": elapsed,
            "avg_fps": self.processed / elapsed,
            "last_fps": 1.0 / delta,
            "avg_ms": 1000.0 * elapsed / self.processed,
        }

    def summary_line(self):
        if self.processed == 0:
            return "Summary: 0 images processed"

        elapsed = max(time.perf_counter() - self.start_time, 1e-9)
        avg_fps = self.processed / elapsed
        avg_ms = 1000.0 * elapsed / self.processed
        total_text = f"/{self.total}" if self.total is not None else ""
        return (
            f"Summary: {self.processed}{total_text} images | "
            f"{avg_fps:.2f} FPS avg | {avg_ms:.1f} ms/img avg | {elapsed:.1f}s elapsed"
        )


def emit_prediction_log(report_path, image_name, detections, output_path=None, failure_message=None, progress=None):
    lines = build_prediction_log(
        image_name=image_name,
        detections=detections,
        output_path=output_path,
        failure_message=failure_message,
    )
    block = "\n".join(lines)
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
    else:
        print(".", end="", flush=True)
    if report_path is not None:
        with Path(report_path).open("a", encoding="utf-8") as handle:
            handle.write(block)
            handle.write("\n\n")
