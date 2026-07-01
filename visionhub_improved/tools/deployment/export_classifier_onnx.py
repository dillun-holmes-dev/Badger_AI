"""Export a native PyTorch classifier checkpoint to ONNX."""

import argparse

from visionhub.models.classification import export_native_classifier_to_onnx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export native PyTorch classifier to ONNX")
    parser.add_argument("--checkpoint", "-r", required=True, help="Checkpoint file or training directory")
    parser.add_argument("--output", "-o", default=None, help="Optional output .onnx file path")
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset version")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy export batch size")
    parser.add_argument("--check", action="store_true", default=False)
    parser.add_argument("--simplify", action="store_true", default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = export_native_classifier_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        check=bool(args.check),
        simplify=bool(args.simplify),
        opset_version=int(args.opset),
        batch_size=int(args.batch_size),
    )
    print(f"Exported ONNX model to: {result['onnx_path']}")
    print(f"Saved metadata to: {result['metadata_path']}")


if __name__ == "__main__":
    main()
