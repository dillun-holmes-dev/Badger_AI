"""
Linux TensorRT export script using trtexec.
Converts all ONNX models in onnx_engines/ to TensorRT .engine files in trt_engines/.

Requirements:
    TensorRT installed (trtexec available on PATH, ideally /opt/tensorrt/bin/trtexec)

Usage:
    visionhub-export-tensorrt
    visionhub-export-tensorrt --no-fp16
    visionhub-export-tensorrt --json-only   # copy JSONs only, skip rebuild
"""
import os
import sys
import shutil
import argparse
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../inference"))
try:
    from tools.inference.class_mapping_utils import find_model_metadata_json
except ImportError:
    from class_mapping_utils import find_model_metadata_json


def resolve_trtexec(explicit_path=None):
    """Find trtexec from an explicit path, PATH, or common install locations."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)

    path_hit = shutil.which("trtexec")
    if path_hit:
        candidates.append(path_hit)

    candidates.extend([
        "/opt/tensorrt/bin/trtexec",
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/tensorrt/bin/trtexec",
        "/usr/local/bin/trtexec",
    ])

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Export ONNX models to TensorRT engines (Linux)")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 precision (default: True)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable FP16, use FP32")
    parser.add_argument("--trtexec", type=str, default=None,
                        help="Path to trtexec binary (default: auto-detect from PATH/common locations)")
    parser.add_argument("--input-dir", type=str, default="onnx_engines",
                        help="Input directory containing ONNX files (default: onnx_engines)")
    parser.add_argument("--output-dir", type=str, default="trt_engines",
                        help="Output directory for engine files (default: trt_engines)")
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--json-only", action="store_true", default=False,
                        help="Only copy class mapping JSONs from onnx_engines to trt_engines, skip engine build")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    onnx_files = [f for f in os.listdir(input_dir) if f.endswith(".onnx")]
    if not onnx_files:
        print(f"No ONNX files found in '{input_dir}'")
        sys.exit(0)

    if args.json_only:
        print("--json-only: copying class mapping JSONs only, skipping engine build\n")
        for onnx_file in onnx_files:
            engine_file = onnx_file.replace(".onnx", ".engine")
            mappings_src = os.path.join(input_dir, onnx_file.replace(".onnx", "_class_mappings.json"))
            if os.path.exists(mappings_src):
                mappings_dst = os.path.join(output_dir, engine_file.replace(".engine", "_class_mappings.json"))
                shutil.copy2(mappings_src, mappings_dst)
                print(f"  Copied: {mappings_dst}")
            else:
                print(f"  No JSON found for: {onnx_file}")
        print("\nDone.")
        return

    trtexec_path = resolve_trtexec(args.trtexec)
    if trtexec_path is None:
        requested = args.trtexec or "auto-detect"
        print(f"ERROR: trtexec not found ({requested}).")
        print("Install TensorRT so `trtexec` is on PATH, or pass --trtexec /path/to/trtexec.")
        sys.exit(1)

    print(f"Using trtexec: {trtexec_path}")
    print(f"Found {len(onnx_files)} ONNX file(s) to convert\n")

    nb, ob, mb = args.min_batch, args.opt_batch, args.max_batch

    for onnx_file in onnx_files:
        engine_file = onnx_file.replace(".onnx", ".engine")
        onnx_path = os.path.join(input_dir, onnx_file)
        engine_path = os.path.join(output_dir, engine_file)

        print(f"[{onnx_file}]")

        onnx_metadata = find_model_metadata_json(onnx_path)
        image_size = int(onnx_metadata.get("image_size") or 640)

        min_shapes = f"images:{nb}x3x{image_size}x{image_size},orig_target_sizes:{nb}x2"
        opt_shapes = f"images:{ob}x3x{image_size}x{image_size},orig_target_sizes:{ob}x2"
        max_shapes = f"images:{mb}x3x{image_size}x{image_size},orig_target_sizes:{mb}x2"

        cmd = [
            trtexec_path,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            f"--minShapes={min_shapes}",
            f"--optShapes={opt_shapes}",
            f"--maxShapes={max_shapes}",
        ]
        if args.fp16:
            cmd.append("--fp16")

        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"  FAILED: {onnx_file} (exit code {result.returncode})")
        else:
            print(f"  Saved engine: {engine_path}")
            mappings_src = os.path.join(input_dir, onnx_file.replace(".onnx", "_class_mappings.json"))
            if os.path.exists(mappings_src):
                mappings_dst = os.path.join(output_dir, engine_file.replace(".engine", "_class_mappings.json"))
                shutil.copy2(mappings_src, mappings_dst)
                print(f"  Copied class mappings: {mappings_dst}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
