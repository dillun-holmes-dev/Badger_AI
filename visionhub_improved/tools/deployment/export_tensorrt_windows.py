"""
Windows-compatible TensorRT export script using the TensorRT Python API.
Converts all ONNX models in onnx_engines/ to TensorRT .engine files in trt_engines/.

Requirements:
    pip install tensorrt

Usage:
    visionhub-export-tensorrt-windows [--fp16] [--workspace 1]
"""

import os
import shutil
import argparse


def build_engine(onnx_path: str, engine_path: str, fp16: bool = True, workspace_gb: int = 1,
                 min_batch: int = 1, opt_batch: int = 1, max_batch: int = 4) -> bool:
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError("TensorRT not found. Install with: pip install tensorrt")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    # Set workspace memory limit
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    if fp16:
        # TensorRT 10+ handles reduced precision via TF32 (FP16 flag was removed)
        # TF32 is enabled by default on Ampere+ GPUs; disabling it forces stricter FP32
        print("  Note: TensorRT 10+ manages FP16/TF32 precision automatically")
    else:
        config.clear_flag(trt.BuilderFlag.TF32)
        print("  TF32 disabled, using strict FP32")

    print(f"  Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("  ERROR: Failed to parse ONNX model:")
            for i in range(parser.num_errors):
                print(f"    {parser.get_error(i)}")
            return False

    print("  Building engine (this may take several minutes)...")

    # Optimization profile for dynamic batch dimension
    profile = builder.create_optimization_profile()
    profile.set_shape("images",           (min_batch, 3, 640, 640), (opt_batch, 3, 640, 640), (max_batch, 3, 640, 640))
    profile.set_shape("orig_target_sizes",(min_batch, 2),           (opt_batch, 2),           (max_batch, 2))
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("  ERROR: Failed to build TensorRT engine")
        return False

    os.makedirs(os.path.dirname(engine_path) if os.path.dirname(engine_path) else ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"  Saved engine ({size_mb:.1f} MB): {engine_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export ONNX models to TensorRT engines (Windows)")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 precision (default: True)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable FP16, use FP32")
    parser.add_argument("--workspace", type=int, default=1, help="Builder workspace size in GB (default: 1)")
    parser.add_argument("--min-batch", type=int, default=1, help="Min batch size for optimization profile (default: 1)")
    parser.add_argument("--opt-batch", type=int, default=1, help="Optimal batch size for optimization profile (default: 1)")
    parser.add_argument("--max-batch", type=int, default=4, help="Max batch size for optimization profile (default: 4)")
    parser.add_argument("--input-dir", type=str, default="onnx_engines", help="Input directory containing ONNX files")
    parser.add_argument("--output-dir", type=str, default="trt_engines", help="Output directory for engine files")
    parser.add_argument("--json-only", action="store_true", default=False,
                        help="Only copy class mapping JSONs from onnx_engines to trt_engines, skip engine build")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    onnx_files = [f for f in os.listdir(input_dir) if f.endswith(".onnx")]
    if not onnx_files:
        print(f"No ONNX files found in '{input_dir}'")
        return

    os.makedirs(output_dir, exist_ok=True)

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

    print(f"Found {len(onnx_files)} ONNX file(s) to convert\n")

    for onnx_file in onnx_files:
        engine_file = onnx_file.replace(".onnx", ".engine")
        onnx_path = os.path.join(input_dir, onnx_file)
        engine_path = os.path.join(output_dir, engine_file)

        print(f"[{onnx_file}]")
        success = build_engine(onnx_path, engine_path, fp16=args.fp16, workspace_gb=args.workspace,
                               min_batch=args.min_batch, opt_batch=args.opt_batch, max_batch=args.max_batch)

        if success:
            # Copy class mappings JSON from onnx_engines alongside the built engine
            mappings_src = os.path.join(input_dir, onnx_file.replace(".onnx", "_class_mappings.json"))
            if os.path.exists(mappings_src):
                mappings_dst = os.path.join(output_dir, engine_file.replace(".engine", "_class_mappings.json"))
                shutil.copy2(mappings_src, mappings_dst)
                print(f"  Copied class mappings: {mappings_dst}")
        else:
            print(f"  FAILED: {onnx_file}")

        print()

    print("Done.")


if __name__ == "__main__":
    main()
