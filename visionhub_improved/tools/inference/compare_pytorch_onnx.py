"""
Diagnostic: compare PyTorch vs ONNX inference on the same image.
Helps identify if there is a genuine score/detection discrepancy.

Usage:
    python tools/inference/compare_pytorch_onnx.py \
        --checkpoint output/detrpose_hgnetv2_s_custom/checkpoint_best_regular.pth \
        --onnx onnx_engines/detrpose_hgnetv2_s_custom.onnx \
        --image path/to/test_image.jpg \
        --conf 0.10
"""
import argparse
import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import onnxruntime as ort
from torchvision.ops.boxes import nms as torchvision_nms

script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from visionhub.core import instantiate


# ────────────────────────────────────────────────────────────────────────────
# Preprocessing (identical for both)
# ────────────────────────────────────────────────────────────────────────────
def preprocess(img_bgr, size=640):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    arr = img_resized.astype(np.float32) / 255.0
    # numpy version (for ONNX)
    np_chw = arr.transpose(2, 0, 1)[np.newaxis]
    # torch version (for PyTorch)
    tensor = torch.from_numpy(np_chw)
    return np_chw, tensor


# ────────────────────────────────────────────────────────────────────────────
# PyTorch inference
# ────────────────────────────────────────────────────────────────────────────
def run_pytorch(checkpoint_path, config_module, img_bgr, conf_thresh, device="cpu"):
    cfg = importlib.import_module(config_module)
    model = instantiate(cfg.model)
    postprocessor = instantiate(cfg.postprocessor)
    if hasattr(postprocessor, "set_dcc") and hasattr(model, "dcc"):
        postprocessor.set_dcc(model.dcc)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ema = ckpt.get("ema")
    if ema and isinstance(ema, dict) and "module" in ema:
        state = ema["module"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    state = {
        k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v
        for k, v in state.items()
    }
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    _, tensor = preprocess(img_bgr)
    tensor = tensor.to(device)

    h, w = img_bgr.shape[:2]
    orig_size = torch.tensor([[h, w]], dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        results = postprocessor(outputs, orig_size)

    r = results[0]
    scores = r["scores"].cpu().numpy()
    labels = r["labels"].cpu().numpy()

    order = np.argsort(-scores)
    scores = scores[order]
    labels = labels[order]

    mask = scores >= conf_thresh
    return scores[mask], labels[mask]


# ────────────────────────────────────────────────────────────────────────────
# ONNX inference — applies same class-aware NMS as PyTorch PostProcess
# ────────────────────────────────────────────────────────────────────────────
NMS_IOU_THRESHOLD = 0.65

def run_onnx(onnx_path, img_bgr, conf_thresh):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    output_names = [o.name for o in sess.get_outputs()]

    np_chw, _ = preprocess(img_bgr)
    h, w = img_bgr.shape[:2]
    orig_size = np.array([[h, w]], dtype=np.int64)

    outputs = sess.run(None, {"images": np_chw, "orig_target_sizes": orig_size})

    scores    = outputs[0][0]   # (N,)
    labels    = outputs[1][0]   # (N,)
    boxes     = outputs[3][0] if "boxes" in output_names and len(outputs) > 3 else None  # (N, 4)

    # Apply class-aware NMS (mirrors PyTorch PostProcess non-deploy path)
    if boxes is not None and len(scores) > 0:
        t_boxes  = torch.from_numpy(boxes.astype(np.float32))
        t_scores = torch.from_numpy(scores.astype(np.float32))
        t_labels = torch.from_numpy(labels.astype(np.float32))
        max_coord = t_boxes.max()
        offsets = t_labels * (max_coord + 1)
        keep = torchvision_nms(t_boxes + offsets[:, None], t_scores, NMS_IOU_THRESHOLD).numpy()
        scores = scores[keep]
        labels = labels[keep]

    order = np.argsort(-scores)
    scores = scores[order]
    labels = labels[order]

    mask = scores >= conf_thresh
    return scores[mask], labels[mask]


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", required=True,
                        help="Path to .pth checkpoint")
    parser.add_argument("--onnx", "-o", required=True,
                        help="Path to .onnx model")
    parser.add_argument("--image", "-i", required=True,
                        help="Path to a single test image")
    parser.add_argument("--conf", type=float, default=0.10,
                        help="Confidence threshold (low to see all candidates)")
    parser.add_argument("--config",
                        default="configs.detrpose.detrpose_hgnetv2_s_custom",
                        help="Python config module path")
    parser.add_argument("--device", default="cpu",
                        help="Device for PyTorch (cpu/cuda)")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: could not read {args.image}")
        sys.exit(1)

    print(f"\nImage: {args.image}  ({img.shape[1]}x{img.shape[0]})")
    print(f"Confidence threshold: {args.conf}")
    print("=" * 60)

    print("\n--- PyTorch (inference.py path) ---")
    pt_scores, pt_labels = run_pytorch(
        args.checkpoint, args.config, img, args.conf, device=args.device)
    print(f"  Detections above {args.conf}: {len(pt_scores)}")
    for s, l in zip(pt_scores[:20], pt_labels[:20]):
        print(f"    class={int(l):3d}  score={s:.4f}")

    print("\n--- ONNX (deploy path, no extra NMS) ---")
    onnx_scores, onnx_labels = run_onnx(args.onnx, img, args.conf)
    print(f"  Detections above {args.conf}: {len(onnx_scores)}")
    for s, l in zip(onnx_scores[:20], onnx_labels[:20]):
        print(f"    class={int(l):3d}  score={s:.4f}")

    print("\n--- Score comparison (top-20 from each) ---")
    n = max(len(pt_scores), len(onnx_scores))
    n = min(n, 20)
    print(f"  {'Rank':<5}  {'PyTorch score':<18} {'PyTorch class':<16}  "
          f"{'ONNX score':<18} {'ONNX class'}")
    print("  " + "-" * 75)
    for i in range(n):
        pt_s  = f"{pt_scores[i]:.4f}"  if i < len(pt_scores)   else "—"
        pt_l  = f"{int(pt_labels[i])}" if i < len(pt_labels)   else "—"
        on_s  = f"{onnx_scores[i]:.4f}" if i < len(onnx_scores) else "—"
        on_l  = f"{int(onnx_labels[i])}"if i < len(onnx_labels) else "—"
        flag  = " ← MISMATCH" if (pt_s != "—" and on_s != "—"
                                  and abs(float(pt_s) - float(on_s)) > 0.05) else ""
        print(f"  {i+1:<5}  {pt_s:<18} {pt_l:<16}  {on_s:<18} {on_l}{flag}")

    if len(pt_scores) and len(onnx_scores):
        max_diff = max(
            abs(float(f"{pt_scores[i]:.4f}") - float(f"{onnx_scores[i]:.4f}"))
            for i in range(min(len(pt_scores), len(onnx_scores)))
        )
        print(f"\n  Max score diff (top-{min(len(pt_scores),len(onnx_scores))}): {max_diff:.4f}")
        if max_diff < 0.02:
            print("  ✅ Scores match closely — ONNX export is numerically correct.")
        else:
            print("  ⚠️  Scores diverge — there is a genuine numerical discrepancy.")


if __name__ == "__main__":
    main()
