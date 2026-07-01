<h2 align="center">VisionHub</h2>

<p align="center">
  Training, inference, export, and benchmarking tools for custom pose, detection, and image-classification workflows.
</p>

<p align="center">
  <a href="LICENSE">
    <img alt="license" src="https://img.shields.io/badge/license-apache%202.0-blue?style=for-the-badge">
  </a>
  <a href="https://www.arxiv.org/abs/2506.13027">
    <img alt="arxiv" src="https://img.shields.io/badge/-DETRPose-gray?style=for-the-badge&logo=arxiv&labelColor=red">
  </a>
</p>

VisionHub started as a DETRPose-based training repo and has grown into a broader computer-vision workspace with:

- config-driven training for custom pose and detection model families
- native PyTorch classifier training built on torchvision backbones
- PyTorch, ONNX, and TensorRT inference utilities
- ONNX export tooling
- latency and FLOPs benchmarking helpers
- Docker and Colab workflows for reproducible training and deployment

It is best thought of as a practical model-workbench for custom datasets rather than a single-model repository.

## Table of contents

- [What VisionHub Is](#what-visionhub-is)
- [Supported Workflows](#supported-workflows)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Docker Workflow](#docker-workflow)
- [Dataset Formats](#dataset-formats)
- [Training](#training)
- [Inference](#inference)
- [Export And Deployment](#export-and-deployment)
- [Benchmarking](#benchmarking)
- [Notebooks](#notebooks)
- [Project Notes](#project-notes)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

## What VisionHub Is

VisionHub currently supports three broad problem families.

### Pose estimation

- `DETRPose`: end-to-end transformer pose estimation
- `RTMOPose`: one-stage real-time pose estimation
- `RTMDet-Pose`: detector-style pose estimation with keypoints

These families share:

- a common dataset layout
- config-based model definition under `configs/`
- a shared training entrypoint
- checkpoint metadata for class mappings and skeleton definitions
- PyTorch, ONNX, and TensorRT deployment paths

### Object detection

- `DETRDet`: end-to-end transformer detection
- `RTMDet`: one-stage dense detection
- `RTMODet`: RTMO-style dense detection

These families share:

- the same COCO-style dataset root layout used by the pose workflows
- config-based model definition under `configs/`
- family-specific train, inference, and ONNX export CLIs
- checkpoint metadata for class mappings and dataset-derived dimensions
- support for empty or negative images so the model learns what not to predict

### Image classification

VisionHub also now includes a native PyTorch classifier workflow based on torchvision architectures, with:

- ImageFolder-style dataset training
- pretrained torchvision backbones
- augmentation presets
- automatic optimizer selection
- ONNX export with sidecar metadata
- PyTorch and ONNX inference CLIs

## Supported Workflows

### Installed CLI commands

After installation, these commands are exposed:

| Area | Command |
| --- | --- |
| Pose training | `visionhub-train` |
| DETR detection training | `visionhub-train-detr-detect` |
| RTMO detection training | `visionhub-train-rtmo-detect` |
| RTMDet detection training | `visionhub-train-rtmdet` |
| RTMDet-Pose training | `visionhub-train-rtmdetpose` |
| Classifier training | `visionhub-train-classifier` |
| DETRPose inference | `visionhub-infer-detr` |
| Unified detection inference endpoint | `visionhub-infer-detect` |
| Unified pose inference endpoint | `visionhub-infer-pose` |
| RTMOPose inference | `visionhub-infer-rtmo` |
| RTMDet-Pose inference | `visionhub-infer-rtmdetpose` |
| DETR detection inference | `visionhub-infer-detr-detect` |
| RTMO detection inference | `visionhub-infer-rtmo-detect` |
| RTMDet detection inference | `visionhub-infer-rtmdet` |
| Native classifier inference | `visionhub-infer-classifier` |
| Generic ONNX inference | `visionhub-infer-onnx` |
| Classifier ONNX inference | `visionhub-infer-classifier-onnx` |
| TensorRT inference | `visionhub-infer-trt` |
| DETRPose ONNX export | `visionhub-export-onnx` |
| RTMOPose ONNX export | `visionhub-export-rtmo-onnx` |
| RTMDet-Pose ONNX export | `visionhub-export-rtmdetpose-onnx` |
| DETR detection ONNX export | `visionhub-export-detr-detect-onnx` |
| RTMO detection ONNX export | `visionhub-export-rtmo-detect-onnx` |
| RTMDet detection ONNX export | `visionhub-export-rtmdet-onnx` |
| Classifier ONNX export | `visionhub-export-classifier-onnx` |
| TensorRT batch export | `visionhub-export-tensorrt` |
| TensorRT batch export (Windows) | `visionhub-export-tensorrt-windows` |
| FLOPs / params benchmark | `visionhub-benchmark-info` |
| PyTorch latency benchmark | `visionhub-benchmark-torch` |
| TensorRT latency benchmark | `visionhub-benchmark-trt` |

### Model families

#### DETRPose configs

- `configs/detrpose/detrpose_hgnetv2_n_custom.py`
- `configs/detrpose/detrpose_hgnetv2_s_custom.py`
- `configs/detrpose/detrpose_hgnetv2_m_custom.py`
- `configs/detrpose/detrpose_hgnetv2_l_custom.py`
- `configs/detrpose/detrpose_hgnetv2_x_custom.py`

#### RTMOPose configs

- `configs/rtmopose/rtmopose_hgnetv2_n_custom.py`
- `configs/rtmopose/rtmopose_hgnetv2_s_custom.py`
- `configs/rtmopose/rtmopose_hgnetv2_m_custom.py`
- `configs/rtmopose/rtmopose_hgnetv2_l_custom.py`

#### RTMDet-Pose configs

- `configs/rtmdetpose/rtmdetpose_hgnetv2_n_custom.py`
- `configs/rtmdetpose/rtmdetpose_hgnetv2_s_custom.py`
- `configs/rtmdetpose/rtmdetpose_hgnetv2_m_custom.py`
- `configs/rtmdetpose/rtmdetpose_hgnetv2_l_custom.py`

#### DETR detection configs

- `configs/detrdet/detrdet_hgnetv2_n_custom.py`
- `configs/detrdet/detrdet_hgnetv2_s_custom.py`
- `configs/detrdet/detrdet_hgnetv2_m_custom.py`
- `configs/detrdet/detrdet_hgnetv2_l_custom.py`
- `configs/detrdet/detrdet_hgnetv2_x_custom.py`

#### RTMDet detection configs

- `configs/rtmdet/rtmdet_hgnetv2_n_custom.py`
- `configs/rtmdet/rtmdet_hgnetv2_s_custom.py`
- `configs/rtmdet/rtmdet_hgnetv2_m_custom.py`
- `configs/rtmdet/rtmdet_hgnetv2_l_custom.py`
- `configs/rtmdet/rtmdet_hgnetv2_x_custom.py`

#### RTMODet configs

- `configs/rtmodet/rtmodet_hgnetv2_n_custom.py`
- `configs/rtmodet/rtmodet_hgnetv2_s_custom.py`
- `configs/rtmodet/rtmodet_hgnetv2_m_custom.py`
- `configs/rtmodet/rtmodet_hgnetv2_l_custom.py`
- `configs/rtmodet/rtmodet_hgnetv2_x_custom.py`

#### Native classifier architectures

- `efficientnet_b0`
- `efficientnet_b1`
- `efficientnet_b3`
- `efficientnet_v2_s`
- `efficientnet_v2_m`
- `efficientnet_v2_l`
- `convnext_tiny`
- `convnext_small`
- `mobilenet_v3_large`
- `regnet_y_400mf`
- `regnet_y_800mf`
- `resnet18`
- `resnet34`

## Repository Layout

The main parts of the repo are:

| Path | Purpose |
| --- | --- |
| `visionhub/` | package code, model components, runtime utilities, CLI wrappers |
| `configs/` | lazy configs for pose and detection model families |
| `tools/inference/` | PyTorch, ONNX, and TensorRT inference scripts |
| `tools/deployment/` | ONNX and TensorRT export scripts |
| `tools/benchmark/` | FLOPs, PyTorch latency, and TensorRT latency helpers |
| `tools/training/` | standalone classifier training tooling |
| `train.py` | main config-driven pose and detection training entrypoint |
| `docker/` | helper script for running the TensorRT container |
| `Dockerfile` | CUDA + TensorRT + Python 3.11.9 environment |
| `notebooks/` | Colab-oriented training notebooks |

## Installation

### Python support

The package metadata targets Python `3.8` through `3.11`.

### Local install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

If you want ONNX export and ONNX inference support:

```bash
pip install -e ".[export]"
```

If you also want notebook support:

```bash
pip install -e ".[export,notebook]"
```

For CUDA training, install a matching PyTorch build first, then install VisionHub:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 torchvision==0.20.1
pip install -e ".[export]"
```

### Optional dependency groups

- `.[export]`: ONNX, ONNX Runtime, ONNX simplification helpers
- `.[tensorrt]`: Python-side TensorRT tooling dependencies
- `.[notebook]`: Jupyter and notebook support
- `.[dev]`: build, test, lint, and publishing tools

## Docker Workflow

For CUDA and TensorRT-heavy workflows, the container path is the most reproducible setup.

Build the image:

```bash
docker build -t visionhub:tensorrt-py3.11.9 .
```

Start an interactive shell:

```bash
./docker/run.sh
```

Run a one-off command:

```bash
./docker/run.sh python train.py \
  --config_file configs/rtmopose/rtmopose_hgnetv2_s_custom.py \
  --device cuda \
  --amp \
  --data_root ./data/coco
```

What the container gives you:

- NVIDIA TensorRT base image
- Python `3.11.9`
- CUDA-compatible PyTorch
- repo dependencies from `docker_requirements.txt`
- TensorRT Python bindings pinned to the container release

Notes:

- `docker/run.sh` bind-mounts the repo to `/workspace/VisionHub`
- `--gpus all` requires the NVIDIA container runtime
- outputs stay on the host because the repo is bind-mounted

## Dataset Formats

VisionHub supports three dataset conventions depending on the workflow.

### Pose datasets

All pose model families expect the same dataset root:

```text
data/
  coco/
    Info.json
    kpts_definition.json
    train/
      coco_instances.json
      images/
      masks/                  # optional
    val/
      coco_instances.json
      images/
      masks/                  # optional
```

Required files:

1. `Info.json`
2. `kpts_definition.json`
3. `train/coco_instances.json`
4. `val/coco_instances.json`

Behavior:

- class count is derived from category IDs
- keypoint count is derived from annotations
- checkpoint exports preserve class mappings and skeleton connections
- `train/` and `val/` can be archived and extracted at runtime in some workflows

`kpts_definition.json` is used to describe category-level keypoint metadata and skeletons. `RTMOPose` and `RTMDet-Pose` also support per-keypoint `sigmas` in the category definitions.

### Detection datasets

Detection configs use the same root layout and COCO annotation files:

```text
data/
  coco/
    train/
      coco_instances.json
      images/
    val/
      coco_instances.json
      images/
```

Detection-specific behavior:

- class mappings are derived from COCO categories
- keypoints are not required
- empty or negative images are preserved during training and evaluation
- checkpoint metadata stores the class mapping so inference and export do not need hardcoded paths

### Classification datasets

Native classifier training expects ImageFolder-style splits:

```text
dataset_root/
  train/
    class_a/
      image1.jpg
    class_b/
      image2.jpg
  val/
    class_a/
      image3.jpg
    class_b/
      image4.jpg
```

Rules:

- `train/` and `val/` must contain the same class folders
- images may be nested inside each class directory
- class names are derived from folder names

## Training

### Pose training

`visionhub-train` is the shared training entrypoint for DETRPose, RTMOPose, and RTMDet-Pose. From a repo checkout, `python train.py` still works as the direct script entrypoint.

Basic example:

```bash
visionhub-train \
  --config_file configs/rtmdetpose/rtmdetpose_hgnetv2_s_custom.py \
  --device cuda \
  --amp \
  --data_root ./data/coco
```

Recommended shortcuts:

```bash
visionhub-train --config_file configs/detrpose/detrpose_hgnetv2_s_custom.py --device cuda --amp --data_root ./data/coco
visionhub-train --config_file configs/rtmopose/rtmopose_hgnetv2_s_custom.py --device cuda --amp --data_root ./data/coco
visionhub-train --config_file configs/rtmdetpose/rtmdetpose_hgnetv2_s_custom.py --device cuda --amp --data_root ./data/coco
```

Common runtime overrides:

- `--output_dir`
- `--epochs`
- `--batch_size`
- `--num_workers`
- `--image_size`
- `--data_root`
- `--save_checkpoint_interval`
- `--grad_accum_steps`
- `--sync_bn`
- `--use_ema` / `--no_ema`
- `--compile_model`
- `--compile_mode`
- `--pretrained_backbone`

Example with overrides:

```bash
visionhub-train \
  --config_file configs/rtmdetpose/rtmdetpose_hgnetv2_m_custom.py \
  --device cuda \
  --amp \
  --data_root ./data/coco \
  --batch_size 16 \
  --num_workers 4 \
  --image_size 640 \
  --epochs 120
```

### Multi-GPU pose training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --master_port=7777 \
  --nproc_per_node=4 \
  --module visionhub.cli.train \
  --config_file configs/<family>/<config>.py \
  --device cuda \
  --amp \
  --data_root ./data/coco
```

### Pose evaluation

```bash
visionhub-train \
  --config_file configs/rtmdetpose/rtmdetpose_hgnetv2_s_custom.py \
  --device cuda \
  --amp \
  --resume output/rtmdetpose_hgnetv2_s_custom/checkpoint_best_regular.pth \
  --eval
```

### Detection training

Detection models use the same `train.py` backend, but the repo exposes family-specific CLI shortcuts for the custom detection families.

Examples:

```bash
visionhub-train-detr-detect \
  --variant s \
  --device cuda \
  --amp \
  --data_root ./data/coco

visionhub-train-rtmo-detect \
  --variant s \
  --device cuda \
  --amp \
  --data_root ./data/coco

visionhub-train-rtmdet \
  --variant s \
  --device cuda \
  --amp \
  --data_root ./data/coco
```

You can still target an exact config manually:

```bash
visionhub-train \
  --config_file configs/rtmdet/rtmdet_hgnetv2_m_custom.py \
  --device cuda \
  --amp \
  --data_root ./data/coco \
  --batch_size 16 \
  --epochs 120
```

Detection notes:

- negatives are kept by default through `allow_empty=True`
- dataset metadata is derived from COCO categories and baked into checkpoints
- the same runtime overrides used by pose training also apply here

### Native classifier training

`visionhub-train-classifier` runs the standalone torchvision-based classifier workflow.

Example:

```bash
visionhub-train-classifier \
  --source_root ./data/classifier_dataset \
  --output_root ./output/classifiers/efficientnet_v2_s_run1 \
  --architecture efficientnet_v2_s \
  --epochs 50 \
  --batch_size 16 \
  --image_size 224 \
  --optimizer auto \
  --augmentation_preset phantom \
  --use-pretrained-weights \
  --device auto
```

This workflow includes:

- architecture alias normalization
- auto optimizer resolution
- augmentation presets
- runtime tensor caching when appropriate
- least-confused checkpoint selection
- ONNX-export-friendly checkpoint metadata

## Inference

Most inference CLIs support the same quality-of-life flags:

- `--no-save-images`
- `--no-draw`
- `--no-report`
- `--draw-only`
- `--benchmark`

### Unified pose CLI endpoint

```bash
visionhub-infer-pose \
  --model output/rtmopose_hgnetv2_s_custom/checkpoint_best_regular.pth \
  --source path/to/test_data \
  --output-dir predictions_pose \
  --json-output predictions_pose/results.json \
  --pretty
```

This command is intended for application-side subprocess calls. It auto-detects:

- PyTorch checkpoints for `DETRPose`, `RTMOPose`, and `RTMDet-Pose`
- exported `.onnx` pose models
- exported TensorRT `.engine` / `.trt` pose models

Its stdout payload is JSON and each per-image result follows the same shape used by the Phantom Factory inference flow:

- `image_path`
- `annotated_path`
- `predictions`
- `inference_type`
- `confidence`
- `inference_time_ms`

Useful options:

- `--runtime auto|pytorch|onnx|tensorrt`
- `--family auto|detrpose|rtmopose|rtmdet`
- `--config configs/rtmopose/rtmopose_hgnetv2_s_custom.py`
- `--reported-source-root /original/test_data/root`
- `--progress-file /tmp/pose_progress.json`
- `--progress-stream stderr`
- `--result-ndjson predictions_pose/results.ndjson`
- `--no-save-images`
- `--no-draw`

For incremental backend updates, `visionhub-infer-pose` now supports three opt-in output channels while preserving the final aggregate JSON payload:

- `--progress-file` rewrites a JSON snapshot as each image completes so a caller can poll for state.
- `--progress-stream` emits structured `VISIONHUB_PROGRESS {...}` lines to `stderr` or `stdout`.
- `--result-ndjson` appends one per-image result object per line as work completes.

### Unified detection CLI endpoint

```bash
visionhub-infer-detect \
  --model output/rtmdetdet_hgnetv2_s_custom/checkpoint_best_regular.pth \
  --source path/to/test_data \
  --output-dir predictions_detect \
  --json-output predictions_detect/results.json \
  --pretty
```

This command is intended for application-side subprocess calls. It auto-detects:

- PyTorch detection checkpoints for `DETR`, `RTMO`, and `RTMDet` detection families
- exported detection `.onnx` models
- exported detection TensorRT `.engine` / `.trt` models

Its stdout payload is JSON and each per-image result follows the same application-friendly shape used by the unified pose flow:

- `image_path`
- `annotated_path`
- `predictions`
- `inference_type`
- `confidence`
- `inference_time_ms`

Useful options:

- `--runtime auto|pytorch|onnx|tensorrt`
- `--family auto|detrdet|rtmodet|rtmdetdet`
- `--variant auto|n|s|m|l|x`
- `--config configs/rtmdetdet/rtmdetdet_hgnetv2_s_custom.py`
- `--reported-source-root /original/test_data/root`
- `--duplicate-iou-thresh 0.5`
- `--progress-file /tmp/detect_progress.json`
- `--progress-stream stderr`
- `--result-ndjson predictions_detect/results.ndjson`
- `--no-save-images`
- `--no-draw`

For incremental backend updates, `visionhub-infer-detect` supports the same three opt-in output channels as `visionhub-infer-pose`:

- `--progress-file` rewrites a JSON snapshot as each image completes so a caller can poll for state.
- `--progress-stream` emits structured `VISIONHUB_PROGRESS {...}` lines to `stderr` or `stdout`.
- `--result-ndjson` appends one per-image result object per line as work completes.

### DETRPose PyTorch inference

```bash
visionhub-infer-detr \
  --checkpoint output/detrpose_hgnetv2_s_custom/checkpoint_best_regular.pth \
  --input path/to/image_or_folder_or_archive \
  --output predictions \
  --conf 0.35 \
  --device cuda
```

Highlights:

- config can be auto-detected in the DETRPose workflow
- supports single images, folders, `.zip`, and `.7z`
- recursively scans nested folders
- uses EMA weights by default

### RTMOPose PyTorch inference

```bash
visionhub-infer-rtmo \
  --checkpoint output/rtmopose_hgnetv2_s_custom/checkpoint_best_regular.pth \
  --config configs.rtmopose.rtmopose_hgnetv2_s_custom \
  --source path/to/image_or_folder \
  --output predictions \
  --conf 0.35 \
  --device cuda
```

### RTMDet-Pose PyTorch inference

```bash
visionhub-infer-rtmdetpose \
  --checkpoint output/rtmdetpose_hgnetv2_s_custom/checkpoint_best_regular.pth \
  --config configs.rtmdetpose.rtmdetpose_hgnetv2_s_custom \
  --source path/to/image_or_folder \
  --output_dir predictions \
  --conf_thresh 0.35 \
  --device cuda
```

RTMDet-specific notes:

- `--nms_thresh` can override config-time NMS
- `--show` enables interactive display
- recursive image discovery is built in

### Detection PyTorch inference

DETR detection:

```bash
visionhub-infer-detr-detect \
  --checkpoint output/detrdet_hgnetv2_s_custom/checkpoint_best_regular.pth \
  --variant s \
  --source path/to/image_or_folder \
  --output_dir predictions_det \
  --conf_thresh 0.35 \
  --device cuda
```

RTMO detection:

```bash
visionhub-infer-rtmo-detect \
  --checkpoint output/rtmodet_hgnetv2_s_custom/checkpoint_best_regular.pth \
  --variant s \
  --source path/to/image_or_folder \
  --output_dir predictions_det \
  --conf_thresh 0.35 \
  --device cuda
```

RTMDet detection:

```bash
visionhub-infer-rtmdet \
  --checkpoint output/rtmdet_hgnetv2_s_custom/checkpoint_best_regular.pth \
  --variant s \
  --source path/to/image_or_folder \
  --output_dir predictions_det \
  --conf_thresh 0.35 \
  --device cuda
```

Detection notes:

- the config module can be inferred from checkpoint metadata or `--variant`
- `--nms_thresh` can override config-time NMS for dense detectors
- outputs are bbox-only and include class names from checkpoint metadata

### Native classifier PyTorch inference

```bash
visionhub-infer-classifier \
  --checkpoint ./output/classifiers/efficientnet_v2_s_run1 \
  --source ./data/classifier_dataset/val \
  --output predictions_classifier \
  --topk 5
```

This reads the native checkpoint metadata, reconstructs the architecture, and reports top-k class predictions.

### ONNX inference

Generic pose ONNX inference:

```bash
visionhub-infer-onnx \
  --onnx onnx_engines/detrpose_hgnetv2_s_custom.onnx \
  --input examples/example1.jpg
```

Classifier ONNX inference:

```bash
visionhub-infer-classifier-onnx \
  --onnx onnx_engines/efficientnet_v2_s_run1.onnx \
  --source ./data/classifier_dataset/val
```

### TensorRT inference

```bash
visionhub-infer-trt \
  --trt trt_engines/detrpose_hgnetv2_s_custom.engine \
  --input examples/example1.jpg
```

For maximum throughput:

```bash
visionhub-infer-trt \
  --trt trt_engines/detrpose_hgnetv2_s_custom.engine \
  --input examples/example1.jpg \
  --benchmark \
  --duplicate-iou-thresh 0
```

## Export And Deployment

### Pose ONNX export

DETRPose:

```bash
visionhub-export-onnx \
  --check \
  -c configs/detrpose/detrpose_hgnetv2_s_custom.py \
  -r output/detrpose_hgnetv2_s_custom/checkpoint_best_regular.pth
```

RTMOPose:

```bash
visionhub-export-rtmo-onnx \
  --check \
  -c configs/rtmopose/rtmopose_hgnetv2_s_custom.py \
  -r output/rtmopose_hgnetv2_s_custom/checkpoint_best_regular.pth
```

RTMDet-Pose:

```bash
visionhub-export-rtmdetpose-onnx \
  --check \
  -c configs/rtmdetpose/rtmdetpose_hgnetv2_s_custom.py \
  -r output/rtmdetpose_hgnetv2_s_custom/checkpoint_best_regular.pth
```

These exports are written under `onnx_engines/` and include sidecar metadata JSON when available.

### Detection ONNX export

DETR detection:

```bash
visionhub-export-detr-detect-onnx \
  --check \
  --variant s \
  --resume output/detrdet_hgnetv2_s_custom/checkpoint_best_regular.pth
```

RTMO detection:

```bash
visionhub-export-rtmo-detect-onnx \
  --check \
  --variant s \
  --resume output/rtmodet_hgnetv2_s_custom/checkpoint_best_regular.pth
```

RTMDet detection:

```bash
visionhub-export-rtmdet-onnx \
  --check \
  --variant s \
  --resume output/rtmdet_hgnetv2_s_custom/checkpoint_best_regular.pth
```

These exports write detection ONNX models that return `scores`, `labels`, and `boxes`, plus sidecar class-mapping metadata when available.

### Classifier ONNX export

```bash
visionhub-export-classifier-onnx \
  --checkpoint ./output/classifiers/efficientnet_v2_s_run1 \
  --check \
  --simplify
```

The classifier exporter writes:

- an ONNX model
- a metadata JSON file with class names
- image size and normalization settings

### TensorRT conversion

Single-model conversion with `trtexec`:

```bash
/opt/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```

Folder-based conversion helpers:

```bash
visionhub-export-tensorrt
visionhub-export-tensorrt-windows
```

In practice, the typical deployment path is:

1. train a model
2. export to ONNX
3. convert ONNX to TensorRT
4. run `visionhub-infer-trt` or your own runtime integration

## Benchmarking

### FLOPs, MACs, and parameter count

```bash
visionhub-benchmark-info \
  --config_file configs/rtmdetpose/rtmdetpose_hgnetv2_s_custom.py
```

### PyTorch latency

```bash
visionhub-benchmark-torch \
  -c configs/detrpose/detrpose_hgnetv2_s_custom.py \
  --resume output/detrpose_hgnetv2_s_custom/checkpoint_best_regular.pth \
  --infer_dir ./data/coco/val/images
```

### TensorRT latency

```bash
visionhub-benchmark-trt \
  --infer_dir ./data/coco/val/images \
  --engine_dir trt_engines
```

## Notebooks

This repo includes Colab-oriented notebooks under `notebooks/`:

- `notebooks/DETRPose_Colab_Training_v2.ipynb`
- `notebooks/RTMOPose_Colab_Training_v2.ipynb`
- `notebooks/RTMDetPose_Colab_Training_v2.ipynb`
- `notebooks/Native_Classifier_Colab_Training_v1.ipynb`

The three pose notebooks now also support detection runs through a `MODEL_TASK = 'pose' | 'detect'` switch, so each architecture notebook can train its detection counterpart without hardcoded dataset paths.

## Project Notes

- VisionHub is still an actively evolving toolkit; some parts are more polished than others.
- The pose and detection workflows are config-driven and closely tied to the repository’s lazy-config structure.
- The classifier workflow is intentionally more standalone and CLI-oriented.
- Some scripts assume optional dependencies are installed, especially ONNX, TensorRT, OpenCV, and CUDA-specific packages.
- TensorRT support is strongest through the provided Docker environment.

If you want more detail on RTMDet-specific export and inference behavior, see [RTMDET_INFERENCE_GUIDE.md](RTMDET_INFERENCE_GUIDE.md).

## Citation

If you use DETRPose or the transformer-based branch in this repo, please cite the original paper:

```bibtex
@misc{janampa2025detrpose,
  title={DETRPose: Real-time end-to-end transformer model for multi-person pose estimation},
  author={Sebastian Janampa and Marios Pattichis},
  year={2025},
  eprint={2506.13027},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2506.13027}
}
```

If you use the custom-dataset extensions or classifier workflow in this repository, also link back to this repo in your work.

## Acknowledgement

This repository began as a fork and extension of [DETRPose](https://github.com/SebastianJanampa/DETRPose) by Sebastian Janampa and Marios Pattichis from The University of New Mexico.

The broader implementation also builds on ideas and components from:

- [DEIM](https://github.com/Intellindust-AI-Lab/DEIM/tree/main)
- [D-FINE](https://github.com/Peterande/D-FINE)
- [Detectron2](https://github.com/facebookresearch/detectron2/tree/main)
- [GroupPose](https://github.com/Michel-liu/GroupPose/tree/main)
- torchvision classification backbones from the PyTorch ecosystem

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
