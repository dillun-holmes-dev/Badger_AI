"""
Data loading and augmentation pipeline for Badger.

Supports:
  - COCO format datasets (via pycocotools)
  - YOLO format datasets (.txt per image)
  - Mosaic augmentation (4 images combined into 1)
  - MixUp augmentation
  - Standard augmentations (HSV, scale, flip, etc.)
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


# =============================================================================
# COCO Dataset
# =============================================================================

class COCODataset(Dataset):
    """
    COCO dataset loader.

    Reads images and converts COCO annotations to Badger format:
      - Bounding boxes: (cx, cy, w, h) normalized to [0, 1]
      - Class labels: integer class IDs

    Uses pycocotools for parsing COCO JSON annotations.
    """

    def __init__(self, root, ann_file, img_size=640, augment=False, mosaic=False):
        """
        Args:
            root: path to image directory
            ann_file: path to COCO annotation JSON
            img_size: target image size (square)
            augment: whether to apply augmentations
            mosaic: whether to use mosaic augmentation
        """
        self.root = root
        self.img_size = img_size
        self.augment = augment
        self.mosaic = mosaic

        # Lazy import — only needed when using COCO
        try:
            from pycocotools.coco import COCO
            self.coco = COCO(ann_file)
        except ImportError:
            raise ImportError(
                "pycocotools is required for COCO datasets. "
                "Install with: pip install pycocotools"
            )

        self.img_ids = list(self.coco.imgs.keys())
        self.cat_ids = self.coco.getCatIds()
        self.class_map = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

    def __len__(self):
        return len(self.img_ids)

    def _load_image(self, img_id):
        """Load an image by COCO image ID."""
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_annotations(self, img_id):
        """Load annotations for an image in Badger format."""
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        classes = []

        img_info = self.coco.loadImgs(img_id)[0]
        img_h, img_w = img_info['height'], img_info['width']

        for ann in anns:
            if ann.get('iscrowd', False):
                continue  # Skip crowd annotations (too messy)

            # COCO bbox: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # Convert to normalized (cx, cy, w, h)
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            boxes.append([cx, cy, nw, nh])
            classes.append(self.class_map[ann['category_id']])

        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int64)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # Load image and annotations
        img = self._load_image(img_id)
        boxes, classes = self._load_annotations(img_id)

        if self.mosaic and np.random.random() < 0.5:
            # Mosaic: combine 4 random images
            # Pick 3 additional random images
            extra_ids = np.random.choice(self.img_ids, 3, replace=False)
            imgs = [self._load_image(eid) for eid in extra_ids]
            img, boxes, classes = self._mosaic_augment(
                [img] + imgs, [boxes] + [self._load_annotations(eid)[0] for eid in extra_ids],
                [classes] + [self._load_annotations(eid)[1] for eid in extra_ids]
            )

        # Resize and pad to square
        img, boxes = self._resize_and_pad(img, boxes, self.img_size)

        # Augment
        if self.augment:
            img, boxes = self._augment(img, boxes)

        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Build target tensor: [num_objects, 6] = [batch_idx, cls, x, y, w, h]
        if len(boxes) > 0:
            targets = np.concatenate([
                np.zeros((len(boxes), 1)),  # batch_idx (will be set by collate)
                classes.reshape(-1, 1),
                boxes
            ], axis=1)
            targets = torch.from_numpy(targets).float()
        else:
            targets = torch.zeros((0, 6), dtype=torch.float32)

        return img, targets

    def _resize_and_pad(self, img, boxes, target_size):
        """
        Resize image maintaining aspect ratio and pad to square.

        YOLO convention: resize the longest side to target_size, pad the rest.
        """
        h, w = img.shape[:2]
        scale = target_size / max(h, w)

        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize
        img_resized = cv2.resize(img, (new_w, new_h))

        # Pad to square
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        img_padded = cv2.copyMakeBorder(
            img_resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Adjust boxes
        if len(boxes) > 0:
            boxes[:, 0] = (boxes[:, 0] * new_w + pad_left) / target_size
            boxes[:, 1] = (boxes[:, 1] * new_h + pad_top) / target_size
            boxes[:, 2] = boxes[:, 2] * new_w / target_size
            boxes[:, 3] = boxes[:, 3] * new_h / target_size

        return img_padded, boxes

    def _augment(self, img, boxes):
        """Apply HSV and geometric augmentations."""
        # HSV augmentation
        img = self._hsv_augment(img)

        # Random horizontal flip
        if np.random.random() < 0.5:
            img = np.fliplr(img)
            if len(boxes) > 0:
                boxes[:, 0] = 1 - boxes[:, 0]

        return img, boxes

    def _hsv_augment(self, img, h_gain=0.015, s_gain=0.7, v_gain=0.4):
        """Random HSV shift (helps with lighting invariance)."""
        r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        img_hsv[..., 0] = (img_hsv[..., 0] * r[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] * r[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] * r[2], 0, 255)
        return cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _mosaic_augment(self, imgs, boxes_list, classes_list, output_size=640):
        """
        Mosaic augmentation: stitch 4 images into 1.

        This is a key YOLOX/YOLOv5 innovation — it combines 4 images into a
        single training sample, which:
          - Increases object diversity per image
          - Helps detect small objects (they appear in context)
          - Acts as strong regularization
        """
        mosaic_img = np.full((output_size * 2, output_size * 2, 3), 114, dtype=np.uint8)

        # Random center point for the mosaic
        cx = int(np.random.uniform(output_size // 2, output_size * 3 // 2))
        cy = int(np.random.uniform(output_size // 2, output_size * 3 // 2))

        all_boxes = []
        all_classes = []

        for i, (img, boxes, classes) in enumerate(zip(imgs, boxes_list, classes_list)):
            h, w = img.shape[:2]

            # Determine placement
            if i == 0:  # top-left
                x1, y1 = max(cx - w, 0), max(cy - h, 0)
                x2, y2 = cx, cy
            elif i == 1:  # top-right
                x1, y1 = cx, max(cy - h, 0)
                x2, y2 = min(cx + w, output_size * 2), cy
            elif i == 2:  # bottom-left
                x1, y1 = max(cx - w, 0), cy
                x2, y2 = cx, min(cy + h, output_size * 2)
            else:  # bottom-right
                x1, y1 = cx, cy
                x2, y2 = min(cx + w, output_size * 2), min(cy + h, output_size * 2)

            # Place image
            img_h, img_w = y2 - y1, x2 - x1
            img_resized = cv2.resize(img, (img_w, img_h))
            mosaic_img[y1:y2, x1:x2] = img_resized

            # Adjust boxes
            if len(boxes) > 0:
                boxes = boxes.copy()
                boxes[:, 0] = (boxes[:, 0] * img_w + x1) / (output_size * 2)
                boxes[:, 1] = (boxes[:, 1] * img_h + y1) / (output_size * 2)
                boxes[:, 2] = boxes[:, 2] * img_w / (output_size * 2)
                boxes[:, 3] = boxes[:, 3] * img_h / (output_size * 2)
                all_boxes.append(boxes)
                all_classes.append(classes)

        # Resize mosaic back to target size
        mosaic_img = cv2.resize(mosaic_img, (output_size, output_size))

        if all_boxes:
            all_boxes = np.concatenate(all_boxes, axis=0)
            all_classes = np.concatenate(all_classes, axis=0)
            # Clip boxes to [0, 1]
            all_boxes = np.clip(all_boxes, 0, 1)
        else:
            all_boxes = np.zeros((0, 4), dtype=np.float32)
            all_classes = np.zeros((0,), dtype=np.int64)

        return mosaic_img, all_boxes, all_classes


# =============================================================================
# Collate function for DataLoader
# =============================================================================

def collate_fn(batch):
    """
    Custom collate function that handles variable numbers of objects per image.

    Returns:
        images: [B, 3, H, W] stacked tensor
        targets: [total_objects, 6] — concatenated targets with batch indices
    """
    images, targets = zip(*batch)

    # Stack images
    images = torch.stack(images, dim=0)

    # Add batch index to targets and concatenate
    for i, t in enumerate(targets):
        if len(t) > 0:
            t[:, 0] = i  # Set batch index

    targets = torch.cat(targets, dim=0)

    return images, targets


# =============================================================================
# DataLoader factory
# =============================================================================

def create_dataloader(dataset_yaml, img_size=640, batch_size=16, augment=False,
                      mosaic=False, num_workers=8, shuffle=True):
    """
    Create a DataLoader from a dataset YAML config.

    Args:
        dataset_yaml: path to YAML file with dataset config
        img_size: target image size
        batch_size: batch size
        augment: apply augmentations
        mosaic: use mosaic augmentation
        num_workers: DataLoader workers

    Returns:
        torch.utils.data.DataLoader
    """
    import yaml

    with open(dataset_yaml, 'r') as f:
        config = yaml.safe_load(f)

    root = os.path.join(config['path'], config['train'] if shuffle else config['val'])
    # TODO: auto-detect annotation file
    ann_file = os.path.join(config['path'], 'annotations', 'instances_train2017.json')

    dataset = COCODataset(
        root=root,
        ann_file=ann_file,
        img_size=img_size,
        augment=augment,
        mosaic=mosaic
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True if shuffle else False
    )

    return dataloader
