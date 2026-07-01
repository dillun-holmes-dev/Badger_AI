"""
Utility functions for handling class mappings in DETRPose inference.
This allows models to output actual class names instead of integer IDs.
"""
import json
import os
import torch
from pathlib import Path
from typing import Any, Dict, Tuple


def get_class_name(class_id, class_mappings):
    """
    Get class name from class ID using baked-in class mappings.
    
    Args:
        class_id: Integer class ID
        class_mappings: Dictionary mapping class IDs to class names
        
    Returns:
        Class name string or "class_{id}" if mapping not found
    """
    if class_mappings and class_id in class_mappings:
        return class_mappings[class_id]
    return f"class_{class_id}"


def load_class_mappings_from_checkpoint(checkpoint_path):
    """
    Load class mappings from a PyTorch checkpoint file.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        
    Returns:
        Dictionary mapping class IDs to class names, or empty dict if not found
    """
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return {}
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'class_mappings' in checkpoint:
            class_mappings = checkpoint['class_mappings']
            print(f"\n{'='*60}")
            print("Loaded class mappings from checkpoint:")
            for class_id, class_name in sorted(class_mappings.items()):
                print(f"  ID {class_id}: {class_name}")
            print(f"{'='*60}\n")
            return class_mappings
        else:
            print("\nWarning: No class mappings found in checkpoint. Predictions will show numeric IDs.\n")
            return {}
    except Exception as e:
        print(f"Error loading class mappings from checkpoint: {e}")
        return {}


def save_class_mappings_json(class_mappings, output_path, skeleton_connections=None, model_metadata=None):
    """
    Save class mappings (and optionally skeleton connections) to a JSON file.
    
    Args:
        class_mappings: Dictionary mapping class IDs to class names
        output_path: Path where to save the JSON file
        skeleton_connections: Optional dict mapping class IDs to list of [a, b] pairs
        model_metadata: Optional dict of sidecar metadata such as image_size/family/variant
    """
    try:
        payload = {}
        if class_mappings:
            payload['class_mappings'] = {str(k): v for k, v in class_mappings.items()}
        if skeleton_connections:
            # Keys must be strings for JSON
            payload['skeleton_connections'] = {
                str(k): v for k, v in skeleton_connections.items()
            }
        if isinstance(model_metadata, dict) and model_metadata:
            payload['model_metadata'] = model_metadata
        with open(output_path, 'w') as f:
            json.dump(payload, f, indent=2)
        
        print(f"Saved model metadata to: {output_path}")
        if skeleton_connections:
            print(f"  Includes skeleton connections for {len(skeleton_connections)} class(es).")
    except Exception as e:
        print(f"Error saving metadata to JSON: {e}")


def load_model_sidecar_json(json_path: str) -> Tuple[Dict[int, Any], Dict[int, Any], Dict[str, Any]]:
    """
    Load class mappings, skeleton connections, and sidecar model metadata.
    
    Args:
        json_path: Path to the metadata JSON file
        
    Returns:
        Tuple of (class_mappings, skeleton_connections, model_metadata).
        Returns ({}, {}, {}) on failure.
    """
    if not os.path.exists(json_path):
        print(f"Warning: Metadata JSON not found at {json_path}")
        return {}, {}, {}
    
    try:
        with open(json_path, 'r') as f:
            payload = json.load(f)
        
        # Support both old format (flat dict of class_id->name)
        # and new format ({class_mappings: {...}, skeleton_connections: {...}})
        if 'class_mappings' in payload or 'skeleton_connections' in payload:
            raw_mappings = payload.get('class_mappings', {})
            raw_skeletons = payload.get('skeleton_connections', {})
            model_metadata = payload.get('model_metadata', {})
        else:
            # Legacy: entire file is a flat class_id->name mapping
            raw_mappings = payload
            raw_skeletons = {}
            model_metadata = {}

        class_mappings = {int(k): v for k, v in raw_mappings.items()}
        skeleton_connections = {int(k): v for k, v in raw_skeletons.items()}
        model_metadata = model_metadata if isinstance(model_metadata, dict) else {}

        print(f"\n{'='*60}")
        print("Loaded class mappings from JSON:")
        for class_id, class_name in sorted(class_mappings.items()):
            print(f"  ID {class_id}: {class_name}")
        if skeleton_connections:
            print(f"Loaded skeleton connections for {len(skeleton_connections)} class(es).")
        if model_metadata:
            print(f"Loaded model metadata keys: {sorted(model_metadata.keys())}")
        print(f"{'='*60}\n")
        
        return class_mappings, skeleton_connections, model_metadata
    except Exception as e:
        print(f"Error loading metadata from JSON: {e}")
        return {}, {}, {}


def load_class_mappings_from_json(json_path):
    class_mappings, skeleton_connections, _ = load_model_sidecar_json(json_path)
    return class_mappings, skeleton_connections


def load_model_metadata_from_json(json_path: str) -> Dict[str, Any]:
    _, _, model_metadata = load_model_sidecar_json(json_path)
    return model_metadata


def _candidate_metadata_json_paths(model_path):
    model_dir = Path(model_path).parent
    model_name = Path(model_path).stem
    return [
        model_dir / f"{model_name}_class_mappings.json",
        model_dir / "class_mappings.json",
        model_dir / f"{model_name}.json",
    ]


def find_class_mappings_json(model_path):
    """
    Try to find a metadata JSON file next to the model file.
    
    Args:
        model_path: Path to the model file (.onnx, .engine, etc.)
        
    Returns:
        Tuple of (class_mappings, skeleton_connections), both with integer keys.
        Returns ({}, {}) if not found.
    """
    for json_path in _candidate_metadata_json_paths(model_path):
        if json_path.exists():
            return load_class_mappings_from_json(str(json_path))
    
    print(f"Warning: Could not find class_mappings.json near {model_path}")
    return {}, {}


def find_model_metadata_json(model_path) -> Dict[str, Any]:
    for json_path in _candidate_metadata_json_paths(model_path):
        if json_path.exists():
            return load_model_metadata_from_json(str(json_path))
    return {}
def print_detections(labels, scores, class_mappings, max_display=10):
    """
    Print detection results with class names.
    
    Args:
        labels: Array of class IDs
        scores: Array of confidence scores
        class_mappings: Dictionary mapping class IDs to class names
        max_display: Maximum number of detections to display
    """
    if len(labels) == 0:
        print("No detections above threshold")
        return
    
    print(f"\nDetections ({len(labels)} total):")
    for i, (label_id, score) in enumerate(zip(labels[:max_display], scores[:max_display])):
        class_name = get_class_name(int(label_id), class_mappings)
        print(f"  [{i+1}] {class_name} (ID: {int(label_id)}, Score: {score:.3f})")
    
    if len(labels) > max_display:
        print(f"  ... and {len(labels) - max_display} more")
