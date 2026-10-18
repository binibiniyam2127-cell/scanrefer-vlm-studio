import os
import json
from typing import Dict, Any, List, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def load_scanrefer_annotations(json_path: str) -> List[Dict[str, Any]]:
    """
    Load annotations from a ScanRefer JSON file.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Annotations file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


class ScanReferDataset(Dataset):
    """
    Dataset for 3D Visual Grounding with ScanRefer annotations.
    """
    def __init__(
        self,
        annotations_path: Union[str, List[str]],
        scans_dir: str,
        num_points: int = 4096,
        tokenizer_name: str = "bert-base-uncased",
        max_seq_length: int = 32,
        augment: bool = False
    ):
        if isinstance(annotations_path, str):
            self.annotations = load_scanrefer_annotations(annotations_path)
        else:
            self.annotations = []
            for p in annotations_path:
                self.annotations.extend(load_scanrefer_annotations(p))
        self.scans_dir = scans_dir
        self.num_points = num_points
        self.max_seq_length = max_seq_length
        self.augment = augment

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._cached_scenes: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.annotations)

    def _get_scene_points(self, scene_id: str) -> np.ndarray:
        if scene_id not in self._cached_scenes:
            # Check for <scene_id>.npy
            path = os.path.join(self.scans_dir, f"{scene_id}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Scene point cloud file not found: {path}")
            self._cached_scenes[scene_id] = np.load(path)
        return self._cached_scenes[scene_id]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.annotations[idx]
        scene_id = item["scene_id"]
        description = item["description"]
        gt_box = np.array(item["bbox"], dtype=np.float32)  # [cx, cy, cz, dx, dy, dz]

        # Load point cloud
        pts = self._get_scene_points(scene_id)

        # Sample points
        if len(pts) >= self.num_points:
            indices = np.random.choice(len(pts), self.num_points, replace=False)
        else:
            indices = np.random.choice(len(pts), self.num_points, replace=True)
        pts_sampled = pts[indices]

        xyz = pts_sampled[:, :3].astype(np.float32)
        rgb = (pts_sampled[:, 3:6] / 255.0).astype(np.float32)

        # Tokenize description
        tokenized = self.tokenizer(
            description,
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        return {
            "xyz": torch.tensor(xyz, dtype=torch.float32),
            "rgb": torch.tensor(rgb, dtype=torch.float32),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "gt_bbox": torch.tensor(gt_box, dtype=torch.float32),
            "scene_id": scene_id,
            "object_id": item.get("object_id", ""),
            "object_name": item.get("object_name", ""),
            "description": description
        }

