from typing import List, Dict, Any, Union
import numpy as np
from .box_utils import calculate_3d_iou


def evaluate_predictions(
    pred_boxes: Union[np.ndarray, List[List[float]]],
    gt_boxes: Union[np.ndarray, List[List[float]]],
    iou_thresholds: List[float] = [0.25, 0.5]
) -> Dict[str, Any]:
    """
    Computes 3D grounding evaluation metrics:
    - Acc@IoU_0.25
    - Acc@IoU_0.50
    - Mean IoU
    """
    ious = []
    for pred, gt in zip(pred_boxes, gt_boxes):
        iou = calculate_3d_iou(pred, gt)
        ious.append(iou)

    ious = np.array(ious, dtype=np.float32)
    results = {
        "num_samples": len(ious),
        "mean_iou": float(np.mean(ious)) if len(ious) > 0 else 0.0,
        "ious": ious.tolist()
    }

    for th in iou_thresholds:
        acc = float(np.mean(ious >= th)) if len(ious) > 0 else 0.0
        results[f"acc@{th:.2f}"] = acc

    return results


def format_metrics_summary(metrics: Dict[str, Any]) -> str:
    """
    Formats the evaluation results into a human-readable table string.
    """
    summary = [
        "================ 3D Visual Grounding Metrics ================",
        f"Total Samples Evaluated: {metrics.get('num_samples', 0)}",
        f"Mean 3D IoU:             {metrics.get('mean_iou', 0.0) * 100:.2f}%",
        f"Accuracy @ IoU >= 0.25:  {metrics.get('acc@0.25', 0.0) * 100:.2f}%",
        f"Accuracy @ IoU >= 0.50:  {metrics.get('acc@0.50', 0.0) * 100:.2f}%",
        "============================================================"
    ]
    return "\n".join(summary)

