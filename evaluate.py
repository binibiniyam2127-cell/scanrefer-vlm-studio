import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import numpy as np
import torch
import plotly.graph_objects as go
from models.vlm_grounding import load_vlm_model
from utils.data_loader import ScanReferDataset
from utils.box_utils import calculate_3d_iou, make_bbox_lines
from utils.metrics import evaluate_predictions, format_metrics_summary

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 3D Visual Grounding Model on ScanRefer")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(BASE_DIR, "checkpoints", "vlm_3d_grounding_model.pth"))
    parser.add_argument("--annotations", type=str, default=os.path.join(BASE_DIR, "data", "scanrefer", "ScanRefer_filtered_val.json"))
    parser.add_argument("--scans_dir", type=str, default=os.path.join(BASE_DIR, "data", "scannet", "scans"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(BASE_DIR, "visualizations"))
    parser.add_argument("--save_html", action="store_true", default=True, help="Save interactive 3D HTML visualizations")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n--- Starting 3D Visual Grounding Evaluation ---")
    print(f"Checkpoint:   {args.checkpoint}")
    print(f"Annotations:  {args.annotations}")
    print(f"Scans Dir:    {args.scans_dir}")

    # 1. Load Model
    model, tokenizer, device = load_vlm_model(args.checkpoint)

    # 2. Load Dataset
    dataset = ScanReferDataset(
        annotations_path=args.annotations,
        scans_dir=args.scans_dir,
        num_points=4096
    )
    print(f"Dataset contains {len(dataset)} evaluation samples.\n")

    pred_boxes = []
    gt_boxes = []
    results_table = []

    # 3. Evaluation Loop
    for idx in range(len(dataset)):
        sample = dataset[idx]
        xyz = sample["xyz"].unsqueeze(0).to(device)
        rgb = sample["rgb"].unsqueeze(0).to(device)
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        mask = sample["attention_mask"].unsqueeze(0).to(device)
        gt_box = sample["gt_bbox"].numpy()

        with torch.no_grad():
            c_logits, pred_b = model(xyz, rgb, input_ids, mask)
            best_idx = torch.argmax(c_logits[0])
            pred_box = pred_b[0, best_idx].cpu().numpy()
            confidence = torch.sigmoid(c_logits[0, best_idx]).item()

        iou = calculate_3d_iou(pred_box, gt_box)
        pred_boxes.append(pred_box)
        gt_boxes.append(gt_box)

        scene_id = sample["scene_id"]
        obj_name = sample["object_name"]
        desc = sample["description"]

        results_table.append({
            "idx": idx,
            "scene": scene_id,
            "object": obj_name,
            "desc": desc,
            "iou": iou,
            "conf": confidence
        })

        print(f"[{idx+1}/{len(dataset)}] Scene: {scene_id} | Object: {obj_name:<10} | Conf: {confidence*100:5.1f}% | 3D IoU: {iou*100:5.1f}%")
        print(f"      Query: \"{desc}\"")

        # 4. Optional: Save 3D HTML Visualization
        if args.save_html:
            pts_np = sample["xyz"].numpy()
            rgb_np = sample["rgb"].numpy()

            fig = go.Figure(data=[
                go.Scatter3d(
                    x=pts_np[:, 0], y=pts_np[:, 1], z=pts_np[:, 2],
                    mode="markers",
                    marker=dict(size=2.5, color=rgb_np, opacity=0.8),
                    name="Room Points"
                ),
                make_bbox_lines(pred_box, color="red", name=f"Pred ({confidence*100:.1f}%)"),
                make_bbox_lines(gt_box, color="green", name="Ground Truth")
            ])
            fig.update_layout(
                title=f"Query: \"{desc}\"<br>3D IoU: {iou*100:.1f}% | Conf: {confidence*100:.1f}%",
                scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"),
                margin=dict(l=0, r=0, b=0, t=50),
                height=650
            )
            html_filename = f"{scene_id}_{obj_name}_eval_{idx}.html"
            html_path = os.path.join(args.output_dir, html_filename)
            fig.write_html(html_path)
            print(f"      Saved visualization -> {html_path}")

    # 5. Compute Metrics
    metrics = evaluate_predictions(pred_boxes, gt_boxes)
    print("\n" + format_metrics_summary(metrics))


if __name__ == "__main__":
    main()

