import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.vlm_grounding import VLM3DVisualGrounding
from utils.data_loader import ScanReferDataset
from utils.box_utils import calculate_3d_iou

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Train / Fine-tune 3D Visual Grounding Model")
    parser.add_argument("--train_annotations", type=str, default=os.path.join(BASE_DIR, "data", "scanrefer", "ScanRefer_filtered_train.json"))
    parser.add_argument("--scans_dir", type=str, default=os.path.join(BASE_DIR, "data", "scannet", "scans"))
    parser.add_argument("--pretrained_ckpt", type=str, default=os.path.join(BASE_DIR, "checkpoints", "vlm_3d_grounding_model.pth"))
    parser.add_argument("--all_scenes", action="store_true", help="Train jointly on both train and validation scenes")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--output_ckpt", type=str, default=os.path.join(BASE_DIR, "checkpoints", "vlm_3d_grounding_model_finetuned.pth"))
    return parser.parse_args()


def compute_loss(conf_logits, pred_boxes, gt_boxes):
    """
    Computes multi-task visual grounding loss:
    1. Positive assignment based on best matching proposal IoU
    2. BCE loss on confidence logits
    3. Smooth L1 loss on 3D box regression for the positive proposal
    """
    bce_loss_fn = nn.BCEWithLogitsLoss()
    box_loss_fn = nn.SmoothL1Loss()

    batch_size = conf_logits.shape[0]
    total_conf_loss = 0.0
    total_box_loss = 0.0

    for b in range(batch_size):
        c_l = conf_logits[b]     # (num_proposals,)
        p_b = pred_boxes[b]      # (num_proposals, 6)
        gt = gt_boxes[b]         # (6,)

        # Compute IoUs for all proposals to find best match
        ious = []
        gt_np = gt.detach().cpu().numpy()
        for p_idx in range(p_b.shape[0]):
            p_np = p_b[p_idx].detach().cpu().numpy()
            ious.append(calculate_3d_iou(p_np, gt_np))

        best_idx = int(np.argmax(ious))

        # Target confidences: 1 for best match, 0 for others
        target_conf = torch.zeros_like(c_l)
        target_conf[best_idx] = 1.0

        # Confidence BCE loss
        conf_loss = bce_loss_fn(c_l, target_conf)
        total_conf_loss += conf_loss

        # Box regression Smooth L1 loss on the best proposal
        box_loss = box_loss_fn(p_b[best_idx], gt)
        total_box_loss += box_loss

    total_loss = (total_conf_loss / batch_size) + 2.0 * (total_box_loss / batch_size)
    return total_loss, total_conf_loss.item() / batch_size, total_box_loss.item() / batch_size


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Dataset & DataLoader
    if args.all_scenes:
        ann_paths = [
            os.path.join(BASE_DIR, "data", "scanrefer", "ScanRefer_filtered_train.json"),
            os.path.join(BASE_DIR, "data", "scanrefer", "ScanRefer_filtered_val.json")
        ]
    else:
        ann_paths = args.train_annotations

    dataset = ScanReferDataset(
        annotations_path=ann_paths,
        scans_dir=args.scans_dir,
        num_points=4096
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    model = VLM3DVisualGrounding().to(device)

    # Load pretrained weights if available
    if args.pretrained_ckpt and os.path.exists(args.pretrained_ckpt):
        print(f"Loading initial weights from: {args.pretrained_ckpt}")
        try:
            state_dict = torch.load(args.pretrained_ckpt, map_location=device, weights_only=True)
        except Exception:
            state_dict = torch.load(args.pretrained_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)

    # Freeze base BERT layers to avoid overfitting on small datasets, fine-tune projection + heads
    for param in model.text_encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    print(f"Starting training for {args.epochs} epochs ({len(dataset)} samples)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            xyz = batch["xyz"].to(device)
            rgb = batch["rgb"].to(device)
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            gt_bbox = batch["gt_bbox"].to(device)

            optimizer.zero_grad()
            conf_logits, pred_boxes = model(xyz, rgb, input_ids, mask)

            loss, conf_l, box_l = compute_loss(conf_logits, pred_boxes, gt_bbox)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(dataloader), 1)
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {avg_loss:.4f} (Conf: {conf_l:.4f}, Box: {box_l:.4f})")

    os.makedirs(os.path.dirname(args.output_ckpt), exist_ok=True)
    torch.save(model.state_dict(), args.output_ckpt)
    print(f"Model saved successfully to: {args.output_ckpt}")


if __name__ == "__main__":
    train()

