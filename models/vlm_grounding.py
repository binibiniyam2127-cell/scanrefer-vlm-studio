import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"

from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class VLM3DVisualGrounding(nn.Module):
    """
    Multimodal 3D Visual Grounding Model for ScanRefer.
    Fuses 3D point cloud spatial features with language features via cross-attention
    to predict 3D bounding boxes [cx, cy, cz, dx, dy, dz] and confidence scores.
    """
    def __init__(
        self,
        text_model_name: str = "bert-base-uncased",
        point_feat_dim: int = 6,
        hidden_dim: int = 256,
        num_proposals: int = 64
    ):
        super().__init__()
        self.num_proposals = num_proposals
        self.hidden_dim = hidden_dim

        # 1. Text Encoder (BERT) & Projection
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)

        # 2. 3D Point Cloud MLP
        self.point_mlp = nn.Sequential(
            nn.Linear(point_feat_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )

        # 3. Downsampler to Proposal Tokens
        self.proposal_downsampler = nn.AdaptiveAvgPool1d(num_proposals)

        # 4. Multimodal Cross-Attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # 5. Prediction Heads
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            xyz: (B, N, 3) 3D coordinates
            rgb: (B, N, 3) normalized RGB colors in [0, 1]
            input_ids: (B, L) tokenized text IDs
            attention_mask: (B, L) text attention mask
        Returns:
            conf_logits: (B, num_proposals) confidence logits for each proposal
            pred_boxes: (B, num_proposals, 6) predicted 3D bounding boxes [cx, cy, cz, dx, dy, dz]
        """
        # Concatenate XYZ + RGB -> (B, N, 6)
        pts = torch.cat([xyz, rgb], dim=-1)

        # Point feature encoding -> (B, N, hidden_dim)
        point_tokens = self.point_mlp(pts)

        # Downsample points to object proposals -> (B, num_proposals, hidden_dim)
        point_tokens = self.proposal_downsampler(point_tokens.transpose(1, 2)).transpose(1, 2)

        # Text encoding & projection -> (B, L, hidden_dim)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feats = self.text_proj(text_outputs.last_hidden_state)

        # Cross-attention: Query=Point proposals, Key/Value=Text tokens
        fused_tokens, _ = self.cross_attention(query=point_tokens, key=text_feats, value=text_feats)
        fused_tokens = self.norm(point_tokens + fused_tokens)

        # Heads
        conf_logits = self.confidence_head(fused_tokens).squeeze(-1)  # (B, num_proposals)
        pred_boxes = self.box_head(fused_tokens)                     # (B, num_proposals, 6)

        return conf_logits, pred_boxes


def load_vlm_model(
    checkpoint_path: Optional[str] = None,
    text_model_name: str = "bert-base-uncased",
    device: Optional[torch.device] = None
) -> Tuple[VLM3DVisualGrounding, AutoTokenizer, torch.device]:
    """
    Factory function to load the model and tokenizer onto the target device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VLM3DVisualGrounding(text_model_name=text_model_name).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except Exception:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from: {checkpoint_path}")
    elif checkpoint_path:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    return model, tokenizer, device

