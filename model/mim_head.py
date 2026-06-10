import torch
import torch.nn as nn


class MIMHead(nn.Module):
    """
    Predicts original patch features from masked context.
    Input: masked patch token sequence
    Output: reconstructed features at masked positions only

    Deliberately lightweight — 2 transformer layers + projection.
    We don't want this head to memorise; we want the encoder to learn.
    """
    def __init__(self, embed_dim=768, num_layers=2, num_heads=8, mlp_ratio=2.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1,
            batch_first=True,
            norm_first=True        # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(embed_dim, embed_dim)     # predict in feature space
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, masked_tokens, mask_indices):
        """
        Args:
            masked_tokens: (B, 197, 768)
            mask_indices:  (B, k)  — which positions to reconstruct
        Returns:
            pred_feats: (B, k, 768) — predicted features at masked positions
        """
        x = self.transformer(masked_tokens)              # (B, 197, 768)
        x = self.norm(x)

        # Extract predictions only at masked positions
        B, k = mask_indices.shape
        D = x.shape[-1]
        pred_feats = torch.gather(
            x,
            dim=1,
            index=mask_indices.unsqueeze(-1).expand(-1, -1, D)
        )                                                # (B, k, 768)
        pred_feats = self.proj(pred_feats)

        return pred_feats