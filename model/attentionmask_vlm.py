import torch
import torch.nn as nn
from .encoders import FrozenImageEncoder, FrozenTextEncoder
from .gate import CrossAttentionGate, MaskSelector
from .mim_head import MIMHead


class AttentionMaskVLM(nn.Module):
    """
    Full AttentionMask-VLM model.

    Frozen:    ViT-B/16 image encoder + CLIP text encoder  (~149M params)
    Trainable: CrossAttentionGate + MaskSelector + MIMHead (~15M params)
    """
    def __init__(self, cfg):
        super().__init__()
        # Frozen encoders
        self.image_encoder = FrozenImageEncoder(
            cfg.model.clip_model, cfg.model.clip_pretrained
        )
        self.text_encoder = FrozenTextEncoder(
            cfg.model.clip_model, cfg.model.clip_pretrained
        )

        # Trainable components
        self.gate = CrossAttentionGate(
            embed_dim=cfg.model.embed_dim,
            num_heads=cfg.model.num_heads
        )
        self.mask_selector = MaskSelector(
            embed_dim=cfg.model.embed_dim,
            mask_ratio=cfg.model.mask_ratio
        )
        self.mim_head = MIMHead(
            embed_dim=cfg.model.embed_dim
        )

        # Temperature parameter (learnable, same as CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def encode_image(self, images):
        """Used at eval time — returns normalised CLS embedding."""
        with torch.no_grad():
            _, cls_embed = self.image_encoder(images)
        return cls_embed / cls_embed.norm(dim=-1, keepdim=True)

    def encode_text(self, tokens):
        """Used at eval time — returns normalised CLS embedding."""
        with torch.no_grad():
            _, cls_embed = self.text_encoder(tokens)
        return cls_embed / cls_embed.norm(dim=-1, keepdim=True)

    def forward(self, images, tokens):
        """
        Full forward pass for training.

        Returns dict with everything needed for loss computation.
        """
        # Step 1: Frozen encoder forward passes
        with torch.no_grad():
            patch_tokens, img_cls = self.image_encoder(images)  # (B,197,768), (B,512)
            text_tokens, txt_cls  = self.text_encoder(tokens)   # (B,77,768),  (B,512)

        # Step 2: Cross-attention gate — identifies high-consensus patches
        attended_patches, patch_scores = self.gate(patch_tokens, text_tokens)
        # attended_patches: (B, 197, 768)
        # patch_scores:     (B, 196) — attention weight per patch

        # Step 3: Mask selector — replace top-k with [MASK] token
        masked_tokens, mask_indices, original_feats = self.mask_selector(
            attended_patches, patch_scores
        )
        # masked_tokens:  (B, 197, 768)
        # mask_indices:   (B, k)
        # original_feats: (B, k, 768) — ground truth for MIM

        # Step 4: MIM head — reconstruct masked patches
        pred_feats = self.mim_head(masked_tokens, mask_indices)
        # pred_feats: (B, k, 768)

        return {
            "img_cls":       img_cls,           # (B, 512) for contrastive
            "txt_cls":       txt_cls,            # (B, 512) for contrastive
            "pred_feats":    pred_feats,         # (B, k, 768) MIM predictions
            "original_feats":original_feats,     # (B, k, 768) MIM targets
            "patch_scores":  patch_scores,       # (B, 196) for visualisation
            "mask_indices":  mask_indices,       # (B, k) for visualisation
            "logit_scale":   self.logit_scale.exp()
        }

    def count_trainable_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {trainable/1e6:.1f}M / Total: {total/1e6:.1f}M")