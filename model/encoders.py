import torch
import torch.nn as nn
import open_clip


class FrozenImageEncoder(nn.Module):
    """
    ViT-B/16 image encoder with patch token output.
    Returns ALL 197 tokens (1 CLS + 196 patch), not just the CLS projection.
    """
    def __init__(self, clip_model_name="ViT-B-16", pretrained="openai"):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )
        self.visual = model.visual
        self._freeze()

    def _freeze(self):
        for p in self.visual.parameters():
            p.requires_grad = False

    def forward(self, images):
        """
        Args:
            images: (B, 3, 224, 224)
        Returns:
            patch_tokens: (B, 197, 768)  — all tokens including CLS at index 0
            cls_embed:    (B, 768)       — projected CLS for contrastive loss
        """
        x = self.visual.conv1(images)                    # (B, 768, 14, 14)
        x = x.reshape(x.shape[0], x.shape[1], -1)       # (B, 768, 196)
        x = x.permute(0, 2, 1)                           # (B, 196, 768)

        # prepend CLS token
        cls = self.visual.class_embedding.expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)                   # (B, 197, 768)
        x = x + self.visual.positional_embedding
        if hasattr(self.visual, "patch_dropout"):
            x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)
        x = self.visual.transformer(x)
        x = self.visual.ln_post(x)

        patch_tokens = x                                  # (B, 197, 768)
        cls_embed = x[:, 0] @ self.visual.proj           # (B, 512) projected

        return patch_tokens, cls_embed


class FrozenTextEncoder(nn.Module):
    """
    CLIP text encoder. Returns token embeddings + projected CLS.
    Uses open_clip's own encode_text pipeline with a hook to capture
    all token embeddings — avoids fragile internal transformer API calls.
    """
    def __init__(self, clip_model_name="ViT-B-16", pretrained="openai"):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )
        self.model = model
        self._freeze()

    def _freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, tokens):
        """
        Args:
            tokens: (B, 77)  — tokenised captions
        Returns:
            token_embeds: (B, 77, 768)
            cls_embed:    (B, 512)   — projected EOS token for contrastive
        """
        captured = {}

        def _hook(module, inp, out):
            captured['token_embeds'] = out  # (B, 77, 768) after ln_final

        handle = self.model.ln_final.register_forward_hook(_hook)
        cls_embed = self.model.encode_text(tokens)       # (B, 512)
        handle.remove()

        token_embeds = captured['token_embeds']          # (B, 77, 768)
        return token_embeds, cls_embed