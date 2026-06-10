import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionGate(nn.Module):
    """
    Computes cross-attention from patch tokens (queries) to text tokens (keys).
    Returns per-patch attention scores averaged across heads.

    This is the core novel component. It identifies which image patches
    are most semantically aligned with the text description.
    """
    def __init__(self, query_dim=768, kv_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads   = num_heads
        self.head_dim    = query_dim // num_heads
        self.scale       = self.head_dim ** -0.5

        # Q from image patches (query_dim=768), K/V from text tokens (kv_dim=512)
        self.W_q = nn.Linear(query_dim, query_dim, bias=False)
        self.W_k = nn.Linear(kv_dim,    query_dim, bias=False)
        self.W_v = nn.Linear(kv_dim,    query_dim, bias=False)
        self.W_o = nn.Linear(query_dim, query_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, patch_tokens, text_tokens):
        """
        Args:
            patch_tokens: (B, 197, 768) — image patch embeddings (incl CLS)
            text_tokens:  (B, 77, 768)  — text token embeddings
        Returns:
            attended:     (B, 197, 768) — patch tokens after cross-attention
            patch_scores: (B, 196)      — per-patch attention score (excl CLS)
        """
        B, N_v, D = patch_tokens.shape
        _, N_t, _ = text_tokens.shape

        # Project to Q, K, V
        Q = self.W_q(patch_tokens)                       # (B, 197, D)
        K = self.W_k(text_tokens)                        # (B, 77, D)
        V = self.W_v(text_tokens)                        # (B, 77, D)

        # Reshape for multi-head attention
        def reshape_heads(x):
            B, N, D = x.shape
            return x.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        Q, K, V = reshape_heads(Q), reshape_heads(K), reshape_heads(V)
        # Q: (B, H, 197, d_h)  K,V: (B, H, 77, d_h)

        # Attention weights
        attn = (Q @ K.transpose(-2, -1)) * self.scale   # (B, H, 197, 77)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Aggregate
        out = attn @ V                                   # (B, H, 197, d_h)
        out = out.permute(0, 2, 1, 3).reshape(B, N_v, D)
        out = self.W_o(out)
        attended = self.norm(patch_tokens + out)         # residual

        # Per-patch score: mean over heads, mean over text tokens
        # attn shape: (B, H, 197, 77)
        # Exclude CLS token (index 0) when computing patch scores
        patch_attn = attn[:, :, 1:, :]                  # (B, H, 196, 77)
        patch_scores = patch_attn.mean(dim=1).max(dim=-1).values  # (B, 196)
        # Using max over text positions: which patches are attended to by ANY token

        return attended, patch_scores


class MaskSelector(nn.Module):
    """
    Selects top-k patches by attention score and replaces them
    with a learned [MASK] embedding in feature space.
    """
    def __init__(self, embed_dim=768, mask_ratio=0.25):
        super().__init__()
        self.mask_ratio = mask_ratio
        # Learned mask token — the only other trainable parameter
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, patch_tokens, patch_scores):
        """
        Args:
            patch_tokens: (B, 197, 768) — all patch tokens (incl CLS at 0)
            patch_scores: (B, 196)      — attention score per patch
        Returns:
            masked_tokens: (B, 197, 768) — tokens with top-k replaced by mask
            mask_indices:  (B, k)        — indices of masked patches (in 1..196)
            original_feats:(B, k, 768)   — original features at masked positions
        """
        B, N, D = patch_tokens.shape
        k = int(196 * self.mask_ratio)  # number of patches to mask

        # Top-k indices by score (highest attention = most informative = mask these)
        _, top_k_idx = patch_scores.topk(k, dim=-1)     # (B, k) — indices in 0..195
        mask_indices = top_k_idx + 1                     # shift by 1 to account for CLS

        # Save original features before masking
        original_feats = torch.gather(
            patch_tokens,
            dim=1,
            index=mask_indices.unsqueeze(-1).expand(-1, -1, D)
        )                                                # (B, k, 768)

        # Replace with learned mask token
        masked_tokens = patch_tokens.clone()
        mask_expand = self.mask_token.expand(B, k, D)
        masked_tokens.scatter_(
            1,
            mask_indices.unsqueeze(-1).expand(-1, -1, D),
            mask_expand
        )

        return masked_tokens, mask_indices, original_feats