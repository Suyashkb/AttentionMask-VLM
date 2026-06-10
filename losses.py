# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def contrastive_loss(img_cls, txt_cls, logit_scale):
    """
    Symmetric CLIP-style contrastive loss.
    Both img_cls and txt_cls must be L2-normalised before calling.
    """
    img_cls = F.normalize(img_cls, dim=-1)
    txt_cls = F.normalize(txt_cls, dim=-1)

    # DataParallel gathers scalar logit_scale from N GPUs → shape (N,); reduce to scalar
    if logit_scale.dim() > 0:
        logit_scale = logit_scale.mean()

    # Similarity matrix: (B, B)
    logits = logit_scale * img_cls @ txt_cls.T

    # Symmetric cross-entropy — diagonal is the positive pair
    B = logits.shape[0]
    labels = torch.arange(B, device=logits.device)

    loss_i2t = F.cross_entropy(logits,   labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2.0


def mim_loss(pred_feats, original_feats):
    """
    L2 reconstruction loss in feature space.
    Normalise targets to prevent scale collapse.

    Args:
        pred_feats:     (B, k, 768) — MIM head predictions
        original_feats: (B, k, 768) — original patch features (detached)
    """
    # Normalise targets (treat as pseudo-labels)
    targets = F.normalize(original_feats.detach(), dim=-1)
    preds   = F.normalize(pred_feats, dim=-1)

    # Cosine similarity loss (1 - cos_sim), averaged over patches and batch
    loss = 1.0 - (preds * targets).sum(dim=-1)     # (B, k)
    return loss.mean()


def total_loss(outputs, cfg):
    """
    Combined loss: λ·L_contrastive + L_MIM
    """
    l_contrastive = contrastive_loss(
        outputs["img_cls"],
        outputs["txt_cls"],
        outputs["logit_scale"]
    )
    l_mim = mim_loss(outputs["pred_feats"], outputs["original_feats"])

    l_total = cfg.training.lambda_contrastive * l_contrastive + \
              cfg.training.lambda_mim * l_mim

    return l_total, l_contrastive, l_mim