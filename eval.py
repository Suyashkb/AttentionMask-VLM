# eval.py
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def extract_features(model, loader, device):
    """Extract all image and text embeddings from a dataset."""
    all_img, all_txt = [], []

    m = model.module if hasattr(model, "module") else model  # unwrap DataParallel

    for images, tokens in tqdm(loader, desc="Extracting features"):
        images = images.to(device)
        tokens = tokens.to(device)
        img_emb = m.encode_image(images)   # normalised (B, 512)
        txt_emb = m.encode_text(tokens)    # normalised (B, 512)
        all_img.append(img_emb.cpu())
        all_txt.append(txt_emb.cpu())

    return torch.cat(all_img), torch.cat(all_txt)


def recall_at_k(sim_matrix, ks=(1, 5, 10)):
    """
    Args:
        sim_matrix: (N_img, N_txt) similarity scores
    Returns:
        dict of R@k for image→text retrieval
    """
    N = sim_matrix.shape[0]
    results = {}

    for k in ks:
        # For each image (row), check if correct text is in top-k
        top_k = sim_matrix.topk(k, dim=1).indices    # (N, k)
        correct = (top_k == torch.arange(N).unsqueeze(1)).any(dim=1)
        results[f"i2t_r{k}"] = correct.float().mean().item() * 100

    # Text→Image
    sim_T = sim_matrix.T
    for k in ks:
        top_k = sim_T.topk(k, dim=1).indices
        correct = (top_k == torch.arange(N).unsqueeze(1)).any(dim=1)
        results[f"t2i_r{k}"] = correct.float().mean().item() * 100

    return results


@torch.no_grad()
def evaluate(model, loader, device):
    """Full retrieval evaluation. Call after each epoch."""
    img_feats, txt_feats = extract_features(model, loader, device)

    # Cosine similarity matrix (N, N)
    sim = img_feats @ txt_feats.T

    metrics = recall_at_k(sim)

    # Mean Recall (primary metric for paper)
    metrics["mean_recall"] = np.mean([
        metrics["i2t_r1"], metrics["i2t_r5"], metrics["i2t_r10"],
        metrics["t2i_r1"], metrics["t2i_r5"], metrics["t2i_r10"]
    ])

    return metrics