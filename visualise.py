# visualise.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def visualise_attention_heatmap(model, image_tensor, caption, save_path=None):
    """
    Shows which patches the model attends to for a given (image, caption) pair.
    This is the most compelling visual for your README.
    """
    model.eval()
    m = model.module if hasattr(model, "module") else model

    with torch.no_grad():
        outputs = m(image_tensor.unsqueeze(0), caption.unsqueeze(0))

    # patch_scores: (1, 196) → reshape to (14, 14)
    scores = outputs["patch_scores"][0].cpu().numpy()
    heatmap = scores.reshape(14, 14)

    # Normalise to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Visualise
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    axes[0].imshow(img_np)
    axes[0].set_title("Original image", fontsize=12)
    axes[0].axis("off")

    # Attention heatmap
    axes[1].imshow(img_np)
    axes[1].imshow(
        np.kron(heatmap, np.ones((16, 16))),  # upscale 14×14 → 224×224
        alpha=0.5, cmap="hot", vmin=0, vmax=1
    )
    axes[1].set_title("Attention heatmap", fontsize=12)
    axes[1].axis("off")

    # Masked patches (top-k highlighted)
    mask_idx = outputs["mask_indices"][0].cpu().numpy() - 1  # back to 0..195
    mask_2d = np.zeros(196)
    mask_2d[mask_idx] = 1
    mask_2d = mask_2d.reshape(14, 14)

    axes[2].imshow(img_np)
    axes[2].imshow(
        np.kron(mask_2d, np.ones((16, 16))),
        alpha=0.6, cmap="Blues", vmin=0, vmax=1
    )
    axes[2].set_title(f"Masked patches (k={len(mask_idx)})", fontsize=12)
    axes[2].axis("off")

    plt.suptitle(f'"{caption}"', fontsize=11, style="italic", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig

from sklearn.manifold import TSNE

def plot_embedding_space(img_feats, txt_feats, labels, n_samples=500):
    """
    Visualise alignment between image and text embeddings.
    Good pairs should cluster together.
    """
    idx = np.random.choice(len(img_feats), n_samples, replace=False)
    combined = torch.cat([img_feats[idx], txt_feats[idx]]).numpy()

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(combined)

    img_2d = emb_2d[:n_samples]
    txt_2d = emb_2d[n_samples:]

    plt.figure(figsize=(10, 8))
    plt.scatter(img_2d[:, 0], img_2d[:, 1], c="steelblue", alpha=0.5, s=10, label="Image")
    plt.scatter(txt_2d[:, 0], txt_2d[:, 1], c="coral",     alpha=0.5, s=10, label="Text")
    # Draw lines between matched pairs
    for i in range(min(50, n_samples)):
        plt.plot([img_2d[i,0], txt_2d[i,0]], [img_2d[i,1], txt_2d[i,1]],
                 "gray", alpha=0.15, linewidth=0.5)
    plt.legend()
    plt.title("Image–Text embedding alignment (t-SNE)")
    plt.savefig("tsne_alignment.png", dpi=150, bbox_inches="tight")
    plt.show()