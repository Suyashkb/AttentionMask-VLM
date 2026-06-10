# AttentionMask-VLM

**Attention-adversarial feature masking for vision-language fine-tuning**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-Model%20Card-yellow)](https://huggingface.co/suyashkb/attentionmask-vlm)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF)](https://www.kaggle.com/suyashkumarbhagat)

> Fine-tuning CLIP with a cross-attention consistency gate that identifies high-consensus image–text patch pairs and masks them in feature space — forcing the model to learn *why* attended regions matter, not just that they do.

---

## Overview

Standard CLIP fine-tuning maximises cosine similarity between global `[CLS]` embeddings. This works well but leaves localised semantic grounding on the table — the model never needs to understand which *parts* of an image correspond to which *parts* of the text.

AttentionMask-VLM adds a second training pressure:

1. A **cross-attention gate** identifies patches where visual and textual attention mutually agree (high-consensus regions)
2. Those patches are **masked in feature space** using a learned `[MASK]` token
3. A lightweight **MIM head** is forced to reconstruct the masked features from context
4. The combined loss — contrastive + MIM — trains the gate to find semantically meaningful regions and the encoder to understand them structurally

Only ~15M parameters are trainable on top of the frozen ~149M CLIP backbone. Trains in ~8–12 hours on Kaggle 2×T4.

This is a public scaled-down implementation previewing Stage 2 of an ongoing CVPR submission.

---

## Architecture

```
Image ──► FrozenViT-B/16 ──► patch tokens (B, 197, 768) ──┐
                                                            ├──► CrossAttentionGate ──► patch scores (B, 196)
Text  ──► FrozenBERT     ──► text tokens  (B,  77, 768) ──┘         │
                                                                      ▼
                                                             MaskSelector (top-k patches)
                                                                      │
                                                             masked tokens (B, 197, 768)
                                                                      │
                                                    ┌─────────────────┴──────────────────┐
                                                    ▼                                    ▼
                                           Contrastive loss                        MIM head
                                        (image–text alignment)              (reconstruct masked patches)
                                                    └─────────────────┬──────────────────┘
                                                                       ▼
                                                          L = λ·L_contrastive + L_MIM
```

### Trainable components (~15M params)

| Component | Role | Params |
|-----------|------|--------|
| `CrossAttentionGate` | Computes cross-attention from patch tokens (Q) to text tokens (K,V); outputs per-patch attention scores | ~9.5M |
| `MaskSelector` | Selects top-k patches by score; replaces with learned `[MASK]` embedding | ~0.8M |
| `MIMHead` | 2-layer transformer that reconstructs masked patch features from context | ~4.7M |

### Frozen backbone (~149M params)

- **Image encoder:** `openai/clip-vit-base-patch16` — ViT-B/16, outputs 197 tokens (1 CLS + 196 patches of 16×16 px)
- **Text encoder:** `openai/clip-vit-base-patch16` text transformer — outputs 77 token embeddings

---

## Results

> Results will be populated after training completes. Baseline numbers from frozen CLIP are shown for reference.

### Flickr30K retrieval

| Method | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 | Mean R |
|--------|---------|---------|----------|---------|---------|----------|--------|
| CLIP frozen (baseline) | 88.0 | 98.7 | 99.4 | 68.7 | 90.7 | 95.1 | 90.1 |
| + Contrastive fine-tune (A1) | — | — | — | — | — | — | — |
| + Gate only, no MIM (A2) | — | — | — | — | — | — | — |
| + Gate + random masking (A3) | — | — | — | — | — | — | — |
| **AttentionMask-VLM (A4, ours)** | — | — | — | — | — | — | — |

### Ablation: contribution of each component

| ID | Gate | Mask strategy | MIM | i2t R@1 | Δ vs baseline |
|----|------|---------------|-----|---------|---------------|
| A0 | — | — | — | 88.0 | — |
| A1 | — | — | — | — | — |
| A2 | ✓ | — | — | — | — |
| A3 | ✓ | random | ✓ | — | — |
| **A4** | **✓** | **top-k (ours)** | **✓** | **—** | **—** |

---

## Attention visualisation

The cross-attention gate produces interpretable heatmaps showing which patches the model identified as high-consensus:

```
[attention heatmap GIFs will be added after Week 2]
```

---

## Installation

```bash
git clone https://github.com/Suyashkb/attentionmask-vlm
cd attentionmask-vlm
pip install -r requirements.txt
```

**requirements.txt**
```
open_clip_torch>=2.20.0
transformers>=4.35.0
datasets>=2.14.0
accelerate>=0.24.0
einops>=0.7.0
wandb>=0.16.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
Pillow>=10.0.0
tqdm>=4.66.0
pyyaml>=6.0
```

### Kaggle (2×T4, recommended)

```python
# Cell 1 — install
!pip install -q open_clip_torch transformers datasets accelerate einops wandb

# Cell 2 — clone repo
!git clone https://github.com/Suyashkb/attentionmask-vlm
%cd attentionmask-vlm
```

---

## Usage

### Training

```bash
python train.py --config configs/base.yaml
```

Key config options (`configs/base.yaml`):

```yaml
model:
  clip_model: "ViT-B-16"
  mask_ratio: 0.25          # fraction of patches to mask (k/196)
  num_heads: 8              # cross-attention heads in gate

training:
  epochs: 10
  lr: 3e-4
  lambda_contrastive: 0.5   # weight on contrastive loss
  lambda_mim: 1.0           # weight on MIM loss
  batch_size: 256
  fp16: true
```

### Evaluation (Flickr30K retrieval)

```bash
python eval.py --checkpoint checkpoints/best_model.pt
```

### Inference — attention heatmap

```python
from model.attentionmask_vlm import AttentionMaskVLM
from visualise import visualise_attention_heatmap
import open_clip, torch
from PIL import Image

# Load model
cfg = load_config("configs/base.yaml")
model = AttentionMaskVLM(cfg)
ckpt = torch.load("checkpoints/best_model.pt", map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Load image + caption
_, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-16")

image = preprocess(Image.open("your_image.jpg")).unsqueeze(0)
tokens = tokenizer(["a dog running in a field"])

# Visualise
visualise_attention_heatmap(model, image[0], tokens[0])
```

---

## Repository structure

```
attentionmask-vlm/
├── configs/
│   └── base.yaml               # all hyperparameters
├── data/
│   ├── dataset.py              # MSCOCO + CC3M dataloaders
│   └── augmentations.py        # image augmentation pipeline
├── model/
│   ├── encoders.py             # frozen ViT-B/16 + BERT wrappers
│   ├── gate.py                 # CrossAttentionGate + MaskSelector
│   ├── mim_head.py             # MIM reconstruction head
│   └── attentionmask_vlm.py    # full model
├── losses.py                   # contrastive + MIM loss
├── train.py                    # training loop (DataParallel, fp16, W&B)
├── eval.py                     # retrieval evaluation (R@1/5/10)
├── visualise.py                # attention heatmaps + t-SNE
├── notebooks/
│   └── kaggle_train.ipynb      # end-to-end Kaggle notebook
└── README.md
```

---

## Design decisions

**Why feature-space masking, not pixel-space?**
Pixel-space masking (as in MAE) forces reconstruction of low-level texture. Feature-space masking forces reconstruction of semantic content — more aligned with what contrastive training cares about.

**Why freeze the encoders?**
Freezing the ~149M CLIP backbone keeps the method compute-efficient (2×T4 feasible), prevents catastrophic forgetting of CLIP's pretraining, and isolates the contribution of the gate — making ablations cleaner.

**Why top-k by attention score, not random?**
The whole hypothesis is that masking *attended* patches (high-consensus regions) is harder and more informative than random masking. Ablation A3 (random) vs A4 (top-k) tests this directly.

**Why cross-attention gate and not self-attention?**
Self-attention within the image would identify salient patches independent of the text. Cross-attention specifically finds patches the text description cares about — which is the correct signal for vision-language grounding.

---

## Related work

| Paper | Venue | Relation |
|-------|-------|----------|
| [CLIP](https://arxiv.org/abs/2103.00020) | ICML 2021 | Backbone and contrastive objective |
| [MAE](https://arxiv.org/abs/2111.06377) | CVPR 2022 | Masked image modelling inspiration |
| [SyncMask](https://arxiv.org/abs/2407.14414) | CVPR 2024 | Closest related work — synchronised masking across modalities |
| [Kaleido-BERT](https://arxiv.org/abs/2103.13557) | CVPR 2021 | Attention-guided masking for VLP |
| [BEiT-3](https://arxiv.org/abs/2208.10442) | CVPR 2023 | Joint MIM + MLM pretraining |

**Key difference from SyncMask:** SyncMask synchronises masking across modalities at the input level. This work applies masking in *feature space* after cross-attention scoring, targeting semantically attended regions rather than random or salient regions independently.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{bhagat2025attentionmaskvlm,
  title   = {AttentionMask-VLM: Attention-Adversarial Feature Masking for Vision-Language Fine-Tuning},
  author  = {Bhagat, Suyash Kumar},
  year    = {2025},
  url     = {https://github.com/Suyashkb/attentionmask-vlm}
}
```

---

## Author

**Suyash Kumar Bhagat** — ML Engineer & Researcher  
MLE Intern @ Hyperverge · IEEE ICIP 2026 · DTU'26  
[LinkedIn](https://www.linkedin.com/in/suyashkb) · [GitHub](https://github.com/Suyashkb) · [Kaggle](https://www.kaggle.com/suyashkumarbhagat)

---

## License

MIT — see [LICENSE](LICENSE) for details.