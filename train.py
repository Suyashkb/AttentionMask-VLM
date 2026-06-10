import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import os
import argparse
import yaml
from types import SimpleNamespace

from model.attentionmask_vlm import AttentionMaskVLM
from data.datasets import build_dataloaders
from losses import total_loss
from eval import evaluate


def get_optimizer(model, cfg):
    """Only optimise trainable components."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        trainable,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.98)       # CLIP betas
    )


def get_scheduler(optimizer, cfg, steps_per_epoch):
    total_steps  = cfg.training.epochs * steps_per_epoch
    warmup_steps = cfg.training.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        # Cosine decay after warmup
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, cfg, epoch, device):
    model.train()
    total_loss_sum = contrastive_sum = mim_sum = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for step, (images, tokens, _) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=cfg.training.fp16):
            outputs = model(images, tokens)
            l_total, l_contrastive, l_mim = total_loss(outputs, cfg)

        scaler.scale(l_total).backward()

        # Gradient clipping (important for gate stability)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            cfg.training.grad_clip
        )

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Logging
        total_loss_sum    += l_total.item()
        contrastive_sum   += l_contrastive.item()
        mim_sum           += l_mim.item()

        if step % cfg.training.log_every == 0:
            pbar.set_postfix({
                "loss": f"{l_total.item():.4f}",
                "ctr":  f"{l_contrastive.item():.4f}",
                "mim":  f"{l_mim.item():.4f}",
                "lr":   f"{scheduler.get_last_lr()[0]:.2e}"
            })
            wandb.log({
                "train/loss_total":       l_total.item(),
                "train/loss_contrastive": l_contrastive.item(),
                "train/loss_mim":         l_mim.item(),
                "train/lr":               scheduler.get_last_lr()[0],
                "train/step":             epoch * len(loader) + step
            })

    n = len(loader)
    return total_loss_sum / n, contrastive_sum / n, mim_sum / n


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project=cfg.paths.wandb_project, config=_namespace_to_dict(cfg))

    model = AttentionMaskVLM(cfg).to(device)
    model.count_trainable_params()

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    train_loader, val_loader = build_dataloaders(cfg)
    optimizer  = get_optimizer(model, cfg)
    scheduler  = get_scheduler(optimizer, cfg, len(train_loader))
    scaler     = GradScaler(enabled=cfg.training.fp16)

    best_r1 = 0.0
    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    for epoch in range(1, cfg.training.epochs + 1):
        train_loss, l_ctr, l_mim = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, cfg, epoch, device
        )
        print(f"\nEpoch {epoch} | loss={train_loss:.4f} | ctr={l_ctr:.4f} | mim={l_mim:.4f}")

        if epoch % cfg.training.eval_every == 0:
            metrics = evaluate(model, val_loader, device)
            print(f"Flickr30K i2t R@1: {metrics['i2t_r1']:.2f}")
            wandb.log({**{f"val/{k}": v for k, v in metrics.items()}, "epoch": epoch})

            if metrics["i2t_r1"] > best_r1:
                best_r1 = metrics["i2t_r1"]
                save_path = os.path.join(cfg.paths.output_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "metrics": metrics
                }, save_path)
                print(f"  Saved best model (R@1={best_r1:.2f})")

    wandb.finish()


def _nested_namespace(d):
    """Recursively convert a dict to SimpleNamespace for attribute-style access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _nested_namespace(v) for k, v in d.items()})
    return d


def _namespace_to_dict(ns):
    """Recursively convert SimpleNamespace back to a plain dict (e.g. for wandb)."""
    if isinstance(ns, SimpleNamespace):
        return {k: _namespace_to_dict(v) for k, v in vars(ns).items()}
    return ns


def load_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _nested_namespace(raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)