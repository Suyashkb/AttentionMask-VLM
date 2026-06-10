import os, json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import open_clip

class COCOCaptionDataset(Dataset):
    """
    Returns (image_tensor, caption_string) pairs.
    Each image is paired with one randomly sampled caption per step.
    """
    def __init__(self, root, split="train", transform=None, tokenizer=None):
        self.root = Path(root)
        self.transform = transform
        self.tokenizer = tokenizer

        ann_file = self.root / "annotations" / f"captions_{split}2017.json"
        with open(ann_file) as f:
            data = json.load(f)

        # Build id → file path map
        self.id2path = {
            img["id"]: self.root / f"{split}2017" / img["file_name"]
            for img in data["images"]
        }

        # Each sample is (image_id, caption)
        self.samples = [
            (ann["image_id"], ann["caption"])
            for ann in data["annotations"]
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, caption = self.samples[idx]
        img = Image.open(self.id2path[image_id]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        tokens = self.tokenizer([caption])[0]  # shape: (77,)
        return img, tokens, caption


def build_dataloaders(cfg):
    _, _, preprocess_train = open_clip.create_model_and_transforms(
        cfg.model.clip_model, pretrained=cfg.model.clip_pretrained
    )
    _, _, preprocess_val = open_clip.create_model_and_transforms(
        cfg.model.clip_model, pretrained=cfg.model.clip_pretrained
    )
    tokenizer = open_clip.get_tokenizer(cfg.model.clip_model)

    train_ds = COCOCaptionDataset(
        cfg.data.data_dir, split="train",
        transform=preprocess_train, tokenizer=tokenizer
    )
    val_ds = COCOCaptionDataset(
        cfg.data.data_dir, split="val",
        transform=preprocess_val, tokenizer=tokenizer
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.data.batch_size,
        shuffle=True, num_workers=cfg.data.num_workers,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.data.batch_size,
        shuffle=False, num_workers=cfg.data.num_workers,
        pin_memory=True
    )
    return train_loader, val_loader

class Flickr30KDataset(Dataset):
    """Used only for retrieval evaluation — not training."""
    def __init__(self, root, transform=None, tokenizer=None):
        self.root = Path(root)
        self.transform = transform
        self.tokenizer = tokenizer
        # Load captions from captions.txt (one per line: img_id#cap_idx\tcaption)
        self.samples = []
        with open(self.root / "captions.txt") as f:
            next(f)  # skip header
            for line in f:
                img_name, caption = line.strip().split("\t", 1)
                self.samples.append((img_name.split("#")[0], caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, caption = self.samples[idx]
        img = Image.open(self.root / "images" / img_name).convert("RGB")
        if self.transform:
            img = self.transform(img)
        tokens = self.tokenizer([caption])[0]
        return img, tokens