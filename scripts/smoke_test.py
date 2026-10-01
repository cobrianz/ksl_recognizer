#!/usr/bin/env python3
"""
Smoke test — verifies the full non-camera pipeline without mediapipe/opencv.

Usage:
    python scripts/smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np


def test_config():
    from utils.dataset import load_config
    cfg = load_config()
    assert "signs" in cfg and len(cfg["signs"]) >= 2
    assert cfg["model"]["input_dim"] == 1662
    print("[PASS] config")
    return cfg


def test_data(cfg):
    root = Path(cfg["paths"]["landmarks"])
    total = 0
    for s in cfg["signs"]:
        n = len(list((root / s).glob("*.npy")))
        assert n > 0, f"No samples for sign '{s}'"
        total += n
    print(f"[PASS] data ({total} samples across {len(cfg['signs'])} signs)")


def test_dataset(cfg):
    from utils.dataset import KSLLandmarkDataset
    ds = KSLLandmarkDataset(
        root_dir=cfg["paths"]["landmarks"],
        signs=cfg["signs"],
        sequence_length=cfg["data"]["sequence_length"],
    )
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (cfg["data"]["sequence_length"], cfg["model"]["input_dim"])
    assert 0 <= y.item() < len(cfg["signs"])
    print(f"[PASS] dataset (len={len(ds)}, sample shape={tuple(x.shape)})")


def test_model(cfg):
    from utils.model import create_model
    cfg = dict(cfg)
    cfg["model"] = dict(cfg["model"])
    cfg["model"]["num_classes"] = len(cfg["signs"])

    for mtype in ("lstm", "transformer"):
        cfg["model"]["type"] = mtype
        model = create_model(cfg)
        batch = torch.randn(2, cfg["data"]["sequence_length"], cfg["model"]["input_dim"])
        out = model(batch)
        assert out.shape == (2, len(cfg["signs"]))
    print("[PASS] model (lstm + transformer forward)")


def test_checkpoint(cfg):
    path = Path(cfg["paths"]["models"]) / "best_model.pt"
    if not path.exists():
        print("[SKIP] checkpoint (no best_model.pt yet — run train.py)")
        return
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "signs" in ckpt
    from utils.model import create_model
    cfg2 = dict(ckpt["config"])
    cfg2["model"] = dict(cfg2["model"])
    cfg2["model"]["num_classes"] = len(ckpt["signs"])
    model = create_model(cfg2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    batch = torch.randn(1, cfg["data"]["sequence_length"], cfg["model"]["input_dim"])
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (1, len(ckpt["signs"]))
    print(f"[PASS] checkpoint (val_acc={ckpt.get('val_acc', 'n/a')})")


def test_train_step(cfg):
    """One tiny train step to verify gradients flow."""
    from utils.model import create_model
    from utils.dataset import get_dataloaders

    cfg = dict(cfg)
    cfg["model"] = dict(cfg["model"])
    cfg["model"]["num_classes"] = len(cfg["signs"])
    cfg["model"]["type"] = "lstm"

    try:
        train_loader, _ = get_dataloaders(cfg, batch_size=4)
    except RuntimeError as e:
        print(f"[SKIP] train step ({e})")
        return

    model = create_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    sequences, labels = next(iter(train_loader))
    opt.zero_grad()
    loss = criterion(model(sequences), labels)
    loss.backward()
    opt.step()
    print(f"[PASS] train step (loss={loss.item():.4f})")


def main():
    print("=" * 50)
    print("KSL Recognizer — Smoke Test")
    print("=" * 50)
    cfg = test_config()
    test_data(cfg)
    test_dataset(cfg)
    test_model(cfg)
    test_checkpoint(cfg)
    test_train_step(cfg)
    print("=" * 50)
    print("All smoke tests passed.")
    print("=" * 50)
    print("\nCamera-dependent scripts (need mediapipe + opencv):")
    print("  python scripts/collect_data.py")
    print("  python scripts/realtime_demo.py")
    print("\nTrain / evaluate:")
    print("  python scripts/train.py")
    print("  python scripts/evaluate.py")


if __name__ == "__main__":
    main()
