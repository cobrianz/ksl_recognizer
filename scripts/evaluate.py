#!/usr/bin/env python3
"""
Evaluation & confusion matrix for trained KSL model.
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.dataset import get_dataloaders, load_config
from utils.model import create_model


def simple_classification_report(y_true, y_pred, target_names):
    """Fallback report when sklearn is not installed."""
    names = list(target_names)
    print(f"{'sign':<12} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    print("-" * 55)
    overall_correct = 0
    for i, name in enumerate(names):
        tp = int(np.sum((y_pred == i) & (y_true == i)))
        fp = int(np.sum((y_pred == i) & (y_true != i)))
        fn = int(np.sum((y_pred != i) & (y_true == i)))
        support = int(np.sum(y_true == i))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        overall_correct += tp
        print(f"{name:<12} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {support:>10d}")
    acc = overall_correct / len(y_true) if len(y_true) else 0.0
    print("-" * 55)
    print(f"{'accuracy':<12} {acc:>10.3f} {'':>10} {'':>10} {len(y_true):>10d}")


def simple_confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_model.pt")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    config = ckpt["config"]
    signs = ckpt["signs"]
    config["model"]["num_classes"] = len(signs)

    model = create_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, val_loader = get_dataloaders(config)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            logits = model(sequences)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\nClassification Report:")
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print(classification_report(all_labels, all_preds, target_names=signs, digits=3))
        cm = confusion_matrix(all_labels, all_preds)
    except ImportError:
        simple_classification_report(all_labels, all_preds, signs)
        cm = simple_confusion_matrix(all_labels, all_preds, len(signs))

    plt.figure(figsize=(10, 8))
    try:
        import seaborn as sns
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=signs, yticklabels=signs,
        )
    except ImportError:
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar()
        tick_marks = np.arange(len(signs))
        plt.xticks(tick_marks, signs, rotation=45, ha="right")
        plt.yticks(tick_marks, signs)
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("KSL Confusion Matrix")
    plt.tight_layout()

    out_path = Path("models") / "confusion_matrix.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Confusion matrix saved to {out_path}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
