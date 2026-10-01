#!/usr/bin/env python3
"""
Generate synthetic landmark sequences for testing the pipeline
when no webcam / real data is available.

Usage:
    python scripts/generate_synthetic_data.py
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.dataset import load_config


def generate_sign_sequence(sign_idx: int, seq_len: int = 30, dim: int = 1662) -> np.ndarray:
    """
    Create a somewhat unique but noisy sequence for each sign.
    This is only for pipeline testing — not for real research.
    """
    rng = np.random.default_rng(seed=sign_idx * 100 + 42)

    # Base pattern unique to the sign
    t = np.linspace(0, 2 * np.pi, seq_len)
    base = np.sin(t * (sign_idx + 1) * 0.5)[:, None] * 0.1

    # Add some structure in different landmark regions
    seq = rng.normal(0, 0.02, size=(seq_len, dim)).astype(np.float32)
    seq += base

    # Make hand regions (last 126 dims = 21*3*2) more distinctive
    hand_start = dim - 126
    seq[:, hand_start:] += rng.normal(
        sign_idx * 0.05, 0.03, size=(seq_len, 126)
    ).astype(np.float32)

    return seq


def main():
    config = load_config()
    signs = config["signs"]
    seq_len = config["data"]["sequence_length"]
    samples_per_sign = 40  # enough for a quick train test
    landmarks_dir = Path(config["paths"]["landmarks"])
    landmarks_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic landmark data for pipeline testing...")
    print("(Replace with real collected data for actual research)\n")

    for i, sign in enumerate(signs):
        sign_dir = landmarks_dir / str(sign)
        sign_dir.mkdir(exist_ok=True)

        for j in range(samples_per_sign):
            seq = generate_sign_sequence(i, seq_len=seq_len)
            # Add slight sample-level noise
            seq += np.random.randn(*seq.shape).astype(np.float32) * 0.01
            out_path = sign_dir / f"sample_{j+1:03d}.npy"
            np.save(out_path, seq)

        print(f"  {sign}: {samples_per_sign} samples")

    print(f"\nDone. Synthetic data written to {landmarks_dir}")
    print("You can now run: python scripts/train.py")


if __name__ == "__main__":
    main()
