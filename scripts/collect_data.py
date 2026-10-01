#!/usr/bin/env python3
"""
Data collection script for Kenyan Sign Language.

Usage:
    python scripts/collect_data.py

Controls:
    - Press a number key (0-9) or the first letter of the sign to start recording
    - Hold the sign steady / perform it while the counter runs
    - Press 'q' to quit
    - Press 's' to skip current sign
"""

import sys
from pathlib import Path

try:
    import cv2
    import mediapipe  # noqa: F401
except ImportError as e:
    print("Missing dependency for webcam scripts.")
    print("Install with:  pip install opencv-python mediapipe")
    print(f"Details: {e}")
    sys.exit(1)

import numpy as np
import time
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.landmarks import LandmarkExtractor, draw_landmarks
from utils.dataset import load_config


def main():
    config = load_config()
    signs = config["signs"]
    seq_len = config["data"]["sequence_length"]
    samples_target = config["data"]["samples_per_sign"]
    landmarks_dir = Path(config["paths"]["landmarks"])
    landmarks_dir.mkdir(parents=True, exist_ok=True)

    for sign in signs:
        (landmarks_dir / sign).mkdir(exist_ok=True)

    print("=" * 60)
    print("KSL Data Collection")
    print("=" * 60)
    print(f"Signs: {signs}")
    print(f"Sequence length: {seq_len} frames")
    print(f"Target samples per sign: {samples_target}")
    print("\nControls:")
    print("  0-9  → start recording that sign index")
    print("  q    → quit")
    print("  s    → skip / cancel current recording")
    print("=" * 60)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current_sign_idx = None
    recording = False
    frames_buffer = []
    countdown = 0

    with LandmarkExtractor(
        min_detection_confidence=config["data"]["min_detection_confidence"],
        min_tracking_confidence=config["data"]["min_tracking_confidence"],
    ) as extractor:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror
            landmarks, results = extractor.get_landmarks(frame)
            annotated = draw_landmarks(frame.copy(), results)

            # Status text
            status = "Ready - press 0-9 to record a sign"
            color = (0, 255, 0)

            if recording:
                frames_buffer.append(landmarks)
                remaining = seq_len - len(frames_buffer)
                status = f"RECORDING: {signs[current_sign_idx]}  ({remaining} frames left)"
                color = (0, 0, 255)

                if len(frames_buffer) >= seq_len:
                    # Save sequence
                    sign_name = signs[current_sign_idx]
                    sign_dir = landmarks_dir / sign_name
                    existing = list(sign_dir.glob("*.npy"))
                    sample_id = len(existing) + 1
                    out_path = sign_dir / f"sample_{sample_id:03d}.npy"
                    np.save(out_path, np.stack(frames_buffer, axis=0))
                    print(f"Saved: {out_path}")

                    frames_buffer = []
                    recording = False
                    current_sign_idx = None
                    status = f"Saved sample for {sign_name}!"
                    color = (255, 255, 0)

            # Show current counts
            counts = []
            for i, s in enumerate(signs):
                n = len(list((landmarks_dir / s).glob("*.npy")))
                counts.append(f"{i}:{s[:6]}({n})")
            count_str = " | ".join(counts)

            cv2.putText(
                annotated, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
            cv2.putText(
                annotated, count_str, (10, annotated.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
            )

            cv2.imshow("KSL Data Collection", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                recording = False
                frames_buffer = []
                current_sign_idx = None
            elif ord("0") <= key <= ord("9"):
                idx = key - ord("0")
                if idx < len(signs) and not recording:
                    current_sign_idx = idx
                    recording = True
                    frames_buffer = []
                    print(f"Started recording: {signs[idx]}")

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection finished.")


if __name__ == "__main__":
    main()
