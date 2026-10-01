#!/usr/bin/env python3
"""
Real-time Kenyan Sign Language recognition demo.

Usage:
    python scripts/realtime_demo.py
    python scripts/realtime_demo.py --model models/best_model.pt
"""

import argparse
import sys
from pathlib import Path
from collections import deque

try:
    import cv2
    import mediapipe  # noqa: F401
except ImportError as e:
    print("Missing dependency for webcam scripts.")
    print("Install with:  pip install opencv-python mediapipe")
    print(f"Details: {e}")
    sys.exit(1)

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.landmarks import LandmarkExtractor, draw_landmarks
from utils.model import create_model
from utils.dataset import load_config


def load_trained_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    signs = ckpt["signs"]

    config["model"]["num_classes"] = len(signs)
    model = create_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, signs, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_model.pt")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Minimum confidence to display prediction")
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Train a model first: python scripts/train.py")
        return

    model, signs, config = load_trained_model(str(model_path), device)
    seq_len = config["data"]["sequence_length"]
    print(f"Loaded model with signs: {signs}")
    print(f"Sequence length: {seq_len}")
    print("Press 'q' to quit")

    # Sliding window of landmarks
    landmark_buffer = deque(maxlen=seq_len)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prediction_text = "Waiting for sequence..."
    confidence = 0.0
    color = (200, 200, 200)

    with LandmarkExtractor(
        min_detection_confidence=config["data"]["min_detection_confidence"],
        min_tracking_confidence=config["data"]["min_tracking_confidence"],
    ) as extractor:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            landmarks, results = extractor.get_landmarks(frame)
            annotated = draw_landmarks(frame.copy(), results)

            landmark_buffer.append(landmarks)

            # Predict when buffer is full
            if len(landmark_buffer) == seq_len:
                seq = np.stack(list(landmark_buffer), axis=0)  # (T, D)
                seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1, T, D)

                with torch.no_grad():
                    logits = model(seq_tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    conf, pred_idx = torch.max(probs, dim=0)
                    confidence = conf.item()
                    pred_sign = signs[pred_idx.item()]

                if confidence >= args.threshold:
                    prediction_text = f"{pred_sign.upper()}"
                    color = (0, 255, 0)
                else:
                    prediction_text = "..."
                    color = (100, 100, 100)

            # UI overlay
            # Semi-transparent banner
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (annotated.shape[1], 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

            cv2.putText(
                annotated,
                f"Prediction: {prediction_text}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                2,
            )
            cv2.putText(
                annotated,
                f"Confidence: {confidence:.2f}  |  Buffer: {len(landmark_buffer)}/{seq_len}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

            # Show top-3 probabilities when available
            if len(landmark_buffer) == seq_len:
                top3 = torch.topk(probs, k=min(3, len(signs)))
                y = 120
                for conf_i, idx_i in zip(top3.values, top3.indices):
                    label = f"{signs[idx_i.item()]}: {conf_i.item():.2f}"
                    cv2.putText(
                        annotated, label, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 1
                    )
                    y += 25

            cv2.imshow("KSL Real-time Recognition", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
