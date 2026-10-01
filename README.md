# Kenyan Sign Language (KSL) Real-time Recognizer

A complete Python computer-vision research starter for **Kenyan Sign Language**.

Built for research: landmark-based recognition, easy experimentation, and a clear path to continuous recognition / translation.

## Features

- MediaPipe Holistic landmark extraction (pose + face + hands)
- Bidirectional LSTM (or Transformer) sequence classifier
- Webcam data collection tool
- Training + evaluation with confusion matrix
- Real-time webcam inference demo
- Synthetic data generator (for pipeline testing without a camera)

## Project Structure

```
ksl_recognizer/
├── config.yaml              # All hyperparameters
├── requirements.txt
├── data/
│   ├── raw/                 # Optional raw videos
│   ├── landmarks/           # Extracted .npy sequences (main data)
│   └── processed/
├── models/                  # Saved checkpoints
├── scripts/
│   ├── collect_data.py      # Record signs from webcam
│   ├── generate_synthetic_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── realtime_demo.py
├── utils/
│   ├── landmarks.py
│   ├── model.py
│   └── dataset.py
└── notebooks/               # For experiments
```


## Verify installation

```bash
python scripts/smoke_test.py
```

This checks config, data, model forward pass, checkpoint loading, and one training step — no camera required.

## Quick Start

### 1. Install dependencies

```bash
cd ksl_recognizer
pip install -r requirements.txt
```

### 2. (Option A) Collect real data with webcam

```bash
python scripts/collect_data.py
```

- Press `0`–`9` to start recording the corresponding sign
- Hold / perform the sign for ~1 second (30 frames)
- Press `q` to quit
- Aim for ≥30–50 samples per sign

### 2. (Option B) Generate synthetic data (pipeline test only)

```bash
python scripts/generate_synthetic_data.py
```

### 3. Train

```bash
python scripts/train.py
# or
python scripts/train.py --epochs 40 --batch_size 16
```

Best model is saved to `models/best_model.pt`.

### 4. Evaluate

```bash
python scripts/evaluate.py
```

Prints classification report and saves a confusion matrix.

### 5. Real-time demo

```bash
python scripts/realtime_demo.py
```

## Configuration

Edit `config.yaml` to change:

- List of signs
- Sequence length
- Model size (hidden_dim, layers)
- Training hyperparameters

## Research Extensions

This starter is designed so you can easily explore:

1. **Landmark vs RGB** – replace the LSTM with a 3D-CNN or SlowFast on raw frames
2. **Signer independence** – leave-one-signer-out evaluation
3. **Few-shot learning** – add prototypical networks for new signs
4. **Continuous recognition** – add CTC or Transformer decoder for sentences
5. **Bias / skin-tone robustness** – systematic augmentation and evaluation
6. **Mobile deployment** – export to ONNX / TFLite

## Datasets to Expand With

- Wanzare et al. (2024) KSL Dataset – arXiv:2410.18295
- KSL Word-based Pose Dataset (Zenodo, 2025) – privacy-preserving MediaPipe poses
- Zindi / Task Mate KSL image classification sets
- AfriSign (includes Kenyan Sign Language)

## Citation / Acknowledgement

If you use this codebase in research, please also cite the original KSL datasets you train on.

---

Built as a research starter for Kenyan Sign Language computer vision work.
