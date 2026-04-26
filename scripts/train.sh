#!/bin/bash
set -e

echo "=========================================="
echo "Efficient Inference CV — Training Pipeline"
echo "=========================================="
echo ""

#step 1: preprocess data (skip if processed batches already exist)
if ls data/processed/train_data_batch_*.pt > /dev/null 2>&1; then
  echo "[1/3] Preprocessed data found — skipping."
else
  echo "[1/3] Preprocessing CIFAR-10 data..."
  python3 scripts/preprocess_data.py
  echo "Preprocessing complete."
fi
echo ""

#step 2: train
echo "[2/3] Training MobileNetV2..."
python3 -m src.training.train
echo ""

#step 3: export
echo "[3/3] Exporting to TorchScript and ONNX..."
python3 -m scripts.export_model
echo ""

echo "=========================================="
echo "Training pipeline complete."
echo "  Trained model: models/trained/mobilenetv2.pth"
echo "  TorchScript:   models/exported/model.pt"
echo "  ONNX:          models/exported/model.onnx"
echo "=========================================="
