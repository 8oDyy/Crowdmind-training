# CrowdMind ML Factory

External Python training pipeline for the CrowdMind project. This repository is **strictly separated** from the backend API and handles all machine learning operations.

## Overview

The ML Factory downloads datasets from the CrowdMind API, trains an MLP classifier, quantizes it to TFLite int8 format, and uploads the trained model back to the API.

**Target platforms:** ESP32-S3, Raspberry Pi, Qt clients

## Project Structure

```
crowdmind-ml-factory/
├── src/crowdmind_ml/
│   ├── api/              # HTTP client for API communication
│   ├── data/             # Dataset loading, validation, splitting
│   ├── features/         # Preprocessing (StandardScaler, LabelEncoder)
│   ├── model/            # Keras MLP builder and trainer
│   ├── tflite/           # TFLite int8 quantization
│   ├── artifacts/        # Metadata generation, hashing, run management
│   └── cli/              # Configuration loading and pipeline orchestration
├── configs/
│   └── train.yaml        # Training hyperparameters
├── scripts/
│   └── train_and_upload.py   # Main entry point
├── runs/                 # Training run outputs (timestamped folders)
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11
- TensorFlow (CPU only)
- See `requirements.txt` for full dependencies

## Installation

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the complete training pipeline with a single command:

```bash
python scripts/train_and_upload.py \
    --dataset-id <uuid> \
    --config configs/train.yaml \
    --api-base https://api.crowdmind.example.com \
    --token <your_api_token>
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--dataset-id` | Yes | UUID of the dataset to train on |
| `--config` | Yes | Path to YAML configuration file |
| `--api-base` | Yes | Base URL of the CrowdMind API |
| `--token` | Yes | API authentication token |
| `--runs-dir` | No | Output directory (default: `./runs`) |
| `--log-level` | No | Logging level (default: `INFO`) |

## Pipeline Steps

1. **Download** - Fetch dataset (CSV + schema.json) from API
2. **Load** - Parse CSV with pandas
3. **Validate** - Check columns, labels, NaN values
4. **Split** - Deterministic train/val/test split (seeded)
5. **Preprocess** - StandardScaler fit on train, transform all
6. **Build** - Create Keras MLP [64, 32] → 7 classes
7. **Train** - Train with early stopping
8. **Evaluate** - Accuracy, precision, recall, F1, confusion matrix
9. **Quantize** - Full int8 TFLite with representative dataset
10. **Export** - Save model.tflite + meta.json
11. **Upload** - Send artifacts to API

## Configuration

Edit `configs/train.yaml` to customize:

```yaml
random_seed: 42

data:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15

model:
  hidden_layers: [64, 32]
  dropout_rate: 0.2
  activation: "relu"

training:
  epochs: 100
  batch_size: 32
  patience: 10

quantization:
  calibration_samples: 100
```

## Output Structure

Each training run creates a timestamped folder:

```
runs/run_20240115_143022_abc12345/
├── data/
│   ├── dataset.csv
│   └── schema.json
├── checkpoints/
│   └── best_model.keras
├── output/
│   ├── model.tflite
│   └── meta.json
└── logs/
```

## meta.json Format

```json
{
  "version": "1.0.0",
  "timestamp": "2024-01-15T14:30:22.123456+00:00",
  "dataset": {
    "id": "uuid",
    "hash": "sha256..."
  },
  "features": {
    "count": 10,
    "preprocessing": {
      "scaler_mean": [...],
      "scaler_scale": [...],
      "label_classes": [...]
    }
  },
  "labels": {
    "count": 7,
    "mapping": {"0": "class_a", ...}
  },
  "model": {
    "architecture": {...},
    "metrics": {
      "test_accuracy": 0.95,
      "precision": 0.94,
      "recall": 0.95,
      "f1": 0.94,
      "confusion_matrix": [...]
    }
  },
  "quantization": {
    "type": "int8",
    "input_type": "int8",
    "output_type": "int8"
  }
}
```

## API Endpoints Expected

The training pipeline expects these API endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/datasets/{id}/download` | Download dataset CSV |
| GET | `/datasets/{id}/schema` | Get dataset schema JSON |
| POST | `/datasets/{id}/models` | Upload trained model |
| GET | `/health` | Health check |

## License

Proprietary - BTS CIEL Project
