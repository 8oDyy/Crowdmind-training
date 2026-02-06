"""Pytest configuration and shared fixtures."""

import shutil
from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def dataset_csv(fixtures_dir: Path) -> Path:
    """Return path to test dataset CSV."""
    return fixtures_dir / "dataset.csv"


@pytest.fixture
def schema_json(fixtures_dir: Path) -> Path:
    """Return path to test schema JSON."""
    return fixtures_dir / "schema.json"


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory for tests."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def smoke_output_dir() -> Path:
    """Create output directory for smoke tests (persisted for CI artifacts)."""
    output_dir = Path(__file__).parent / "smoke" / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def test_config() -> dict:
    """Return minimal test configuration."""
    return {
        "random_seed": 42,
        "data": {
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        },
        "model": {
            "hidden_layers": [16, 8],
            "dropout_rate": 0.1,
            "activation": "relu",
        },
        "training": {
            "epochs": 2,
            "batch_size": 8,
            "patience": 2,
        },
        "quantization": {
            "calibration_samples": 10,
        },
    }
