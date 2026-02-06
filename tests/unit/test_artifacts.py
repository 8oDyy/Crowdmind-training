"""Unit tests for artifacts module."""

import json
from pathlib import Path

import pytest

from crowdmind_ml.artifacts import DatasetHasher, MetadataGenerator, RunManager
from crowdmind_ml.features.preprocessor import PreprocessingParams
from crowdmind_ml.model.trainer import TrainingMetrics


class TestDatasetHasher:
    """Tests for DatasetHasher."""
    
    def test_hash_file(self, dataset_csv: Path):
        """Test file hashing produces consistent result."""
        hash1 = DatasetHasher.hash_file(dataset_csv)
        hash2 = DatasetHasher.hash_file(dataset_csv)
        
        assert hash1 == hash2
        assert len(hash1) == 64
        assert all(c in "0123456789abcdef" for c in hash1)
    
    def test_hash_nonexistent_file(self, tmp_path: Path):
        """Test hashing non-existent file raises error."""
        fake_file = tmp_path / "nonexistent.csv"
        
        with pytest.raises(FileNotFoundError):
            DatasetHasher.hash_file(fake_file)
    
    def test_hash_bytes(self):
        """Test bytes hashing."""
        data = b"test data"
        hash_value = DatasetHasher.hash_bytes(data)
        
        assert len(hash_value) == 64


class TestRunManager:
    """Tests for RunManager."""
    
    def test_create_run(self, tmp_path: Path):
        """Test run folder creation."""
        manager = RunManager(tmp_path / "runs")
        run_dir = manager.create_run("test-dataset-id")
        
        assert run_dir.exists()
        assert (run_dir / "data").exists()
        assert (run_dir / "checkpoints").exists()
        assert (run_dir / "output").exists()
        assert (run_dir / "logs").exists()
    
    def test_list_runs(self, tmp_path: Path):
        """Test listing runs."""
        import time
        manager = RunManager(tmp_path / "runs")
        
        manager.create_run("dataset-1")
        time.sleep(1.1)  # Ensure different timestamps
        manager.create_run("dataset-2")
        
        runs = manager.list_runs()
        assert len(runs) == 2
    
    def test_cleanup_run(self, tmp_path: Path):
        """Test run cleanup."""
        manager = RunManager(tmp_path / "runs")
        run_dir = manager.create_run("test-dataset")
        
        assert run_dir.exists()
        manager.cleanup_run(run_dir)
        assert not run_dir.exists()


class TestMetadataGenerator:
    """Tests for MetadataGenerator."""
    
    def test_generate_metadata(self, tmp_path: Path):
        """Test metadata generation."""
        generator = MetadataGenerator()
        
        preprocessing_params = PreprocessingParams(
            scaler_mean=[0.5, 0.5, 0.5],
            scaler_scale=[0.2, 0.2, 0.2],
            label_classes=["class_0", "class_1", "class_2"]
        )
        
        training_metrics = TrainingMetrics(
            train_accuracy=0.95,
            val_accuracy=0.90,
            test_accuracy=0.88,
            precision=0.87,
            recall=0.86,
            f1=0.865,
            confusion_matrix=[[10, 1], [2, 12]],
            history={"loss": [0.5, 0.3], "accuracy": [0.8, 0.9]}
        )
        
        meta_path = generator.generate(
            dataset_id="test-uuid",
            dataset_hash="abc123",
            num_features=3,
            label_mapping={0: "class_0", 1: "class_1", 2: "class_2"},
            preprocessing_params=preprocessing_params,
            architecture_info={"type": "MLP", "hidden_layers": [64, 32]},
            training_metrics=training_metrics,
            quantization_info={"type": "int8"},
            config={"random_seed": 42},
            output_path=tmp_path / "meta.json"
        )
        
        assert meta_path.exists()
        
        with open(meta_path) as f:
            meta = json.load(f)
        
        assert meta["dataset"]["id"] == "test-uuid"
        assert meta["dataset"]["hash"] == "abc123"
        assert meta["features"]["count"] == 3
        assert meta["labels"]["count"] == 3
        assert meta["model"]["metrics"]["test_accuracy"] == 0.88
        assert meta["quantization"]["type"] == "int8"
