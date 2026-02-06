"""Unit tests for data loading, validation, and splitting."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crowdmind_ml.data import DatasetLoader, DataValidator, DataSplitter
from crowdmind_ml.data.validator import ValidationError


class TestDatasetLoader:
    """Tests for DatasetLoader."""
    
    def test_load_dataset(self, dataset_csv: Path, schema_json: Path):
        """Test loading dataset and schema."""
        loader = DatasetLoader(dataset_csv, schema_json)
        df, schema = loader.load()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert isinstance(schema, dict)
        assert "features" in schema
        assert "label_column" in schema
    
    def test_load_nonexistent_csv(self, schema_json: Path, tmp_path: Path):
        """Test loading non-existent CSV raises error."""
        fake_csv = tmp_path / "nonexistent.csv"
        loader = DatasetLoader(fake_csv, schema_json)
        
        with pytest.raises(FileNotFoundError):
            loader.load()
    
    def test_get_raw_bytes(self, dataset_csv: Path, schema_json: Path):
        """Test getting raw bytes for hashing."""
        loader = DatasetLoader(dataset_csv, schema_json)
        raw_bytes = loader.get_raw_bytes()
        
        assert isinstance(raw_bytes, bytes)
        assert len(raw_bytes) > 0


class TestDataValidator:
    """Tests for DataValidator."""
    
    def test_validate_valid_dataset(self, dataset_csv: Path, schema_json: Path):
        """Test validation passes for valid dataset."""
        loader = DatasetLoader(dataset_csv, schema_json)
        df, schema = loader.load()
        
        validator = DataValidator(schema)
        validator.validate(df)
    
    def test_validate_missing_column(self, dataset_csv: Path, schema_json: Path):
        """Test validation fails for missing column."""
        loader = DatasetLoader(dataset_csv, schema_json)
        df, schema = loader.load()
        
        df_missing = df.drop(columns=["feature_1"])
        validator = DataValidator(schema)
        
        with pytest.raises(ValidationError):
            validator.validate(df_missing)
    
    def test_validate_invalid_label(self, dataset_csv: Path, schema_json: Path):
        """Test validation fails for invalid label."""
        loader = DatasetLoader(dataset_csv, schema_json)
        df, schema = loader.load()
        
        df.loc[0, "label"] = "invalid_class"
        validator = DataValidator(schema)
        
        with pytest.raises(ValidationError):
            validator.validate(df)
    
    def test_validate_nan_values(self, dataset_csv: Path, schema_json: Path):
        """Test validation fails for NaN values."""
        loader = DatasetLoader(dataset_csv, schema_json)
        df, schema = loader.load()
        
        df.loc[0, "feature_1"] = np.nan
        validator = DataValidator(schema)
        
        with pytest.raises(ValidationError):
            validator.validate(df)


class TestDataSplitter:
    """Tests for DataSplitter."""
    
    def test_split_ratios(self, dataset_csv: Path, schema_json: Path):
        """Test split produces correct ratios."""
        loader = DatasetLoader(dataset_csv, schema_json)
        df, schema = loader.load()
        
        splitter = DataSplitter(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        
        split = splitter.split(
            df,
            feature_columns=schema["features"],
            label_column=schema["label_column"]
        )
        
        total = len(split.X_train) + len(split.X_val) + len(split.X_test)
        assert total == len(df)
        
        assert len(split.X_train) == len(split.y_train)
        assert len(split.X_val) == len(split.y_val)
        assert len(split.X_test) == len(split.y_test)
    
    def test_split_deterministic(self, dataset_csv: Path, schema_json: Path):
        """Test split is deterministic with same seed."""
        loader = DatasetLoader(dataset_csv, schema_json)
        df, schema = loader.load()
        
        splitter1 = DataSplitter(random_seed=42)
        splitter2 = DataSplitter(random_seed=42)
        
        split1 = splitter1.split(df, schema["features"], schema["label_column"])
        split2 = splitter2.split(df, schema["features"], schema["label_column"])
        
        np.testing.assert_array_equal(split1.X_train, split2.X_train)
        np.testing.assert_array_equal(split1.y_train, split2.y_train)
    
    def test_split_invalid_ratios(self):
        """Test split raises error for invalid ratios."""
        with pytest.raises(ValueError):
            DataSplitter(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)
