"""Unit tests for feature preprocessing."""

from pathlib import Path

import numpy as np
import pytest

from crowdmind_ml.data import DatasetLoader, DataSplitter
from crowdmind_ml.features import FeaturePreprocessor


class TestFeaturePreprocessor:
    """Tests for FeaturePreprocessor."""
    
    def test_fit_transform(self, dataset_csv: Path, schema_json: Path):
        """Test preprocessing fit and transform."""
        loader = DatasetLoader(dataset_csv, schema_json)
        df, schema = loader.load()
        
        splitter = DataSplitter(random_seed=42)
        data_split = splitter.split(df, schema["features"], schema["label_column"])
        
        preprocessor = FeaturePreprocessor()
        processed = preprocessor.fit_transform(data_split)
        
        assert processed.X_train.shape[0] == len(data_split.X_train)
        assert processed.X_val.shape[0] == len(data_split.X_val)
        assert processed.X_test.shape[0] == len(data_split.X_test)
        
        assert processed.X_train.dtype == np.float32
    
    def test_preprocessing_params(self, dataset_csv: Path, schema_json: Path):
        """Test preprocessing parameters are captured."""
        loader = DatasetLoader(dataset_csv, schema_json)
        df, schema = loader.load()
        
        splitter = DataSplitter(random_seed=42)
        data_split = splitter.split(df, schema["features"], schema["label_column"])
        
        preprocessor = FeaturePreprocessor()
        processed = preprocessor.fit_transform(data_split)
        
        params = processed.params
        assert len(params.scaler_mean) == processed.num_features
        assert len(params.scaler_scale) == processed.num_features
        assert len(params.label_classes) == processed.num_classes
    
    def test_label_mapping(self, dataset_csv: Path, schema_json: Path):
        """Test label mapping is correct."""
        loader = DatasetLoader(dataset_csv, schema_json)
        df, schema = loader.load()
        
        splitter = DataSplitter(random_seed=42)
        data_split = splitter.split(df, schema["features"], schema["label_column"])
        
        preprocessor = FeaturePreprocessor()
        processed = preprocessor.fit_transform(data_split)
        
        label_mapping = preprocessor.label_mapping
        assert len(label_mapping) == 7
        assert all(isinstance(k, int) for k in label_mapping.keys())
        assert all(isinstance(v, str) for v in label_mapping.values())
    
    def test_normalized_data_stats(self, dataset_csv: Path, schema_json: Path):
        """Test normalized training data has expected statistics."""
        loader = DatasetLoader(dataset_csv, schema_json)
        df, schema = loader.load()
        
        splitter = DataSplitter(random_seed=42)
        data_split = splitter.split(df, schema["features"], schema["label_column"])
        
        preprocessor = FeaturePreprocessor()
        processed = preprocessor.fit_transform(data_split)
        
        train_mean = np.mean(processed.X_train, axis=0)
        train_std = np.std(processed.X_train, axis=0)
        
        np.testing.assert_array_almost_equal(train_mean, np.zeros(5), decimal=1)
        np.testing.assert_array_almost_equal(train_std, np.ones(5), decimal=1)
