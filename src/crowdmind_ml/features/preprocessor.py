"""Feature preprocessing with StandardScaler.

IMPORTANT: Scaler is fit ONLY on training data, then applied to all splits.
This prevents data leakage from validation/test sets.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

from ..data.splitter import DataSplit

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingParams:
    """Parameters from fitted preprocessors for reproducibility.
    
    Attributes:
        scaler_mean: Mean values from StandardScaler.
        scaler_scale: Scale values from StandardScaler.
        label_classes: Original label classes before encoding.
    """
    scaler_mean: list[float]
    scaler_scale: list[float]
    label_classes: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "label_classes": self.label_classes,
        }


@dataclass
class ProcessedData:
    """Container for preprocessed data splits.
    
    Attributes:
        X_train: Normalized training features.
        X_val: Normalized validation features.
        X_test: Normalized test features.
        y_train: Encoded training labels.
        y_val: Encoded validation labels.
        y_test: Encoded test labels.
        params: Preprocessing parameters for reproducibility.
        num_classes: Number of unique classes.
        num_features: Number of input features.
    """
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    params: PreprocessingParams
    num_classes: int
    num_features: int


class FeaturePreprocessor:
    """Preprocesses features and labels for model training.
    
    Applies:
    - StandardScaler normalization (fit on train only)
    - LabelEncoder for categorical labels
    """
    
    def __init__(self) -> None:
        """Initialize preprocessor with unfitted transformers."""
        self._scaler = StandardScaler()
        self._label_encoder = LabelEncoder()
        self._is_fitted = False
    
    def fit_transform(self, data_split: DataSplit) -> ProcessedData:
        """Fit on training data and transform all splits.
        
        Args:
            data_split: Raw data splits from DataSplitter.
        
        Returns:
            ProcessedData with normalized features and encoded labels.
        """
        logger.info("Fitting preprocessors on training data...")
        
        self._scaler.fit(data_split.X_train)
        logger.info(f"StandardScaler fitted. Mean shape: {self._scaler.mean_.shape}")
        
        all_labels = np.concatenate([
            data_split.y_train,
            data_split.y_val,
            data_split.y_test
        ])
        self._label_encoder.fit(all_labels)
        logger.info(f"LabelEncoder fitted. Classes: {self._label_encoder.classes_}")
        
        X_train = self._scaler.transform(data_split.X_train).astype(np.float32)
        X_val = self._scaler.transform(data_split.X_val).astype(np.float32)
        X_test = self._scaler.transform(data_split.X_test).astype(np.float32)
        
        y_train = self._label_encoder.transform(data_split.y_train)
        y_val = self._label_encoder.transform(data_split.y_val)
        y_test = self._label_encoder.transform(data_split.y_test)
        
        self._is_fitted = True
        
        params = PreprocessingParams(
            scaler_mean=self._scaler.mean_.tolist(),
            scaler_scale=self._scaler.scale_.tolist(),
            label_classes=self._label_encoder.classes_.tolist(),
        )
        
        logger.info(
            f"Preprocessing complete. "
            f"Features: {X_train.shape[1]}, Classes: {len(params.label_classes)}"
        )
        
        return ProcessedData(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            params=params,
            num_classes=len(params.label_classes),
            num_features=X_train.shape[1],
        )
    
    @property
    def label_mapping(self) -> dict[int, str]:
        """Get mapping from encoded labels to original labels."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor not fitted yet.")
        return {i: label for i, label in enumerate(self._label_encoder.classes_)}
