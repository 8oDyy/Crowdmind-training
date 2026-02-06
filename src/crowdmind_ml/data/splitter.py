"""Deterministic dataset splitting."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """Container for train/validation/test splits.
    
    Attributes:
        X_train: Training features.
        X_val: Validation features.
        X_test: Test features.
        y_train: Training labels.
        y_val: Validation labels.
        y_test: Test labels.
    """
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


class DataSplitter:
    """Deterministic dataset splitter with fixed seed.
    
    Splits data into train/validation/test sets with reproducible results.
    """
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> None:
        """Initialize the splitter.
        
        Args:
            train_ratio: Fraction of data for training.
            val_ratio: Fraction of data for validation.
            test_ratio: Fraction of data for testing.
            random_seed: Random seed for reproducibility.
        
        Raises:
            ValueError: If ratios don't sum to 1.0.
        """
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"Ratios must sum to 1.0, got {total:.4f}"
            )
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
    
    def split(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        label_column: str
    ) -> DataSplit:
        """Split the dataset into train/validation/test sets.
        
        Args:
            df: DataFrame containing features and labels.
            feature_columns: List of feature column names.
            label_column: Name of the label column.
        
        Returns:
            DataSplit containing all splits.
        """
        logger.info(
            f"Splitting dataset with ratios: "
            f"train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio}"
        )
        
        X = df[feature_columns].values.astype(np.float32)
        y = df[label_column].values
        
        val_test_ratio = self.val_ratio + self.test_ratio
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=val_test_ratio,
            random_state=self.random_seed,
            stratify=y
        )
        
        test_ratio_adjusted = self.test_ratio / val_test_ratio
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=test_ratio_adjusted,
            random_state=self.random_seed,
            stratify=y_temp
        )
        
        logger.info(
            f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )
        
        return DataSplit(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test
        )
