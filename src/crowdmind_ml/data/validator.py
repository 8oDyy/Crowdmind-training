"""Dataset validation against schema."""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when dataset validation fails."""
    pass


class DataValidator:
    """Validates dataset against schema requirements.
    
    Performs the following checks:
    - Required columns exist
    - Label column contains valid values
    - No NaN values in critical columns
    - Data types match schema expectations
    """
    
    def __init__(self, schema: dict[str, Any]) -> None:
        """Initialize the validator with schema.
        
        Args:
            schema: Schema dictionary containing:
                - features: list of feature column names
                - label_column: name of the label column
                - labels: list of valid label values
        """
        self.schema = schema
        self.feature_columns = schema.get("features", [])
        self.label_column = schema.get("label_column", "label")
        self.valid_labels = schema.get("labels", [])
    
    def validate(self, df: pd.DataFrame) -> None:
        """Validate the dataset against the schema.
        
        Args:
            df: DataFrame to validate.
        
        Raises:
            ValidationError: If validation fails.
        """
        logger.info("Starting dataset validation...")
        
        self._validate_columns(df)
        self._validate_labels(df)
        self._validate_nan(df)
        self._validate_dtypes(df)
        
        logger.info("Dataset validation passed.")
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Check that all required columns exist."""
        required_columns = self.feature_columns + [self.label_column]
        missing = set(required_columns) - set(df.columns)
        
        if missing:
            raise ValidationError(
                f"Missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )
        
        logger.info(f"All {len(required_columns)} required columns present.")
    
    def _validate_labels(self, df: pd.DataFrame) -> None:
        """Check that label column contains only valid values."""
        if not self.valid_labels:
            logger.warning("No valid labels specified in schema, skipping label validation.")
            return
        
        actual_labels = set(df[self.label_column].unique())
        invalid_labels = actual_labels - set(self.valid_labels)
        
        if invalid_labels:
            raise ValidationError(
                f"Invalid labels found: {invalid_labels}. "
                f"Valid labels: {self.valid_labels}"
            )
        
        logger.info(f"All labels valid. Found {len(actual_labels)} unique labels.")
    
    def _validate_nan(self, df: pd.DataFrame) -> None:
        """Check for NaN values in feature and label columns."""
        columns_to_check = self.feature_columns + [self.label_column]
        nan_counts = df[columns_to_check].isna().sum()
        columns_with_nan = nan_counts[nan_counts > 0]
        
        if not columns_with_nan.empty:
            raise ValidationError(
                f"NaN values found in columns: {columns_with_nan.to_dict()}"
            )
        
        logger.info("No NaN values in critical columns.")
    
    def _validate_dtypes(self, df: pd.DataFrame) -> None:
        """Check that feature columns are numeric."""
        for col in self.feature_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValidationError(
                    f"Feature column '{col}' is not numeric. "
                    f"Dtype: {df[col].dtype}"
                )
        
        logger.info("All feature columns are numeric.")
