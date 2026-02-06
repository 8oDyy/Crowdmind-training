"""Dataset loading from CSV and schema files."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads dataset CSV and schema.json files.
    
    Attributes:
        csv_path: Path to the dataset CSV file.
        schema_path: Path to the schema.json file.
    """
    
    def __init__(self, csv_path: Path, schema_path: Path) -> None:
        """Initialize the dataset loader.
        
        Args:
            csv_path: Path to the dataset CSV file.
            schema_path: Path to the schema.json file.
        """
        self.csv_path = Path(csv_path)
        self.schema_path = Path(schema_path)
    
    def load(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Load the dataset and schema.
        
        Returns:
            Tuple of (DataFrame, schema_dict).
        
        Raises:
            FileNotFoundError: If files don't exist.
            ValueError: If files are malformed.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        
        logger.info(f"Loading dataset from: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        logger.info(f"Loading schema from: {self.schema_path}")
        with open(self.schema_path, "r") as f:
            schema = json.load(f)
        
        logger.info(f"Schema loaded: {list(schema.keys())}")
        
        return df, schema
    
    def get_raw_bytes(self) -> bytes:
        """Get raw bytes of the CSV file for hashing.
        
        Returns:
            Raw bytes of the CSV file.
        """
        return self.csv_path.read_bytes()
