"""Metadata generation for trained models."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..features.preprocessor import PreprocessingParams
from ..model.trainer import TrainingMetrics

logger = logging.getLogger(__name__)


class MetadataGenerator:
    """Generates meta.json files for trained models.
    
    The meta.json contains all information needed to:
    - Reproduce the training run
    - Deploy the model correctly
    - Track model lineage
    """
    
    def generate(
        self,
        dataset_id: str,
        dataset_hash: str,
        num_features: int,
        label_mapping: dict[int, str],
        preprocessing_params: PreprocessingParams,
        architecture_info: dict[str, Any],
        training_metrics: TrainingMetrics,
        quantization_info: dict[str, Any],
        config: dict[str, Any],
        output_path: Path
    ) -> Path:
        """Generate and save meta.json file.
        
        Args:
            dataset_id: UUID of the training dataset.
            dataset_hash: SHA256 hash of the dataset.
            num_features: Number of input features.
            label_mapping: Mapping from class index to label name.
            preprocessing_params: Fitted preprocessing parameters.
            architecture_info: Model architecture details.
            training_metrics: Training and evaluation metrics.
            quantization_info: Quantization configuration.
            config: Original training configuration.
            output_path: Path to save meta.json.
        
        Returns:
            Path to the saved meta.json file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "id": dataset_id,
                "hash": dataset_hash,
            },
            "features": {
                "count": num_features,
                "preprocessing": preprocessing_params.to_dict(),
            },
            "labels": {
                "count": len(label_mapping),
                "mapping": {str(k): v for k, v in label_mapping.items()},
            },
            "model": {
                "architecture": architecture_info,
                "metrics": training_metrics.to_dict(),
            },
            "quantization": quantization_info,
            "config": config,
        }
        
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to: {output_path}")
        
        return output_path
