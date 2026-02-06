"""Configuration loading from YAML files."""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load training configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
    
    Returns:
        Configuration dictionary.
    
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is malformed.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading configuration from: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    config = _apply_defaults(config)
    _validate_config(config)
    
    logger.info(f"Configuration loaded successfully")
    
    return config


def _apply_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Apply default values for missing configuration keys.
    
    Args:
        config: Raw configuration dictionary.
    
    Returns:
        Configuration with defaults applied.
    """
    defaults = {
        "random_seed": 42,
        "data": {
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        },
        "model": {
            "hidden_layers": [64, 32],
            "dropout_rate": 0.2,
            "activation": "relu",
        },
        "training": {
            "epochs": 100,
            "batch_size": 32,
            "patience": 10,
        },
        "quantization": {
            "calibration_samples": 100,
        },
    }
    
    def deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge override into base."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    return deep_merge(defaults, config or {})


def _validate_config(config: dict[str, Any]) -> None:
    """Validate configuration values.
    
    Args:
        config: Configuration dictionary.
    
    Raises:
        ValueError: If configuration is invalid.
    """
    data_config = config.get("data", {})
    total_ratio = (
        data_config.get("train_ratio", 0) +
        data_config.get("val_ratio", 0) +
        data_config.get("test_ratio", 0)
    )
    
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(
            f"Data split ratios must sum to 1.0, got {total_ratio}"
        )
    
    training_config = config.get("training", {})
    if training_config.get("epochs", 0) <= 0:
        raise ValueError("epochs must be positive")
    if training_config.get("batch_size", 0) <= 0:
        raise ValueError("batch_size must be positive")
    
    logger.info("Configuration validation passed")
