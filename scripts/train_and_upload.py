#!/usr/bin/env python3
"""CrowdMind ML Factory - Main Training Script.

This script executes the complete training pipeline:
1. Downloads dataset version from API
2. Validates and preprocesses data
3. Trains MLP model
4. Quantizes to TFLite int8
5. Uploads model version to API

Usage:
    python scripts/train_and_upload.py \
        --dataset-version-id <uuid> \
        --config configs/train.yaml \
        --api-base <api_url> \
        --schema-file schema.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crowdmind_ml.cli import load_config, TrainingPipeline


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the training pipeline."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CrowdMind ML Factory - Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--dataset-version-id",
        type=str,
        required=True,
        help="UUID of the dataset version to train on"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the training configuration YAML file"
    )
    
    parser.add_argument(
        "--api-base",
        type=str,
        required=True,
        help="Base URL of the CrowdMind API"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default="",
        help="Authentication token for the API (optional)"
    )
    
    parser.add_argument(
        "--schema-file",
        type=Path,
        required=True,
        help="Path to the schema JSON file (features, label_column, labels)"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name for the model to create (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).parent.parent / "runs",
        help="Directory for training run outputs (default: ./runs)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point for the training pipeline.
    
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args()
    
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("CROWDMIND ML FACTORY")
    logger.info("=" * 60)
    logger.info(f"Dataset Version ID: {args.dataset_version_id}")
    logger.info(f"Config: {args.config}")
    logger.info(f"API Base: {args.api_base}")
    logger.info(f"Schema: {args.schema_file}")
    logger.info(f"Runs Dir: {args.runs_dir}")
    
    try:
        config = load_config(args.config)
        
        with open(args.schema_file, "r") as f:
            schema = json.load(f)
        
        pipeline = TrainingPipeline(
            dataset_version_id=args.dataset_version_id,
            config=config,
            api_base=args.api_base,
            api_token=args.token,
            runs_dir=args.runs_dir,
            model_name=args.model_name,
            schema=schema
        )
        
        run_dir = pipeline.run()
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"Output: {run_dir}")
        logger.info("=" * 60)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
