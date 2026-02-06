#!/usr/bin/env python3
"""Test script for CrowdMind API integration.

This script tests the complete flow:
1. Create a dataset via API
2. Generate synthetic dataset version
3. Run training pipeline
4. Upload model version to API

Usage:
    python scripts/test_api_integration.py --api-base https://staging-api.crowdmind.fr/api/v1
"""

import argparse
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crowdmind_ml.api import CrowdMindAPIClient
from crowdmind_ml.cli import load_config, TrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test CrowdMind API Integration")
    parser.add_argument(
        "--api-base",
        type=str,
        default="https://staging-api.crowdmind.fr/api/v1",
        help="Base URL of the CrowdMind API"
    )
    parser.add_argument(
        "--token",
        type=str,
        default="",
        help="API authentication token (optional)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "train.yaml",
        help="Path to training config"
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=500,
        help="Number of rows to generate for the test dataset"
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="User ID for created_by field (auto-generated if not provided)"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("CROWDMIND API INTEGRATION TEST")
    logger.info("=" * 60)
    logger.info(f"API Base: {args.api_base}")
    
    client = CrowdMindAPIClient(args.api_base, args.token)
    
    # Step 1: Health check
    logger.info("-" * 40)
    logger.info("Step 1: Health check")
    if not client.health_check():
        logger.error("API health check failed!")
        return 1
    logger.info("API is healthy")
    
    # Step 2: Create dataset
    logger.info("-" * 40)
    logger.info("Step 2: Creating test dataset")
    
    user_id = args.user_id or str(uuid.uuid4())
    
    try:
        dataset_info = client.create_dataset(
            name="test-integration-dataset",
            dataset_type="synthetic",
            created_by=user_id,
            description="Integration test dataset"
        )
        dataset_id = dataset_info["id"]
        logger.info(f"Dataset created with ID: {dataset_id}")
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        return 1
    
    # Step 3: Generate synthetic dataset version
    logger.info("-" * 40)
    logger.info(f"Step 3: Generating synthetic version with {args.n_rows} rows")
    
    schema = {
        "features": ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
        "label_column": "label",
        "labels": ["class_0", "class_1", "class_2"]
    }
    
    try:
        version_info = client.generate_synthetic_dataset(
            dataset_id=dataset_id,
            version="1.0",
            n=args.n_rows,
            seed=42,
            labels=schema["labels"]
        )
        version_id = version_info["id"]
        logger.info(f"Dataset version created: {version_id}")
    except Exception as e:
        logger.error(f"Failed to generate synthetic dataset: {e}")
        return 1
    
    # Step 4: Run training pipeline
    logger.info("-" * 40)
    logger.info("Step 4: Running training pipeline")
    
    try:
        config = load_config(args.config)
        
        pipeline = TrainingPipeline(
            dataset_version_id=version_id,
            config=config,
            api_base=args.api_base,
            api_token=args.token,
            runs_dir=Path(__file__).parent.parent / "runs",
            model_name="test-integration-model",
            schema=schema
        )
        
        run_dir = pipeline.run()
        logger.info(f"Training completed. Output: {run_dir}")
    except Exception as e:
        logger.exception(f"Training pipeline failed: {e}")
        return 1
    
    logger.info("=" * 60)
    logger.info("INTEGRATION TEST COMPLETED SUCCESSFULLY")
    logger.info(f"Dataset ID: {dataset_id}")
    logger.info(f"Dataset Version ID: {version_id}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
