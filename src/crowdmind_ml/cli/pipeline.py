"""Main training pipeline orchestration."""

import logging
from pathlib import Path
from typing import Any

from ..api import CrowdMindAPIClient
from ..data import DatasetLoader, DataValidator, DataSplitter
from ..features import FeaturePreprocessor
from ..model import ModelBuilder, ModelTrainer
from ..tflite import TFLiteQuantizer
from ..artifacts import MetadataGenerator, DatasetHasher, RunManager

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates the complete training pipeline.
    
    Steps:
    1. Download dataset from API
    2. Load and validate dataset
    3. Split data deterministically
    4. Preprocess features (fit on train only)
    5. Build and train MLP model
    6. Quantize to TFLite int8
    7. Generate metadata
    8. Upload model to API
    """
    
    def __init__(
        self,
        dataset_id: str,
        config: dict[str, Any],
        api_base: str,
        api_token: str,
        runs_dir: Path
    ) -> None:
        """Initialize the training pipeline.
        
        Args:
            dataset_id: UUID of the dataset to train on.
            config: Training configuration dictionary.
            api_base: Base URL of the CrowdMind API.
            api_token: Authentication token for API.
            runs_dir: Directory for training run outputs.
        """
        self.dataset_id = dataset_id
        self.config = config
        self.random_seed = config.get("random_seed", 42)
        
        self.api_client = CrowdMindAPIClient(api_base, api_token)
        self.run_manager = RunManager(runs_dir)
        
        self._run_dir = None
    
    def run(self) -> Path:
        """Execute the complete training pipeline.
        
        Returns:
            Path to the run folder containing all outputs.
        
        Raises:
            Exception: If any pipeline step fails.
        """
        logger.info("=" * 60)
        logger.info("CROWDMIND ML FACTORY - Training Pipeline")
        logger.info("=" * 60)
        logger.info(f"Dataset ID: {self.dataset_id}")
        logger.info(f"Random seed: {self.random_seed}")
        
        self._run_dir = self.run_manager.create_run(self.dataset_id)
        logger.info(f"Run folder: {self._run_dir}")
        
        try:
            csv_path, schema_path = self._step_download_dataset()
            df, schema = self._step_load_dataset(csv_path, schema_path)
            self._step_validate_dataset(df, schema)
            data_split = self._step_split_dataset(df, schema)
            processed_data = self._step_preprocess(data_split)
            model, arch_info = self._step_build_model(processed_data)
            trained_model, metrics = self._step_train_model(model, processed_data)
            tflite_path = self._step_quantize(trained_model, processed_data)
            meta_path = self._step_generate_metadata(
                csv_path, processed_data, arch_info, metrics
            )
            self._step_upload_model(tflite_path, meta_path)
            
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Run folder: {self._run_dir}")
            logger.info(f"Model: {tflite_path}")
            logger.info(f"Metadata: {meta_path}")
            logger.info("=" * 60)
            
            return self._run_dir
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _step_download_dataset(self) -> tuple[Path, Path]:
        """Step 1: Download dataset from API."""
        logger.info("-" * 40)
        logger.info("STEP 1: Downloading dataset from API")
        
        data_dir = self.run_manager.get_data_dir(self._run_dir)
        csv_path, schema_path = self.api_client.download_dataset(
            self.dataset_id, data_dir
        )
        
        return csv_path, schema_path
    
    def _step_load_dataset(self, csv_path: Path, schema_path: Path):
        """Step 2: Load dataset using pandas."""
        logger.info("-" * 40)
        logger.info("STEP 2: Loading dataset")
        
        loader = DatasetLoader(csv_path, schema_path)
        return loader.load()
    
    def _step_validate_dataset(self, df, schema) -> None:
        """Step 3: Validate dataset against schema."""
        logger.info("-" * 40)
        logger.info("STEP 3: Validating dataset")
        
        validator = DataValidator(schema)
        validator.validate(df)
    
    def _step_split_dataset(self, df, schema):
        """Step 4: Split dataset deterministically."""
        logger.info("-" * 40)
        logger.info("STEP 4: Splitting dataset")
        
        data_config = self.config.get("data", {})
        splitter = DataSplitter(
            train_ratio=data_config.get("train_ratio", 0.7),
            val_ratio=data_config.get("val_ratio", 0.15),
            test_ratio=data_config.get("test_ratio", 0.15),
            random_seed=self.random_seed
        )
        
        return splitter.split(
            df,
            feature_columns=schema.get("features", []),
            label_column=schema.get("label_column", "label")
        )
    
    def _step_preprocess(self, data_split):
        """Step 5: Preprocess features (fit on train only)."""
        logger.info("-" * 40)
        logger.info("STEP 5: Preprocessing features")
        
        preprocessor = FeaturePreprocessor()
        processed_data = preprocessor.fit_transform(data_split)
        
        self._preprocessor = preprocessor
        
        return processed_data
    
    def _step_build_model(self, processed_data):
        """Step 6: Build Keras MLP model."""
        logger.info("-" * 40)
        logger.info("STEP 6: Building model")
        
        model_config = self.config.get("model", {})
        builder = ModelBuilder(
            num_features=processed_data.num_features,
            num_classes=processed_data.num_classes,
            hidden_layers=model_config.get("hidden_layers", [64, 32]),
            dropout_rate=model_config.get("dropout_rate", 0.2),
            activation=model_config.get("activation", "relu")
        )
        
        model = builder.build()
        arch_info = builder.get_architecture_info()
        
        return model, arch_info
    
    def _step_train_model(self, model, processed_data):
        """Step 7: Train the model."""
        logger.info("-" * 40)
        logger.info("STEP 7: Training model")
        
        training_config = self.config.get("training", {})
        trainer = ModelTrainer(
            epochs=training_config.get("epochs", 100),
            batch_size=training_config.get("batch_size", 32),
            patience=training_config.get("patience", 10),
            random_seed=self.random_seed
        )
        
        checkpoint_dir = self.run_manager.get_checkpoints_dir(self._run_dir)
        return trainer.train(model, processed_data, checkpoint_dir)
    
    def _step_quantize(self, model, processed_data) -> Path:
        """Step 8: Quantize to TFLite int8."""
        logger.info("-" * 40)
        logger.info("STEP 8: Quantizing to TFLite int8")
        
        quant_config = self.config.get("quantization", {})
        quantizer = TFLiteQuantizer(
            num_calibration_samples=quant_config.get("calibration_samples", 100)
        )
        
        self._quantizer = quantizer
        
        output_dir = self.run_manager.get_output_dir(self._run_dir)
        tflite_path = output_dir / "model.tflite"
        
        return quantizer.quantize(
            model,
            processed_data.X_train,
            tflite_path
        )
    
    def _step_generate_metadata(
        self, csv_path, processed_data, arch_info, metrics
    ) -> Path:
        """Step 9: Generate meta.json."""
        logger.info("-" * 40)
        logger.info("STEP 9: Generating metadata")
        
        dataset_hash = DatasetHasher.hash_file(csv_path)
        
        output_dir = self.run_manager.get_output_dir(self._run_dir)
        meta_path = output_dir / "meta.json"
        
        generator = MetadataGenerator()
        return generator.generate(
            dataset_id=self.dataset_id,
            dataset_hash=dataset_hash,
            num_features=processed_data.num_features,
            label_mapping=self._preprocessor.label_mapping,
            preprocessing_params=processed_data.params,
            architecture_info=arch_info,
            training_metrics=metrics,
            quantization_info=self._quantizer.get_quantization_info(),
            config=self.config,
            output_path=meta_path
        )
    
    def _step_upload_model(self, tflite_path: Path, meta_path: Path) -> None:
        """Step 10: Upload model to API."""
        logger.info("-" * 40)
        logger.info("STEP 10: Uploading model to API")
        
        self.api_client.upload_model(
            self.dataset_id,
            tflite_path,
            meta_path
        )
