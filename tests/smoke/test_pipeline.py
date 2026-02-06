"""Smoke test for the complete ML training pipeline.

This test runs the full pipeline with minimal data and epochs to verify:
1. TensorFlow trains without error
2. TFLite export works
3. Int8 quantization works
4. Output files are generated (model.tflite, meta.json)
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from crowdmind_ml.data import DatasetLoader, DataValidator, DataSplitter
from crowdmind_ml.features import FeaturePreprocessor
from crowdmind_ml.model import ModelBuilder, ModelTrainer
from crowdmind_ml.tflite import TFLiteQuantizer
from crowdmind_ml.artifacts import MetadataGenerator, DatasetHasher


class TestFullPipeline:
    """Smoke test for the complete training pipeline."""
    
    def test_full_pipeline_smoke(
        self,
        dataset_csv: Path,
        schema_json: Path,
        test_config: dict,
        smoke_output_dir: Path
    ):
        """Run complete pipeline with minimal settings.
        
        This test verifies the entire ML pipeline works end-to-end:
        - Data loading and validation
        - Preprocessing
        - Model training (2 epochs)
        - TFLite quantization
        - Metadata generation
        """
        loader = DatasetLoader(dataset_csv, schema_json)
        df, schema = loader.load()
        
        assert len(df) > 0, "Dataset should not be empty"
        
        validator = DataValidator(schema)
        validator.validate(df)
        
        splitter = DataSplitter(
            train_ratio=test_config["data"]["train_ratio"],
            val_ratio=test_config["data"]["val_ratio"],
            test_ratio=test_config["data"]["test_ratio"],
            random_seed=test_config["random_seed"]
        )
        data_split = splitter.split(
            df,
            feature_columns=schema["features"],
            label_column=schema["label_column"]
        )
        
        assert len(data_split.X_train) > 0, "Training set should not be empty"
        
        preprocessor = FeaturePreprocessor()
        processed = preprocessor.fit_transform(data_split)
        
        assert processed.num_features == len(schema["features"])
        assert processed.num_classes == len(schema["labels"])
        
        builder = ModelBuilder(
            num_features=processed.num_features,
            num_classes=processed.num_classes,
            hidden_layers=test_config["model"]["hidden_layers"],
            dropout_rate=test_config["model"]["dropout_rate"]
        )
        model = builder.build()
        
        assert model is not None, "Model should be built"
        assert model.input_shape == (None, processed.num_features)
        assert model.output_shape == (None, processed.num_classes)
        
        trainer = ModelTrainer(
            epochs=test_config["training"]["epochs"],
            batch_size=test_config["training"]["batch_size"],
            patience=test_config["training"]["patience"],
            random_seed=test_config["random_seed"]
        )
        
        trained_model, metrics = trainer.train(model, processed)
        
        assert metrics.train_accuracy > 0, "Training accuracy should be positive"
        assert metrics.test_accuracy >= 0, "Test accuracy should be non-negative"
        
        quantizer = TFLiteQuantizer(
            num_calibration_samples=test_config["quantization"]["calibration_samples"]
        )
        
        tflite_path = smoke_output_dir / "model.tflite"
        quantizer.quantize(trained_model, processed.X_train, tflite_path)
        
        assert tflite_path.exists(), "TFLite model should be created"
        assert tflite_path.stat().st_size > 0, "TFLite model should not be empty"
        
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        assert input_details[0]["dtype"] == np.int8, "Input should be int8"
        assert output_details[0]["dtype"] == np.int8, "Output should be int8"
        
        dataset_hash = DatasetHasher.hash_file(dataset_csv)
        
        generator = MetadataGenerator()
        meta_path = generator.generate(
            dataset_id="smoke-test-dataset",
            dataset_hash=dataset_hash,
            num_features=processed.num_features,
            label_mapping=preprocessor.label_mapping,
            preprocessing_params=processed.params,
            architecture_info=builder.get_architecture_info(),
            training_metrics=metrics,
            quantization_info=quantizer.get_quantization_info(),
            config=test_config,
            output_path=smoke_output_dir / "meta.json"
        )
        
        assert meta_path.exists(), "meta.json should be created"
        
        with open(meta_path) as f:
            meta = json.load(f)
        
        assert "dataset" in meta
        assert "features" in meta
        assert "labels" in meta
        assert "model" in meta
        assert "quantization" in meta
        assert meta["quantization"]["type"] == "int8"
        
        print(f"\n{'='*60}")
        print("SMOKE TEST PASSED")
        print(f"{'='*60}")
        print(f"Model size: {tflite_path.stat().st_size / 1024:.2f} KB")
        print(f"Test accuracy: {metrics.test_accuracy:.4f}")
        print(f"Output dir: {smoke_output_dir}")
        print(f"{'='*60}\n")


class TestTFLiteInference:
    """Test TFLite model inference."""
    
    def test_tflite_inference(
        self,
        dataset_csv: Path,
        schema_json: Path,
        test_config: dict,
        tmp_path: Path
    ):
        """Test that the quantized model can run inference."""
        loader = DatasetLoader(dataset_csv, schema_json)
        df, schema = loader.load()
        
        splitter = DataSplitter(random_seed=42)
        data_split = splitter.split(df, schema["features"], schema["label_column"])
        
        preprocessor = FeaturePreprocessor()
        processed = preprocessor.fit_transform(data_split)
        
        builder = ModelBuilder(
            num_features=processed.num_features,
            num_classes=processed.num_classes,
            hidden_layers=[16, 8]
        )
        model = builder.build()
        
        trainer = ModelTrainer(epochs=1, batch_size=8, patience=1)
        trained_model, _ = trainer.train(model, processed)
        
        quantizer = TFLiteQuantizer(num_calibration_samples=10)
        tflite_path = tmp_path / "test_model.tflite"
        quantizer.quantize(trained_model, processed.X_train, tflite_path)
        
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_scale, input_zero_point = input_details[0]["quantization"]
        sample = processed.X_test[0:1]
        sample_quantized = (sample / input_scale + input_zero_point).astype(np.int8)
        
        interpreter.set_tensor(input_details[0]["index"], sample_quantized)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]["index"])
        
        assert output.shape == (1, processed.num_classes)
        print(f"Inference output shape: {output.shape}")
