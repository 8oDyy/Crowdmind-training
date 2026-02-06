"""TensorFlow Lite full int8 quantization.

Produces a fully quantized int8 model compatible with:
- ESP32-S3
- Raspberry Pi
- Qt clients

Uses representative dataset for calibration.
"""

import logging
from pathlib import Path
from typing import Callable, Generator

import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


class TFLiteQuantizer:
    """Quantizes Keras models to TensorFlow Lite int8 format.
    
    Performs full integer quantization using a representative dataset
    for calibration. This ensures the model runs efficiently on
    microcontrollers and edge devices.
    """
    
    def __init__(self, num_calibration_samples: int = 100) -> None:
        """Initialize the quantizer.
        
        Args:
            num_calibration_samples: Number of samples for calibration.
        """
        self.num_calibration_samples = num_calibration_samples
    
    def quantize(
        self,
        model: keras.Model,
        representative_data: np.ndarray,
        output_path: Path
    ) -> Path:
        """Quantize model to full int8 TFLite format.
        
        Args:
            model: Trained Keras model.
            representative_data: Sample data for calibration (typically training data).
            output_path: Path to save the .tflite file.
        
        Returns:
            Path to the saved .tflite file.
        """
        logger.info("Starting TFLite int8 quantization...")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        def representative_dataset_gen() -> Generator[list[np.ndarray], None, None]:
            """Generate representative samples for quantization calibration."""
            indices = np.random.choice(
                len(representative_data),
                size=min(self.num_calibration_samples, len(representative_data)),
                replace=False
            )
            for idx in indices:
                sample = representative_data[idx:idx+1].astype(np.float32)
                yield [sample]
        
        converter.representative_dataset = representative_dataset_gen
        
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        logger.info("Converting model to TFLite...")
        tflite_model = converter.convert()
        
        output_path.write_bytes(tflite_model)
        
        model_size_kb = len(tflite_model) / 1024
        logger.info(
            f"TFLite model saved to: {output_path} "
            f"(size: {model_size_kb:.2f} KB)"
        )
        
        self._verify_quantization(output_path)
        
        return output_path
    
    def _verify_quantization(self, model_path: Path) -> None:
        """Verify the model is properly quantized.
        
        Args:
            model_path: Path to the .tflite file.
        """
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_dtype = input_details[0]["dtype"]
        output_dtype = output_details[0]["dtype"]
        
        logger.info(f"Input dtype: {input_dtype}, Output dtype: {output_dtype}")
        
        if input_dtype != np.int8 or output_dtype != np.int8:
            logger.warning(
                "Model may not be fully int8 quantized. "
                f"Input: {input_dtype}, Output: {output_dtype}"
            )
        else:
            logger.info("Full int8 quantization verified successfully.")
    
    def get_quantization_info(self) -> dict:
        """Get quantization configuration for metadata.
        
        Returns:
            Dictionary with quantization details.
        """
        return {
            "type": "int8",
            "optimization": "DEFAULT",
            "input_type": "int8",
            "output_type": "int8",
            "calibration_samples": self.num_calibration_samples,
            "target_ops": "TFLITE_BUILTINS_INT8",
        }
