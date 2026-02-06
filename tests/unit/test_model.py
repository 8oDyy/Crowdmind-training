"""Unit tests for model building."""

import pytest
import tensorflow as tf

from crowdmind_ml.model import ModelBuilder


class TestModelBuilder:
    """Tests for ModelBuilder."""
    
    def test_build_model(self):
        """Test model builds correctly."""
        builder = ModelBuilder(
            num_features=5,
            num_classes=7,
            hidden_layers=[64, 32],
            dropout_rate=0.2
        )
        
        model = builder.build()
        
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape == (None, 5)
        assert model.output_shape == (None, 7)
    
    def test_build_custom_architecture(self):
        """Test model with custom hidden layers."""
        builder = ModelBuilder(
            num_features=10,
            num_classes=3,
            hidden_layers=[128, 64, 32],
            dropout_rate=0.3
        )
        
        model = builder.build()
        
        assert model.input_shape == (None, 10)
        assert model.output_shape == (None, 3)
    
    def test_architecture_info(self):
        """Test architecture info is correct."""
        builder = ModelBuilder(
            num_features=5,
            num_classes=7,
            hidden_layers=[64, 32],
            dropout_rate=0.2,
            activation="relu"
        )
        
        info = builder.get_architecture_info()
        
        assert info["type"] == "MLP"
        assert info["num_features"] == 5
        assert info["num_classes"] == 7
        assert info["hidden_layers"] == [64, 32]
        assert info["dropout_rate"] == 0.2
        assert info["activation"] == "relu"
        assert info["output_activation"] == "softmax"
    
    def test_model_compiles(self):
        """Test model is compiled with optimizer and loss."""
        builder = ModelBuilder(num_features=5, num_classes=7)
        model = builder.build()
        
        assert model.optimizer is not None
        assert model.loss is not None
