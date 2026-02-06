"""Keras MLP model builder.

Builds a small MLP suitable for embedded deployment:
- Input: N features
- Hidden layers: configurable (default [64, 32])
- Output: 7 classes with softmax activation
"""

import logging
from typing import Any

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


class ModelBuilder:
    """Builds Keras MLP models for classification.
    
    Attributes:
        num_features: Number of input features.
        num_classes: Number of output classes.
        hidden_layers: List of hidden layer sizes.
        dropout_rate: Dropout rate for regularization.
    """
    
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_layers: list[int] = None,
        dropout_rate: float = 0.2,
        activation: str = "relu"
    ) -> None:
        """Initialize the model builder.
        
        Args:
            num_features: Number of input features.
            num_classes: Number of output classes.
            hidden_layers: List of hidden layer sizes. Default: [64, 32].
            dropout_rate: Dropout rate between layers.
            activation: Activation function for hidden layers.
        """
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers or [64, 32]
        self.dropout_rate = dropout_rate
        self.activation = activation
    
    def build(self) -> keras.Model:
        """Build and compile the Keras model.
        
        Returns:
            Compiled Keras Sequential model.
        """
        logger.info(
            f"Building MLP: input={self.num_features}, "
            f"hidden={self.hidden_layers}, output={self.num_classes}"
        )
        
        model = keras.Sequential(name="crowdmind_mlp")
        
        model.add(layers.InputLayer(input_shape=(self.num_features,)))
        
        for i, units in enumerate(self.hidden_layers):
            model.add(layers.Dense(
                units,
                activation=self.activation,
                name=f"hidden_{i+1}"
            ))
            if self.dropout_rate > 0:
                model.add(layers.Dropout(self.dropout_rate, name=f"dropout_{i+1}"))
        
        model.add(layers.Dense(
            self.num_classes,
            activation="softmax",
            name="output"
        ))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        logger.info(f"Model compiled. Total parameters: {model.count_params()}")
        
        return model
    
    def get_architecture_info(self) -> dict[str, Any]:
        """Get model architecture information for metadata.
        
        Returns:
            Dictionary with architecture details.
        """
        return {
            "type": "MLP",
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "hidden_layers": self.hidden_layers,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "output_activation": "softmax",
        }
