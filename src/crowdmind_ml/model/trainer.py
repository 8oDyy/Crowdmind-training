"""Model training with evaluation metrics."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from ..features.preprocessor import ProcessedData

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training and evaluation metrics.
    
    Attributes:
        train_accuracy: Final training accuracy.
        val_accuracy: Final validation accuracy.
        test_accuracy: Test set accuracy.
        precision: Weighted precision score.
        recall: Weighted recall score.
        f1: Weighted F1 score.
        confusion_matrix: Confusion matrix as nested list.
        history: Training history dictionary.
    """
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    history: dict[str, list[float]]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "test_accuracy": self.test_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "confusion_matrix": self.confusion_matrix,
        }


class ModelTrainer:
    """Trains Keras models with early stopping and evaluation.
    
    Attributes:
        epochs: Maximum number of training epochs.
        batch_size: Training batch size.
        patience: Early stopping patience.
        random_seed: Random seed for reproducibility.
    """
    
    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        random_seed: int = 42
    ) -> None:
        """Initialize the trainer.
        
        Args:
            epochs: Maximum number of training epochs.
            batch_size: Training batch size.
            patience: Early stopping patience (epochs without improvement).
            random_seed: Random seed for reproducibility.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_seed = random_seed
        
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
    
    def train(
        self,
        model: keras.Model,
        data: ProcessedData,
        checkpoint_dir: Optional[Path] = None
    ) -> tuple[keras.Model, TrainingMetrics]:
        """Train the model and evaluate on test set.
        
        Args:
            model: Compiled Keras model.
            data: Preprocessed data splits.
            checkpoint_dir: Optional directory for model checkpoints.
        
        Returns:
            Tuple of (trained_model, metrics).
        """
        logger.info(
            f"Starting training: epochs={self.epochs}, "
            f"batch_size={self.batch_size}, patience={self.patience}"
        )
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    filepath=str(checkpoint_dir / "best_model.keras"),
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=1
                )
            )
        
        history = model.fit(
            data.X_train,
            data.y_train,
            validation_data=(data.X_val, data.y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        metrics = self._evaluate(model, data, history)
        
        logger.info(
            f"Training complete. Test accuracy: {metrics.test_accuracy:.4f}, "
            f"F1: {metrics.f1:.4f}"
        )
        
        return model, metrics
    
    def _evaluate(
        self,
        model: keras.Model,
        data: ProcessedData,
        history: keras.callbacks.History
    ) -> TrainingMetrics:
        """Evaluate the trained model on all splits.
        
        Args:
            model: Trained Keras model.
            data: Preprocessed data splits.
            history: Training history.
        
        Returns:
            TrainingMetrics with all evaluation results.
        """
        logger.info("Evaluating model on test set...")
        
        y_pred = model.predict(data.X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        test_accuracy = accuracy_score(data.y_test, y_pred_classes)
        precision = precision_score(
            data.y_test, y_pred_classes, average="weighted", zero_division=0
        )
        recall = recall_score(
            data.y_test, y_pred_classes, average="weighted", zero_division=0
        )
        f1 = f1_score(
            data.y_test, y_pred_classes, average="weighted", zero_division=0
        )
        cm = confusion_matrix(data.y_test, y_pred_classes)
        
        report = classification_report(
            data.y_test, y_pred_classes, zero_division=0
        )
        logger.info(f"Classification Report:\n{report}")
        
        train_accuracy = history.history["accuracy"][-1]
        val_accuracy = history.history["val_accuracy"][-1]
        
        return TrainingMetrics(
            train_accuracy=float(train_accuracy),
            val_accuracy=float(val_accuracy),
            test_accuracy=float(test_accuracy),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            confusion_matrix=cm.tolist(),
            history={
                "loss": [float(x) for x in history.history["loss"]],
                "val_loss": [float(x) for x in history.history["val_loss"]],
                "accuracy": [float(x) for x in history.history["accuracy"]],
                "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
            }
        )
