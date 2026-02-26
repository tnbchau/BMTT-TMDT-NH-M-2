"""
DEEP LEARNING EXPERIMENT MODULE
- LSTM Architecture for Text Classification
- Supports Bidirectional LSTM, Batch Norm, Dropout
- Handles Class Imbalance automatically
- Includes Manual Hyperparameter Search
"""

import numpy as np
import pandas as pd
import os
import joblib
from typing import Dict, Any, Optional, List, Tuple
from itertools import product

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Embedding, LSTM, Bidirectional, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
from sklearn.utils import compute_class_weight

from .base_experiment import BaseExperiment


class LSTMExperiment(BaseExperiment):
    """
    LSTM-based Deep Learning Experiment.
    Automatically handles class imbalance via class_weight.
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        embedding_dim: int = 128,
        lstm_units: int = 128,
        dropout_rate: float = 0.3,
        learning_rate: float = 1e-3,
        use_bidirectional: bool = True, 
        use_batch_norm: bool = False,
        name: Optional[str] = None
    ):
        name = name or "dl_lstm"
        super().__init__(name=name)

        self.vocab_size = vocab_size
        self.max_len = max_len

        # Hyperparameters
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.use_bidirectional = use_bidirectional
        self.use_batch_norm = use_batch_norm

        self.model = self._build_model()
        self.history = None

    # ------------------ MODEL BUILDING ------------------

    def _build_model(self) -> tf.keras.Model:
        model = Sequential()
        
        # 1. Embedding Layer
        model.add(Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_len
        ))
        
        # 2. LSTM Layer
        lstm_layer = LSTM(self.lstm_units, return_sequences=False)
        
        if self.use_bidirectional:
            model.add(Bidirectional(lstm_layer))
        else:
            model.add(lstm_layer)
        
        # 3. Batch Normalization (Optional - helps convergence)
        if self.use_batch_norm:
            model.add(BatchNormalization())
        
        # 4. Dropout (Regularization)
        model.add(Dropout(self.dropout_rate))
        
        # 5. Output Layer (Binary Classification)
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    # ------------------ CORE TRAINING ------------------

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs: int = 20,
        batch_size: int = 64,
        patience: int = 3,
        save_best: bool = False,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the model with callbacks and class weights.
        """
        
        # 1. Calculate Class Weights (Handling Imbalance)
        # Nếu data có 90% nhãn 0 và 10% nhãn 1, class_weight sẽ tăng trọng số cho nhãn 1
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))
        
        print(f"[INFO] Training LSTM with class weights: {class_weight_dict}")

        # 2. Prepare Callbacks
        callbacks = self._get_callbacks(
            X_val=X_val,
            patience=patience,
            save_best=save_best,
            checkpoint_path=checkpoint_path
        )

        # 3. Fit Model
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict, # <-- KEY ADDITION
            verbose=1
        )

        self.is_trained = True
        
        # Summary
        hist = self.history.history
        summary = {
            "final_train_acc": hist["accuracy"][-1],
            "final_train_loss": hist["loss"][-1]
        }
        if X_val is not None:
            summary["final_val_acc"] = hist["val_accuracy"][-1]
            summary["final_val_loss"] = hist["val_loss"][-1]
            summary["best_val_f1"] = self._calculate_val_f1(X_val, y_val)

        return summary

    def _get_callbacks(self, X_val, patience, save_best, checkpoint_path) -> List:
        callbacks = []
        monitor = "val_loss" if X_val is not None else "loss"

        # Stop if no improvement
        callbacks.append(EarlyStopping(
            monitor=monitor, patience=patience, restore_best_weights=True, verbose=1
        ))

        # Reduce LR if stuck
        callbacks.append(ReduceLROnPlateau(
            monitor=monitor, factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ))

        # Checkpoint
        if save_best and checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            callbacks.append(ModelCheckpoint(
                filepath=checkpoint_path, monitor=monitor, save_best_only=True, verbose=1
            ))
        
        return callbacks

    def _calculate_val_f1(self, X_val, y_val):
        """Helper to get F1 score on validation set specifically"""
        y_pred = self.predict(X_val)
        return f1_score(y_val, y_pred)

    # ------------------ PREDICTION ------------------

    def predict(self, X) -> np.ndarray:
        self._check_trained()
        y_prob = self.model.predict(X, verbose=0).ravel()
        return (y_prob >= 0.5).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        self._check_trained()
        return self.model.predict(X, verbose=0).ravel()

    def evaluate(self, X, y) -> Dict[str, Any]:
        self._check_trained()
        y_prob = self.predict_proba(X)
        y_pred = (y_prob >= 0.5).astype(int)

        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_prob),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist()
        }

    # ------------------ SAVE / LOAD ------------------

    def save(self, path: str) -> None:
        """Saves model + metadata separately"""
        if not os.path.exists(path):
            os.makedirs(path)

        # 1. Save Keras Model
        model_path = os.path.join(path, "lstm_model.keras")
        self.model.save(model_path)

        # 2. Save Metadata
        meta = {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "max_len": self.max_len,
            "embedding_dim": self.embedding_dim,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "use_bidirectional": self.use_bidirectional,
            "use_batch_norm": self.use_batch_norm,
            "is_trained": self.is_trained,
            "experiment_class": "LSTMExperiment"
        }
        joblib.dump(meta, os.path.join(path, "meta.pkl"))
        print(f"[INFO] Deep Learning model saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Reconstructs experiment from disk"""
        meta_path = os.path.join(path, "meta.pkl")
        model_path = os.path.join(path, "lstm_model.keras")

        if not os.path.exists(meta_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model files in {path}")

        meta = joblib.load(meta_path)
        
        # Init blank experiment
        exp = cls(
            vocab_size=meta["vocab_size"],
            max_len=meta["max_len"],
            embedding_dim=meta["embedding_dim"],
            lstm_units=meta["lstm_units"],
            dropout_rate=meta["dropout_rate"],
            learning_rate=meta["learning_rate"],
            use_bidirectional=meta.get("use_bidirectional", False),
            use_batch_norm=meta.get("use_batch_norm", False),
            name=meta["name"]
        )
        
        # Load weights
        exp.model = load_model(model_path)
        exp.is_trained = meta["is_trained"]
        
        print(f"[INFO] Deep Learning model loaded from {path}")
        return exp

    def _check_trained(self):
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")


# ======================================================
# HELPER: MANUAL TUNING FUNCTION
# ======================================================

def manual_dl_search(
    X_train, y_train, X_val, y_val,
    vocab_size: int,
    max_len: int,
    param_grid: Dict[str, List],
    epochs: int = 10
) -> Dict[str, Any]:
    """
    Runs a grid search for LSTM hyperparameters.
    """
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    print(f"[INFO] Starting Manual Search. Total combinations: {len(combinations)}")
    
    best_score = -1
    best_params = None
    history_log = []

    for i, params in enumerate(combinations):
        print(f"\n--- Combo {i+1}/{len(combinations)}: {params} ---")
        
        try:
            exp = LSTMExperiment(
                vocab_size=vocab_size,
                max_len=max_len,
                **params
            )
            
            # Train ngắn hạn để test
            exp.train(
                X_train, y_train, X_val, y_val,
                epochs=epochs,
                patience=2, # Dừng sớm nếu không tốt
                save_best=False
            )
            
            # Đánh giá bằng F1 Score
            metrics = exp.evaluate(X_val, y_val)
            score = metrics["f1"]
            
            print(f"   >>> Val F1: {score:.4f} | Acc: {metrics['accuracy']:.4f}")
            
            history_log.append({
                "params": params,
                "f1": score,
                "accuracy": metrics["accuracy"],
                "roc_auc": metrics.get("roc_auc", 0)
            })
            
            if score > best_score:
                best_score = score
                best_params = params
                print("   >>> NEW BEST MODEL FOUND!")
                
        except Exception as e:
            print(f"   [ERROR] Failed with params {params}: {str(e)}")

    print(f"\n[RESULT] Best F1: {best_score:.4f}")
    print(f"[RESULT] Best Params: {best_params}")

    return {
        "best_params": best_params,
        "best_score": best_score,
        "all_results": history_log
    }