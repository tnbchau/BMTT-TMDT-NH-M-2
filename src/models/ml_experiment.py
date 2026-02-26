"""
ML EXPERIMENT MODULE
- Wraps Scikit-Learn models
- Handles Hyperparameter Tuning (Grid/Random Search)
- Manages Training, Evaluation, and Persistence
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
import joblib

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, make_scorer
)

from .base_experiment import BaseExperiment

class MLExperiment(BaseExperiment):
    """
    Generic ML experiment with hyperparameter tuning support.
    Designed for Phishing Detection (imbalanced data aware).
    """

    MODEL_REGISTRY = {
        "svm": (
            SVC,
            dict(
                kernel="linear",
                probability=True,       
                class_weight="balanced", 
                random_state=42
            )
        ),
        "naive_bayes": (
            MultinomialNB,
            dict(alpha=1.0)
        ),
        "random_forest": (
            RandomForestClassifier,
            dict(
                n_estimators=100,
                max_depth=20,
                class_weight="balanced",
                n_jobs=-1,              
                random_state=42
            )
        ),
        "logistic": (
            LogisticRegression,
            dict(
                solver="liblinear",     
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
        )
    }

    TUNING_GRIDS = {
        "svm": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear", "rbf"] 
        },
        "naive_bayes": {
            "alpha": [0.01, 0.1, 0.5, 1.0, 2.0]
        },
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        },
        "logistic": {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"]
        }
    }

    def __init__(
        self, 
        model_type: str, 
        name: Optional[str] = None,
        use_tuning: bool = False,
        tuning_method: str = "grid", 
        cv_folds: int = 5,            
        model_params: Optional[Dict] = None
    ):
        if model_type not in self.MODEL_REGISTRY:
            raise ValueError(f"Unsupported model_type: {model_type}. Supported: {list(self.MODEL_REGISTRY.keys())}")

        self.model_type = model_type
        self.use_tuning = use_tuning
        self.tuning_method = tuning_method
        self.cv_folds = cv_folds
        
        name = name or f"ml_{model_type}{'_tuned' if use_tuning else ''}"
        super().__init__(name=name)

        model_cls, default_params = self.MODEL_REGISTRY[model_type]
        if model_params:
            self.base_params = {**default_params, **model_params}
        else:
            self.base_params = default_params.copy()
        self.base_model_cls = model_cls
        self.model = model_cls(**self.base_params)
        
        self.best_params = None
        self.training_history = {}

    # ------------------ CORE TRAINING ------------------

    def train(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Any]:
        """Train the model, optionally with hyperparameter tuning."""
        
        if self.use_tuning:
            self._train_with_tuning(X_train, y_train)
        else:
            print(f"[INFO] Training {self.model_type} with default params...")
            self.model.fit(X_train, y_train)
        
        self.is_trained = True

        train_score = self.model.score(X_train, y_train)
        self.training_history["train_accuracy"] = train_score
        print(f"[RESULT] Train Accuracy: {train_score:.4f}")

        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            self.training_history["val_accuracy"] = val_score
            print(f"[RESULT] Val Accuracy:   {val_score:.4f}")
        
        if self.best_params:
            self.training_history["best_params"] = self.best_params

        return self.training_history

    def _train_with_tuning(self, X_train, y_train):
        """Internal method to run Grid/Random Search."""
        
        param_grid = self.TUNING_GRIDS.get(self.model_type, {})
        if not param_grid:
            print(f"[WARNING] No tuning grid found for {self.model_type}. Training with defaults.")
            self.model.fit(X_train, y_train)
            return

        scorer = make_scorer(f1_score, average='binary')

        print(f"[INFO] Starting {self.tuning_method.upper()} search for {self.model_type}...")
        print(f"[INFO] Grid: {param_grid}")

        if self.tuning_method == "grid":
            search = GridSearchCV(
                estimator=self.base_model_cls(**self.base_params),
                param_grid=param_grid,
                cv=self.cv_folds,
                scoring=scorer,
                n_jobs=-1,
                verbose=1
            )
        else:  # random
            search = RandomizedSearchCV(
                estimator=self.base_model_cls(**self.base_params),
                param_distributions=param_grid,
                n_iter=15,
                cv=self.cv_folds,
                scoring=scorer,
                n_jobs=-1,
                random_state=42,
                verbose=1
            )

        search.fit(X_train, y_train)

        self.model = search.best_estimator_
        self.best_params = search.best_params_
        
        self.training_history["best_cv_f1_score"] = search.best_score_
        
        print(f"[SUCCESS] Tuning complete.")
        print(f"   Best Params: {self.best_params}")
        print(f"   Best CV F1:  {search.best_score_:.4f}")

    # ------------------ PREDICTION & EVALUATION ------------------

    def predict(self, X) -> np.ndarray:
        self._check_trained()
        return self.model.predict(X)

    def predict_proba(self, X) -> Optional[np.ndarray]:
        self._check_trained()
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def evaluate(self, X, y) -> Dict[str, Any]:
        """Comprehensive evaluation metrics."""
        self._check_trained()

        y_pred = self.predict(X)
        
        # Basic Metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist()
        }

        # ROC-AUC (Only if probabilities available)
        y_proba = self.predict_proba(X)
        if y_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y, y_proba[:, 1])
            except ValueError:
                metrics["roc_auc"] = None

        if self.best_params:
            metrics["best_params"] = self.best_params

        return metrics

    # ------------------ SAVE / LOAD ------------------

    def save(self, path: str) -> None:
        """Saves the entire experiment state."""
        state = {
            "model": self.model,
            "model_type": self.model_type,
            "name": self.name,
            "use_tuning": self.use_tuning,
            "best_params": self.best_params,
            "training_history": self.training_history,
            "is_trained": self.is_trained,
            "base_params": self.base_params, 
            "experiment_class": "MLExperiment"
        }
        joblib.dump(state, path)
        print(f"[INFO] Model saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Loads a saved experiment."""
        data = joblib.load(path)
        
        # Reconstruct object
        exp = cls(
            model_type=data["model_type"], 
            name=data["name"],
            use_tuning=data.get("use_tuning", False)
        )
        
        exp.model = data["model"]
        exp.best_params = data.get("best_params")
        exp.training_history = data.get("training_history", {})
        exp.is_trained = data.get("is_trained", False)
        exp.base_params = data.get("base_params", {})
        
        print(f"[INFO] Model loaded from {path}")
        return exp

    def _check_trained(self):
        if not self.is_trained:
            raise RuntimeError(f"Model {self.name} is not trained yet. Call .train() first.")