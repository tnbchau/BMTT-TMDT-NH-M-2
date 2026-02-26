from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


class BaseExperiment(ABC):
    """
    Base class for both ML and DL experiments
    """

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None
    ) -> None:
        """Train model"""
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Predict labels"""
        pass

    @abstractmethod
    def evaluate(self, X, y) -> Dict[str, Any]:
        """Evaluate model"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model"""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str):
        """Load model"""
        pass