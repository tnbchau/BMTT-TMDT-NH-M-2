"""
UNIFIED FEATURE EXTRACTION MODULE
- Same base features for ML models
- Model-specific feature adaptation
- Improved adapters and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional

from scipy.sparse import hstack, csr_matrix, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib

# ======================================================
# CONSTANTS
# ======================================================
DEFAULT_TEXT_COL = "cleaned_text"
DEFAULT_NUMERIC_COLS = ["num_urls", "num_ip_urls"]
DEFAULT_BINARY_COLS = ["has_url", "has_ip_url"]


# ======================================================
# FEATURE EXTRACTOR
# ======================================================

class FeatureExtractor:
    """
    Extracts base features shared across all models.
    
    Features:
    - Text features: TF-IDF vectors from cleaned text
    - Numeric features: URL counts (scaled)
    - Binary features: URL presence indicators
    """

    def __init__(
        self,
        text_col: str = DEFAULT_TEXT_COL,
        numeric_cols: Optional[List[str]] = None,
        binary_cols: Optional[List[str]] = None,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        scale_numeric: bool = True
    ):
        """
        Initialize feature extractor
        
        Args:
            text_col: Column name for text data
            numeric_cols: List of numeric feature columns
            binary_cols: List of binary feature columns
            max_features: Max vocabulary size for TF-IDF
            ngram_range: N-gram range for TF-IDF (default: unigrams + bigrams)
            scale_numeric: Whether to standardize numeric features
        """
        self.text_col = text_col
        self.numeric_cols = numeric_cols or DEFAULT_NUMERIC_COLS.copy()
        self.binary_cols = binary_cols or DEFAULT_BINARY_COLS.copy()

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            stop_words="english",
            sublinear_tf=True,  # Use log scaling for term frequency
            min_df=2  # Ignore very rare terms
        )

        self.scaler = StandardScaler() if scale_numeric else None
        self.is_fitted = False

    # -------------------------
    # FIT / TRANSFORM
    # -------------------------

    def fit(self, X: pd.DataFrame) -> "FeatureExtractor":
        """
        Fit the feature extractor on training data
        
        Args:
            X: DataFrame with text and feature columns
            
        Returns:
            self (for chaining)
        """
        self._validate_columns(X)
        
        # Fit TF-IDF vectorizer
        self.vectorizer.fit(X[self.text_col])

        # Fit scaler on numeric features
        if self.scaler and self.numeric_cols:
            self.scaler.fit(X[self.numeric_cols].values)

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> Dict[str, Union[csr_matrix, np.ndarray, List[str]]]:
        """
        Transform data into feature matrices
        
        Args:
            X: DataFrame with text and feature columns
            
        Returns:
            Dictionary containing:
            - "text": TF-IDF text features (sparse)
            - "numeric": Scaled numeric features (sparse)
            - "binary": Binary features (sparse)
            - "combined": All features concatenated (sparse)
            - "text_feature_names": List of text feature names
            - "numeric_feature_names": List of numeric feature names
            - "binary_feature_names": List of binary feature names
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureExtractor must be fitted before transform")
        
        self._validate_columns(X)

        # ===== TEXT FEATURES =====
        X_text = self.vectorizer.transform(X[self.text_col])

        # ===== NUMERIC FEATURES =====
        if self.numeric_cols:
            X_num = X[self.numeric_cols].values.astype(np.float32)
            if self.scaler:
                X_num = self.scaler.transform(X_num)
            X_num = csr_matrix(X_num)
        else:
            X_num = csr_matrix((len(X), 0))

        # ===== BINARY FEATURES =====
        if self.binary_cols:
            X_bin = csr_matrix(X[self.binary_cols].values.astype(np.float32))
        else:
            X_bin = csr_matrix((len(X), 0))

        # ===== COMBINED FEATURES =====
        X_combined = hstack([X_text, X_num, X_bin], format='csr')

        return {
            "text": X_text,
            "numeric": X_num,
            "binary": X_bin,
            "combined": X_combined,
            "text_feature_names": self.vectorizer.get_feature_names_out().tolist(),
            "numeric_feature_names": self.numeric_cols,
            "binary_feature_names": self.binary_cols
        }

    def fit_transform(self, X: pd.DataFrame) -> Dict[str, Union[csr_matrix, np.ndarray, List[str]]]:
        """
        Fit and transform in one step
        
        Args:
            X: DataFrame with text and feature columns
            
        Returns:
            Feature dictionary (same as transform())
        """
        return self.fit(X).transform(X)

    # -------------------------
    # VALIDATION
    # -------------------------

    def _validate_columns(self, X: pd.DataFrame) -> None:
        """
        Validate that required columns exist in dataframe
        
        Args:
            X: DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = [self.text_col] + self.numeric_cols + self.binary_cols
        missing_cols = set(required_cols) - set(X.columns)
        
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}\n"
                f"Available columns: {list(X.columns)}"
            )
        
        if X[self.text_col].isna().any():
            raise ValueError(f"Text column '{self.text_col}' contains null values")

    # -------------------------
    # UTILITIES
    # -------------------------

    def get_feature_names(self) -> List[str]:
        """
        Get all feature names in order
        
        Returns:
            List of feature names [text features, numeric features, binary features]
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureExtractor must be fitted first")
        
        return (
            self.vectorizer.get_feature_names_out().tolist()
            + self.numeric_cols
            + self.binary_cols
        )

    def get_feature_counts(self) -> Dict[str, int]:
        """
        Get count of each feature type
        
        Returns:
            Dictionary with counts for each feature type
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureExtractor must be fitted first")
        
        return {
            "text": len(self.vectorizer.get_feature_names_out()),
            "numeric": len(self.numeric_cols),
            "binary": len(self.binary_cols),
            "total": len(self.get_feature_names())
        }

    def transform_with_labels(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Union[Dict, Tuple[Dict, np.ndarray]]:
        """
        Transform features and optionally return labels
        
        Args:
            X: DataFrame with features
            y: Optional labels
            
        Returns:
            features dict, or (features dict, labels array) if y provided
        """
        features = self.transform(X)
        if y is not None:
            return features, y.values
        return features

    # -------------------------
    # SAVE / LOAD
    # -------------------------

    def save(self, path: str) -> None:
        """
        Save feature extractor to disk
        
        Args:
            path: File path to save to
        """
        joblib.dump(self, path)
        print(f"[INFO] FeatureExtractor saved to {path}")

    @staticmethod
    def load(path: str) -> "FeatureExtractor":
        """
        Load feature extractor from disk
        
        Args:
            path: File path to load from
            
        Returns:
            Loaded FeatureExtractor instance
        """
        extractor = joblib.load(path)
        print(f"[INFO] FeatureExtractor loaded from {path}")
        return extractor


# ======================================================
# FEATURE ADAPTER (MODEL-SPECIFIC)
# ======================================================

class FeatureAdapter:
    """
    Adapts base features for specific model types.
    
    Different models perform better with different feature combinations:
    - SVM: Text + Binary (handles high-dimensional sparse features well)
    - Naive Bayes: Text + Binary (requires non-negative features)
    - Random Forest: All features (handles mixed feature types)
    - Logistic Regression: All features (handles mixed feature types)
    """

    @staticmethod
    def for_svm(features: Dict) -> csr_matrix:
        """
        Adapt features for SVM
        
        Uses text TF-IDF + binary indicators
        (Numeric features can be redundant with binary indicators for SVM)
        
        Args:
            features: Feature dictionary from FeatureExtractor
            
        Returns:
            Sparse feature matrix
        """
        return hstack([features["text"], features["binary"]], format='csr')

    @staticmethod
    def for_naive_bayes(features: Dict) -> csr_matrix:
        """
        Adapt features for Multinomial Naive Bayes
        
        Uses TF-IDF text features + binary indicators
        (Both are non-negative, which MultinomialNB requires)
        
        Args:
            features: Feature dictionary from FeatureExtractor
            
        Returns:
            Sparse feature matrix
        """
        return hstack([features["text"], features["binary"]], format='csr')

    @staticmethod
    def for_random_forest(features: Dict) -> csr_matrix:
        """
        Adapt features for Random Forest
        
        Uses all features - RF handles mixed types well
        
        Args:
            features: Feature dictionary from FeatureExtractor
            
        Returns:
            Sparse feature matrix with all features
        """
        # Random Forest benefits from all feature types
        return features["combined"]

    @staticmethod
    def for_logistic(features: Dict) -> csr_matrix:
        """
        Adapt features for Logistic Regression
        
        Uses all features - LR handles mixed types well
        
        Args:
            features: Feature dictionary from FeatureExtractor
            
        Returns:
            Sparse feature matrix with all features
        """
        return features["combined"]

    # -------------------------
    # HELPERS
    # -------------------------

    @staticmethod
    def get_all_adaptations(features: Dict) -> Dict[str, csr_matrix]:
        """
        Generate all model-specific adaptations at once
        
        Useful for comparing different models on same data
        
        Args:
            features: Feature dictionary from FeatureExtractor
            
        Returns:
            Dictionary mapping model types to adapted features
        """
        return {
            "svm": FeatureAdapter.for_svm(features),
            "naive_bayes": FeatureAdapter.for_naive_bayes(features),
            "random_forest": FeatureAdapter.for_random_forest(features),
            "logistic": FeatureAdapter.for_logistic(features),
            "combined": features["combined"]  # Original combined features
        }

    @staticmethod
    def adapt_for_model(features: Dict, model_type: str) -> csr_matrix:
        """
        Adapt features for a specific model type
        
        Args:
            features: Feature dictionary from FeatureExtractor
            model_type: One of ["svm", "naive_bayes", "random_forest", "logistic"]
            
        Returns:
            Adapted sparse feature matrix
            
        Raises:
            ValueError: If model_type is unknown
        """
        adapters = {
            "svm": FeatureAdapter.for_svm,
            "naive_bayes": FeatureAdapter.for_naive_bayes,
            "random_forest": FeatureAdapter.for_random_forest,
            "logistic": FeatureAdapter.for_logistic,
        }

        if model_type not in adapters:
            raise ValueError(
                f"Unknown model type: {model_type}\n"
                f"Supported types: {list(adapters.keys())}"
            )

        return adapters[model_type](features)

    @staticmethod
    def get_feature_info(features: Dict) -> Dict[str, Dict[str, int]]:
        """
        Get information about feature dimensions for each model type
        
        Args:
            features: Feature dictionary from FeatureExtractor
            
        Returns:
            Dictionary with shape info for each model adaptation
        """
        adaptations = FeatureAdapter.get_all_adaptations(features)
        
        return {
            model_type: {
                "n_samples": matrix.shape[0],
                "n_features": matrix.shape[1],
                "sparsity": 1 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
            }
            for model_type, matrix in adaptations.items()
        }


# ======================================================
# QUICK TEST
# ======================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FEATURE EXTRACTION TEST")
    print("=" * 60)
    
    # Create sample data
    df = pd.DataFrame({
        "cleaned_text": [
            "verify account now click here",
            "meeting tomorrow at office",
            "urgent password update required immediately",
            "monthly newsletter update"
        ],
        "num_urls": [2, 0, 1, 0],
        "num_ip_urls": [1, 0, 0, 0],
        "has_url": [1, 0, 1, 0],
        "has_ip_url": [1, 0, 0, 0],
        "label": [1, 0, 1, 0]
    })

    print("\n[INFO] Sample data:")
    print(df)

    # Initialize extractor
    print("\n[INFO] Initializing FeatureExtractor...")
    extractor = FeatureExtractor(max_features=100, ngram_range=(1, 2))

    # Fit and transform
    print("\n[INFO] Fitting and transforming...")
    features = extractor.fit_transform(df)

    # Display feature counts
    print("\n[INFO] Feature counts:")
    for name, count in extractor.get_feature_counts().items():
        print(f"  {name}: {count}")

    # Get all adaptations
    print("\n[INFO] Generating model-specific adaptations...")
    adapted = FeatureAdapter.get_all_adaptations(features)

    print("\n[INFO] Adapted feature shapes:")
    for model_type, matrix in adapted.items():
        print(f"  {model_type:15s}: {matrix.shape}")

    # Feature info
    print("\n[INFO] Detailed feature information:")
    info = FeatureAdapter.get_feature_info(features)
    for model_type, details in info.items():
        print(f"\n  {model_type}:")
        for key, value in details.items():
            if key == "sparsity":
                print(f"    {key}: {value:.2%}")
            else:
                print(f"    {key}: {value}")
