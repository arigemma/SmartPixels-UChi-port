"""Sklearn model wrappers for signal/background classification."""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class SklearnModelWrapper:
    """Wrapper for sklearn models to handle 3D data."""

    def __init__(self, model, model_name: str = "sklearn_model"):
        """
        Args:
            model: Sklearn model instance
            model_name: Name for saving/loading
        """
        self.model = model
        self.model_name = model_name

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model.

        Args:
            X: Training data of shape (N, T, H, W) or (N, features)
            y: Training labels
        """
        # Flatten if needed
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        print(f"Training {self.model_name} on {X.shape[0]} samples with {X.shape[1]} features")
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Data of shape (N, T, H, W) or (N, features)

        Returns:
            Predicted labels
        """
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Data of shape (N, T, H, W) or (N, features)

        Returns:
            Predicted probabilities
        """
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict_proba(X)

    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Saved {self.model_name} to {path}")

    @classmethod
    def load(cls, path: str, model_name: Optional[str] = None):
        """Load model from disk."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        if model_name is None:
            model_name = Path(path).stem
        return cls(model, model_name)


def create_random_forest(
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: int = 1
) -> SklearnModelWrapper:
    """
    Create Random Forest classifier.

    Args:
        n_estimators: Number of trees
        max_depth: Maximum depth of trees
        min_samples_split: Minimum samples to split a node
        min_samples_leaf: Minimum samples at leaf node
        max_features: Number of features to consider for splits
        n_jobs: Number of parallel jobs
        random_state: Random seed
        verbose: Verbosity level

    Returns:
        Wrapped Random Forest model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose
    )
    return SklearnModelWrapper(model, model_name="random_forest")


def create_logistic_regression(
    C: float = 1.0,
    max_iter: int = 1000,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: int = 1
) -> SklearnModelWrapper:
    """
    Create Logistic Regression classifier.

    Args:
        C: Inverse of regularization strength
        max_iter: Maximum number of iterations
        n_jobs: Number of parallel jobs
        random_state: Random seed
        verbose: Verbosity level

    Returns:
        Wrapped Logistic Regression model
    """
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose
    )
    return SklearnModelWrapper(model, model_name="logistic_regression")


def create_svm(
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    probability: bool = True,
    max_iter: int = -1,
    verbose: bool = True,
    random_state: int = 42
) -> SklearnModelWrapper:
    """
    Create SVM classifier.

    Args:
        C: Regularization parameter
        kernel: Kernel type
        gamma: Kernel coefficient
        probability: Whether to enable probability estimates
        max_iter: Maximum iterations
        verbose: Verbosity
        random_state: Random seed

    Returns:
        Wrapped SVM model
    """
    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=probability,
        max_iter=max_iter,
        verbose=verbose,
        random_state=random_state
    )
    return SklearnModelWrapper(model, model_name="svm")
