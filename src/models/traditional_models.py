"""Traditional ML baselines: Logistic Regression, Linear SVM, Random Forest.

Each pairs a TF-IDF vectorizer with a scikit-learn classifier inside a
Pipeline, so `fit`/`predict` take raw text just like the deep models do.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.models.base_model import BaseSentimentModel


class TraditionalModel(BaseSentimentModel):
    """Generic TF-IDF + sklearn classifier wrapper."""

    def __init__(self, cfg: dict, estimator, name: str) -> None:
        super().__init__(cfg)
        self.name = name
        t = cfg["traditional"]["tfidf"]
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=t["max_features"],
                        ngram_range=tuple(t["ngram_range"]),
                        min_df=t["min_df"],
                        sublinear_tf=True,
                    ),
                ),
                ("clf", estimator),
            ]
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
        self.pipeline.fit(list(X_train), list(y_train))

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.pipeline.predict(list(X)))

    def predict_proba(self, X):
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            return self.pipeline.predict_proba(list(X))
        return None

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, directory / f"{self.name}.joblib")


# ---------------------------------------------------------------------- #
#  Concrete models — thin subclasses, each an interview talking point
# ---------------------------------------------------------------------- #
class LogisticRegressionModel(TraditionalModel):
    def __init__(self, cfg: dict) -> None:
        super().__init__(
            cfg,
            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
            name="logistic_regression",
        )


class SVMModel(TraditionalModel):
    def __init__(self, cfg: dict) -> None:
        # LinearSVC scales to 20k TF-IDF features far better than kernel SVC
        super().__init__(
            cfg,
            LinearSVC(C=0.5, class_weight="balanced"),
            name="linear_svm",
        )


class RandomForestModel(TraditionalModel):
    def __init__(self, cfg: dict) -> None:
        super().__init__(
            cfg,
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                n_jobs=-1,
                class_weight="balanced",
                random_state=cfg["split"]["random_state"],
            ),
            name="random_forest",
        )
