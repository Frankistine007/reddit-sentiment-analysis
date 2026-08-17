"""Abstract base class every model wrapper implements.

This is the core OOP contract: main.py never cares whether it's talking to
Logistic Regression or RoBERTa — every model exposes fit / predict / save.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseSentimentModel(ABC):
    """Common interface for all sentiment classifiers."""

    name: str = "base"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.history: dict[str, list[float]] = {}   # training curves (if any)

    # ------------------------------------------------------------------ #
    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
        """Train the model. X is raw text (list[str]) for all models —
        feature extraction is each model's own responsibility."""

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Return predicted class ids (np.ndarray of ints)."""

    @abstractmethod
    def save(self, directory: str | Path) -> None:
        """Persist model artefacts."""

    # ------------------------------------------------------------------ #
    def predict_proba(self, X) -> np.ndarray | None:
        """Optional; return None if the model can't produce probabilities."""
        return None
