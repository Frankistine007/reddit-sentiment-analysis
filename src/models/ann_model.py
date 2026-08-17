"""Feed-forward ANN (PyTorch) on top of TF-IDF features.

The bridge between traditional ML and transformers: same features as the
sklearn baselines, but a learned non-linear classifier with proper
training curves, dropout, and early stopping.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_model import BaseSentimentModel

logger = logging.getLogger(__name__)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], n_classes: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # noqa: D102
        return self.net(x)


class ANNModel(BaseSentimentModel):
    name = "ann_tfidf"

    def __init__(self, cfg: dict, n_classes: int = 3) -> None:
        super().__init__(cfg)
        self.a = cfg["ann"]
        t = cfg["traditional"]["tfidf"]
        self.n_classes = n_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vectorizer = TfidfVectorizer(
            max_features=t["max_features"],
            ngram_range=tuple(t["ngram_range"]),
            min_df=t["min_df"],
            sublinear_tf=True,
        )
        self.model: _MLP | None = None

    # ------------------------------------------------------------------ #
    def _loader(self, X, y=None, shuffle=False) -> DataLoader:
        X_t = torch.tensor(X.toarray(), dtype=torch.float32)
        tensors = (X_t,) if y is None else (X_t, torch.tensor(np.asarray(y), dtype=torch.long))
        return DataLoader(TensorDataset(*tensors), batch_size=self.a["batch_size"], shuffle=shuffle)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
        Xtr = self.vectorizer.fit_transform(list(X_train))
        self.model = _MLP(Xtr.shape[1], self.a["hidden_dims"], self.n_classes, self.a["dropout"]).to(self.device)

        train_loader = self._loader(Xtr, y_train, shuffle=True)
        val_loader = None
        if X_val is not None:
            val_loader = self._loader(self.vectorizer.transform(list(X_val)), y_val)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.a["lr"])

        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val, patience_left = float("inf"), self.a["patience"]
        best_state = None

        for epoch in range(1, self.a["epochs"] + 1):
            self.model.train()
            total = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(xb)
            train_loss = total / len(train_loader.dataset)
            self.history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader, criterion)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                logger.info("ANN epoch %d  train=%.4f  val=%.4f  acc=%.4f", epoch, train_loss, val_loss, val_acc)
                if val_loss < best_val:
                    best_val, patience_left = val_loss, self.a["patience"]
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_left -= 1
                    if patience_left == 0:
                        logger.info("Early stopping at epoch %d", epoch)
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, criterion) -> tuple[float, float]:
        self.model.eval()
        total, correct, n = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            logits = self.model(xb)
            total += criterion(logits, yb).item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(xb)
        return total / n, correct / n

    @torch.no_grad()
    def predict(self, X) -> np.ndarray:
        self.model.eval()
        loader = self._loader(self.vectorizer.transform(list(X)))
        preds = [self.model(xb[0].to(self.device)).argmax(1).cpu() for xb in loader]
        return torch.cat(preds).numpy()

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / f"{self.name}.pt")
        joblib.dump(self.vectorizer, directory / f"{self.name}_vectorizer.joblib")
