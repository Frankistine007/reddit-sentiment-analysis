"""Fine-tuning wrapper for HuggingFace transformers.

One class handles bert-base-uncased, distilbert-base-uncased, roberta-base
(or any AutoModelForSequenceClassification checkpoint) — the model name is
just a constructor argument, which is exactly why the OOP design pays off.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.models.base_model import BaseSentimentModel

logger = logging.getLogger(__name__)


class _TextDataset(Dataset):
    """Tokenizes lazily inside __getitem__ — memory-friendly for 10k+ rows."""

    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = list(texts)
        self.labels = None if labels is None else np.asarray(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class TransformerModel(BaseSentimentModel):
    def __init__(self, cfg: dict, model_name: str, n_classes: int = 3) -> None:
        super().__init__(cfg)
        self.t = cfg["transformer"]
        self.model_name = model_name
        self.name = model_name.replace("/", "_")
        self.n_classes = n_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=n_classes
        ).to(self.device)

    # ------------------------------------------------------------------ #
    def _loader(self, texts, labels=None, shuffle=False) -> DataLoader:
        ds = _TextDataset(texts, labels, self.tokenizer, self.t["max_length"])
        return DataLoader(ds, batch_size=self.t["batch_size"], shuffle=shuffle)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
        train_loader = self._loader(X_train, y_train, shuffle=True)
        val_loader = self._loader(X_val, y_val) if X_val is not None else None

        epochs = self.t["epochs"]
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=float(self.t["lr"]), weight_decay=self.t["weight_decay"]
        )
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.t["warmup_ratio"]),
            num_training_steps=total_steps,
        )

        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val, patience_left = float("inf"), self.t["patience"]

        for epoch in range(1, epochs + 1):
            self.model.train()
            total = 0.0
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                out = self.model(**batch)
                out.loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total += out.loss.item() * batch["labels"].size(0)
            train_loss = total / len(train_loader.dataset)
            self.history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                logger.info(
                    "[%s] epoch %d  train=%.4f  val=%.4f  acc=%.4f",
                    self.name, epoch, train_loss, val_loss, val_acc,
                )
                if val_loss < best_val:
                    best_val, patience_left = val_loss, self.t["patience"]
                else:
                    patience_left -= 1
                    if patience_left == 0:
                        logger.info("[%s] early stopping at epoch %d", self.name, epoch)
                        break

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total, correct, n = 0.0, 0, 0
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(**batch)
            total += out.loss.item() * batch["labels"].size(0)
            correct += (out.logits.argmax(-1) == batch["labels"]).sum().item()
            n += batch["labels"].size(0)
        return total / n, correct / n

    @torch.no_grad()
    def predict(self, X) -> np.ndarray:
        self.model.eval()
        preds = []
        for batch in self._loader(X):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            preds.append(self.model(**batch).logits.argmax(-1).cpu())
        return torch.cat(preds).numpy()

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        self.model.eval()
        probs = []
        for batch in self._loader(X):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            probs.append(torch.softmax(self.model(**batch).logits, dim=-1).cpu())
        return torch.cat(probs).numpy()

    def save(self, directory: str | Path) -> None:
        out = Path(directory) / self.name
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)
