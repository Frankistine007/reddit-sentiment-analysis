"""All Matplotlib plots: class distribution, text lengths, training curves,
confusion matrices, and the final model-comparison bar chart."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})


class Plotter:
    def __init__(self, cfg: dict) -> None:
        self.dir = Path(cfg["output"]["plots_dir"])
        self.dir.mkdir(parents=True, exist_ok=True)
        self.labels = [cfg["labels"][k] for k in sorted(cfg["labels"])]

    def _save(self, fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(self.dir / f"{name}.png", bbox_inches="tight")
        plt.close(fig)

    # -------------------------- data analysis ------------------------- #
    def class_distribution(self, df: pd.DataFrame, title_suffix: str = "") -> None:
        counts = df["label"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar([self.labels[i] for i in counts.index], counts.values,
                      color=["#d9534f", "#f0ad4e", "#5cb85c"])
        ax.bar_label(bars)
        ax.set_ylabel("Samples")
        ax.set_title(f"Class Distribution {title_suffix}".strip())
        self._save(fig, f"class_distribution{('_' + title_suffix) if title_suffix else ''}".replace(" ", "_").lower())

    def text_length_distribution(self, df: pd.DataFrame) -> None:
        lengths = df["text"].str.split().str.len()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(lengths, bins=50, color="#5bc0de", edgecolor="black", alpha=0.8)
        ax.axvline(lengths.median(), color="red", linestyle="--",
                   label=f"median = {lengths.median():.0f}")
        ax.set_xlabel("Words per sample")
        ax.set_ylabel("Frequency")
        ax.set_title("Text Length Distribution")
        ax.legend()
        self._save(fig, "text_length_distribution")

    # ------------------------- training curves ------------------------ #
    def training_curves(self, history: dict, model_name: str) -> None:
        if not history.get("train_loss"):
            return
        epochs = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(epochs, history["train_loss"], "o-", label="train loss")
        if history.get("val_loss"):
            axes[0].plot(epochs, history["val_loss"], "s-", label="val loss")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title(f"{model_name} — Loss"); axes[0].legend()

        if history.get("val_acc"):
            axes[1].plot(epochs, history["val_acc"], "d-", color="#5cb85c", label="val accuracy")
            axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
            axes[1].set_title(f"{model_name} — Validation Accuracy"); axes[1].legend()
        self._save(fig, f"curves_{model_name}")

    # ------------------------ confusion matrix ------------------------ #
    def confusion_matrix(self, cm: np.ndarray, model_name: str) -> None:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(self.labels)), self.labels)
        ax.set_yticks(range(len(self.labels)), self.labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {model_name}")
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.grid(False)
        self._save(fig, f"confusion_{model_name}")

    # ------------------------ model comparison ------------------------ #
    def model_comparison(self, results_df: pd.DataFrame) -> None:
        df = results_df.sort_values("accuracy")
        x = np.arange(len(df))
        width = 0.35
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(x - width / 2, df["accuracy"], height=width, label="Accuracy", color="#337ab7")
        ax.barh(x + width / 2, df["f1_macro"], height=width, label="F1 (macro)", color="#f0ad4e")
        ax.set_yticks(x, df["model"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score")
        ax.set_title("Model Comparison: Traditional ML → ANN → Transformers")
        for i, (acc, f1) in enumerate(zip(df["accuracy"], df["f1_macro"])):
            ax.text(acc + 0.005, i - width / 2, f"{acc:.3f}", va="center", fontsize=8)
            ax.text(f1 + 0.005, i + width / 2, f"{f1:.3f}", va="center", fontsize=8)
        ax.legend(loc="lower right")
        self._save(fig, "model_comparison")
