"""Evaluation: metrics per model + a comparison table across all models."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.models.base_model import BaseSentimentModel

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, cfg: dict) -> None:
        self.labels = [cfg["labels"][k] for k in sorted(cfg["labels"])]
        self.reports_dir = Path(cfg["output"]["reports_dir"])
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[dict] = []
        self.confusions: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    def evaluate(self, model: BaseSentimentModel, X_test, y_test) -> dict:
        t0 = time.time()
        y_pred = model.predict(X_test)
        infer_time = time.time() - t0

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(
            y_test, y_pred, target_names=self.labels, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)
        self.confusions[model.name] = cm

        result = {
            "model": model.name,
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "inference_sec": round(infer_time, 2),
        }
        self.results.append(result)
        logger.info("%s → acc=%.4f  f1_macro=%.4f", model.name, acc, f1_macro)

        with open(self.reports_dir / f"{model.name}_report.json", "w") as f:
            json.dump({"summary": result, "per_class": report}, f, indent=2)
        return result

    # ------------------------------------------------------------------ #
    def comparison_table(self) -> pd.DataFrame:
        df = pd.DataFrame(self.results).sort_values("accuracy", ascending=False)
        df.to_csv(self.reports_dir / "model_comparison.csv", index=False)
        logger.info("\n%s", df.to_string(index=False))
        return df
