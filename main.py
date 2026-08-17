"""End-to-end pipeline orchestrator.

Usage:
    python main.py                          # auto data source, all model stages
    python main.py --source premade         # force premade Kaggle CSV
    python main.py --source reddit          # force live PRAW scrape
    python main.py --stages traditional ann # skip transformers (fast run)
    python main.py --sample 5000            # subsample for quick experiments
"""
from __future__ import annotations

import argparse
import logging

from src.config_loader import Config
from src.data.dataset_loader import DatasetLoader
from src.data.preprocessor import TextPreprocessor
from src.evaluation.evaluator import Evaluator
from src.models.ann_model import ANNModel
from src.models.traditional_models import (
    LogisticRegressionModel,
    RandomForestModel,
    SVMModel,
)
from src.models.transformer_models import TransformerModel
from src.visualization.plotter import Plotter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("pipeline")


class SentimentPipeline:
    """Composes all components; each stage is a method (easy to run/skip)."""

    def __init__(self, source: str, stages: list[str], sample: int | None) -> None:
        self.cfg = Config.load().raw
        self.source = source
        self.stages = stages
        self.sample = sample
        self.plotter = Plotter(self.cfg)
        self.evaluator = Evaluator(self.cfg)
        self.models_dir = self.cfg["output"]["models_dir"]

    # -------------------------- stage 1: data ------------------------- #
    def prepare_data(self) -> None:
        df = DatasetLoader(self.cfg).load(self.source)
        if self.sample:
            df = df.sample(min(self.sample, len(df)), random_state=42)

        self.plotter.class_distribution(df, "raw")
        pre = TextPreprocessor(self.cfg)
        df = pre.run(df)
        self.plotter.class_distribution(df, "processed")
        self.plotter.text_length_distribution(df)

        train, val, test = pre.split(df)
        self.X_train, self.y_train = train["text"], train["label"]
        self.X_val, self.y_val = val["text"], val["label"]
        self.X_test, self.y_test = test["text"], test["label"]

    # --------------------- stage 2: run one model --------------------- #
    def _run(self, model) -> None:
        logger.info("=== Training %s ===", model.name)
        model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
        self.evaluator.evaluate(model, self.X_test, self.y_test)
        self.plotter.confusion_matrix(self.evaluator.confusions[model.name], model.name)
        if model.history:
            self.plotter.training_curves(model.history, model.name)
        model.save(self.models_dir)

    # ------------------------- stage runners --------------------------- #
    def run_traditional(self) -> None:
        for cls in (LogisticRegressionModel, SVMModel, RandomForestModel):
            self._run(cls(self.cfg))

    def run_ann(self) -> None:
        self._run(ANNModel(self.cfg))

    def run_transformers(self) -> None:
        for name in self.cfg["transformer"]["models"]:
            self._run(TransformerModel(self.cfg, name))

    # ----------------------------- main -------------------------------- #
    def run(self) -> None:
        self.prepare_data()
        if "traditional" in self.stages:
            self.run_traditional()
        if "ann" in self.stages:
            self.run_ann()
        if "transformer" in self.stages:
            self.run_transformers()

        df = self.evaluator.comparison_table()
        self.plotter.model_comparison(df)
        logger.info("Done. Plots → outputs/plots, reports → outputs/reports")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Reddit sentiment classification pipeline")
    p.add_argument("--source", default="auto", choices=["auto", "reddit", "premade", "raw_csv"])
    p.add_argument("--stages", nargs="+", default=["traditional", "ann", "transformer"],
                   choices=["traditional", "ann", "transformer"])
    p.add_argument("--sample", type=int, default=None, help="subsample N rows for quick runs")
    args = p.parse_args()

    SentimentPipeline(args.source, args.stages, args.sample).run()
