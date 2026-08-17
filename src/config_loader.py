"""Singleton config loader — one source of truth for all modules."""
from __future__ import annotations

import yaml
from pathlib import Path


class Config:
    """Loads config.yaml once and exposes it with attribute/dict access.

    Usage:
        cfg = Config.load()
        cfg["transformer"]["models"]
    """

    _instance: "Config | None" = None

    def __init__(self, path: str | Path = "config/config.yaml") -> None:
        self.path = Path(path)
        with open(self.path, "r", encoding="utf-8") as f:
            self._cfg: dict = yaml.safe_load(f)

    # ---------- singleton access ----------
    @classmethod
    def load(cls, path: str | Path = "config/config.yaml") -> "Config":
        if cls._instance is None:
            cls._instance = cls(path)
        return cls._instance

    # ---------- dict-like access ----------
    def __getitem__(self, key: str):
        return self._cfg[key]

    def get(self, key: str, default=None):
        return self._cfg.get(key, default)

    @property
    def raw(self) -> dict:
        return self._cfg
