"""Load and validate config.yaml into a typed, read-only object.

See Prompt.md - Phase 1. Every other module should accept a ``cfg`` argument
produced by :func:`load_config` rather than reading YAML or env vars directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class Config:
    """Thin attribute/dict wrapper around the parsed YAML.

    TODO(Phase 1): replace with a pydantic ``BaseModel`` (or frozen dataclass
    tree) that fails loudly on unknown keys and coerces types.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, item: str) -> Any:
        try:
            value = self._data[item]
        except KeyError as exc:  # pragma: no cover - placeholder behaviour
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def __getitem__(self, item: str) -> Any:
        return self._data[item]

    def get(self, item: str, default: Any = None) -> Any:
        return self._data.get(item, default)

    def to_dict(self) -> dict[str, Any]:
        return self._data


def load_config(path: str | Path | None = None) -> Config:
    """Read ``config.yaml`` (or *path*) and return a :class:`Config`."""
    cfg_path = Path(path) if path else _DEFAULT_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw)
